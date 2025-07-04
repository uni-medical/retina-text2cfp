# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Lumina-T2I using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import partial
import json
import logging
import os
import random
import socket
from time import time
from PIL import Image
from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import ItemProcessor, MyDataset, read_general2
from grad_norm import calculate_l2_grad_norm, get_model_parallel_dim_dict, scale_grad
from imgproc import generate_crop_size_list, var_center_crop
import models
from parallel import distributed_init, get_intra_node_process_group
from transport import create_transport, Sampler
from tqdm import tqdm 
from torchvision.transforms import ToPILImage

#############################################################################
#                            Data item Processor                            #
#############################################################################


class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform):
        self.image_transform = transform
    
    def process_item(self, data_item, training_mode=False):
        try:
            image_root = data_item.get("_image_root", None)

            if "caption" in data_item:
                if "image" not in data_item:
                    raise ValueError("Missing 'image' key in data_item with 'caption'.")
                image_path = data_item.get("image", "")
                full_image_path = read_general2(data_item["image"], image_root)
                if full_image_path is None:
                    raise ValueError(f"Cannot find image file: {data_item['image']} with root: {image_root}")
                image = Image.open(full_image_path).convert("RGB")
                text = data_item.get("caption", "")
                
                image = self.image_transform(image)
                return image, text, image_path
            else:
                raise ValueError("Data item must contain 'caption' key.")

        except Exception as e:
            print(f"Error processing data_item: {e}")
            return None, "", ""

    
#############################################################################
#                           Training Helper Functions                       #
#############################################################################


def dataloader_collate_fn(samples):
    image = [x[0] for x in samples]
    caps = [x[1] for x in samples] 
    return image, caps


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            captions.append("")
    
    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


#############################################################################
#                                Training Loop                              #
#############################################################################


def initialize_progress_bar(total_steps):
    return tqdm(total=total_steps, desc="Training Progress", leave=True)


def update_progress_bar(pbar, step, loss=None, grad_norm=None, lr=None, max_steps=100):
    postfix = {"step": f"{step}/{max_steps}"}
    if loss is not None:
        postfix["loss"] = f"{loss:.4f}"
    if grad_norm is not None:
        postfix["grad_norm"] = f"{grad_norm:.4f}"
    if lr is not None:
        postfix["lr"] = f"{lr:.6f}"
    pbar.set_postfix(postfix)
    pbar.update()


def calculate_l2_grad_norm(model, model_parallel_dim_dict):
    """
    Calculate the L2 norm of the gradients.
    """
    grad_norm_sq = 0.0
    for n, p in model.named_parameters():
        if p.grad is not None:
            mp_dim = model_parallel_dim_dict.get(n, None)
            if mp_dim is not None:
                mp_size = fs_init.get_model_parallel_world_size()
                grad_norm_sq += torch.norm(p.grad, p=2).item() ** 2 * mp_size / min(mp_size, p.shape[mp_dim])
            else:
                grad_norm_sq += torch.norm(p.grad, p=2).item() ** 2
    return grad_norm_sq**0.5


def find_latest_checkpoint(checkpoint_path):
    """
    Find the latest checkpoint in the directory if path ends with 'latest'.
    """
    if not checkpoint_path or not checkpoint_path.endswith('latest'):
        return checkpoint_path
        
    parent_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(parent_dir):
        return checkpoint_path
    
    checkpoint_dirs = []
    for d in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, d)
        if os.path.isdir(full_path) and d.isdigit():
            checkpoint_dirs.append((int(d), full_path))
    
    if not checkpoint_dirs:
        return checkpoint_path
    
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: x[0], reverse=True)[0][1]
    return latest_checkpoint


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()

    assert args.global_batch_size % dp_world_size == 0, "Batch size must be divisible by data parallel world size."
    local_batch_size = args.global_batch_size // dp_world_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)

    os.makedirs(args.results_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + args.id
    time_based_dir = os.path.join(args.results_dir, current_time)
    os.makedirs(time_based_dir, exist_ok=True)

    checkpoint_dir = os.path.join(time_based_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    args_file_path = os.path.join(time_based_dir, "args.json")
    with open(args_file_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if rank == 0:
        logger = create_logger(time_based_dir)
        logger.info(f"Experiment directory: {time_based_dir}")
        tb_logger = SummaryWriter(
            os.path.join(
                time_based_dir,
                "tensorboard",
                datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname(),
            )
        )
    else:
        logger = create_logger(None)
        tb_logger = None
        
    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))
    logger.info(f"Setting up language model: google/gemma-2b")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer.padding_side = "right"

    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            torch_dtype=torch.bfloat16,
        )
        .get_decoder()
        .cuda()
    )
    text_encoder = setup_lm_fsdp_sync(text_encoder)
    cap_feat_dim = text_encoder.config.hidden_size
        
    if args.model in ["NextDiT_2B_patch2", "DiT_Llama2_7B_patch2", "NextDiT_2B_GQA_patch2"]:
        model = models.__dict__[args.model](
            in_channels=16 if args.vae == "sd3" else 4,
            qk_norm=args.qk_norm,
            cap_feat_dim=cap_feat_dim,
        )
    else:
        raise ValueError(f"Model {args.model} is not supported!")
            
    logger.info(f"DiT Parameters: {model.parameter_count():,}")
    model_patch_size = model.patch_size
    model_parallel_dim_dict = get_model_parallel_dim_dict(model)

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    if args.resume:
        latest_resume = find_latest_checkpoint(args.resume)
        if latest_resume != args.resume:
            logger.info(f"Found latest checkpoint for resume: {latest_resume}")
            args.resume = latest_resume
    
    if args.init_from:
        latest_init = find_latest_checkpoint(args.init_from)
        if latest_init != args.init_from:
            logger.info(f"Found latest checkpoint for init_from: {latest_init}")
            args.init_from = latest_init

    model_ema = deepcopy(model)
    if args.resume:
        if dp_rank == 0:
            logger.info(f"Resuming model weights from: {args.resume}")
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
            logger.info(f"Resuming ema weights from: {args.resume}")
            model_ema.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated_ema.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
    elif args.init_from:
        if dp_rank == 0:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(
                os.path.join(
                    args.init_from,
                    f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
                ),
                map_location="cpu",
            )

            size_mismatch_keys = []
            model_state_dict = model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpected keys: {unexpected_keys}")
    dist.barrier()

    if args.checkpointing:
        checkpointing_list = list(model.transformer_blocks)
        checkpointing_list_ema = list(model_ema.transformer_blocks)
    else:
        checkpointing_list = []
        checkpointing_list_ema = []

    model = setup_fsdp_sync(model, args)
    model_ema = setup_fsdp_sync(model_ema, args)

    if args.checkpointing:
        print("Applying gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list,
        )
        apply_activation_checkpointing(
            model_ema,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_ema,
        )

    logger.info(f"Model:\n{model}\n")

    transport = create_transport("Linear", "velocity", None, None, None, snr_type=args.snr_type)
    
    if args.vae == "sd3":
        logger.info("Using SD3 VAE")
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae").to(device)
    elif args.vae == "sdxl":
        logger.info("Using SDXL VAE")
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}"
            if args.local_diffusers_model_root is None
            else os.path.join(args.local_diffusers_model_root, f"stabilityai/sd-vae-ft-{args.vae}")
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    elif args.init_from:
        with open(os.path.join(args.init_from, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    logger.info(f"Resume step: {resume_step}")
    
    logger.info("Creating data transform...")
    patch_size = 8 * model_patch_size
    logger.info(f"Patch size: {patch_size}")
    max_num_patches = round((args.image_size / patch_size) ** 2)
    logger.info(f"Limiting number of patches to {max_num_patches}")
    crop_size_list = generate_crop_size_list(max_num_patches, patch_size)
    logger.info("List of crop sizes:")
    for i in range(0, len(crop_size_list), 6):
        logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[i : i + 6]]))
    
    image_transform = transforms.Compose(
        [
            transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = MyDataset(
        args.data_path,
        item_processor=T2IItemProcessor(image_transform),
        cache_on_disk=args.cache_data_on_disk,
    )
    num_samples = args.global_batch_size * args.max_steps
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Total # samples to consume: {num_samples:,} ({num_samples / len(dataset):.2f} epochs)")
    
    sampler = get_train_sampler(
        dataset,
        dp_rank,
        dp_world_size,
        args.global_batch_size,
        args.max_steps,
        resume_step,
        args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataloader_collate_fn,
        drop_last=True
    )

    model.train()

    log_steps = 0
    running_loss = 0
    running_grad_norm = 0
    start_time = time()
    
    pbar = initialize_progress_bar(total_steps=len(loader))

    logger.info(f"Training for {args.max_steps:,} steps...")
    for step, (x, caps) in enumerate(loader, start=resume_step):
        x = [img.to(device, non_blocking=True) for img in x]
         
        with torch.no_grad():
            vae_scale = {"sdxl": 0.13025, "sd3": 1.5305, "ema": 0.18215, "mse": 0.18215}[args.vae]
            vae_shift = {"sdxl": 0.0, "sd3": 0.0609, "ema": 0.0, "mse": 0.0}[args.vae]

            if step == resume_step:
                logger.warning(f"VAE scale: {vae_scale}, VAE shift: {vae_shift}")
            
            x = [(vae.encode(img[None]).latent_dist.sample()[0] - vae_shift) * vae_scale for img in x]

        with torch.no_grad():
            cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, args.caption_dropout_prob)

        loss_item = 0.0
        opt.zero_grad()
        for mb_idx in range((local_batch_size - 1) // args.micro_batch_size + 1):
            mb_st = mb_idx * args.micro_batch_size
            mb_ed = min((mb_idx + 1) * args.micro_batch_size, local_batch_size)
            last_mb = mb_ed == local_batch_size

            x_mb = x[mb_st:mb_ed]
            cap_feats_mb = cap_feats[mb_st:mb_ed]
            cap_mask_mb = cap_mask[mb_st:mb_ed]

            model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb)
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                loss_dict = transport.training_losses(model, x_mb, model_kwargs)
            loss = loss_dict["loss"].sum() / local_batch_size
            loss_item += loss.item()
            with model.no_sync() if args.data_parallel in ["sdp", "fsdp"] and not last_mb else contextlib.nullcontext():
                loss.backward()

        grad_norm = calculate_l2_grad_norm(model, model_parallel_dim_dict)
        if grad_norm > args.grad_clip:
            scale_grad(model, args.grad_clip / grad_norm)

        if tb_logger is not None:
            tb_logger.add_scalar("train/loss", loss_item, step)
            tb_logger.add_scalar("train/grad_norm", grad_norm, step)
            tb_logger.add_scalar("train/lr", opt.param_groups[0]["lr"], step)

        opt.step()
        update_ema(model_ema, model)
            
        update_progress_bar(pbar, step, loss=loss.item(), grad_norm=grad_norm, 
                          lr=opt.param_groups[0]["lr"], max_steps=args.max_steps)

        running_loss += loss_item
        running_grad_norm += grad_norm
        log_steps += 1
        if (step + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            end_time = time()
            secs_per_step = (end_time - start_time) / log_steps
            imgs_per_sec = args.global_batch_size * log_steps / (end_time - start_time)
            
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            avg_grad_norm = running_grad_norm / log_steps
            logger.info(
                f"(step={step + 1:07d}) "
                f"Train Loss: {avg_loss:.4f}, "
                f"Train Grad Norm: {avg_grad_norm:.4f}, "
                f"Train Secs/Step: {secs_per_step:.2f}, "
                f"Train Imgs/Sec: {imgs_per_sec:.2f}"
            )
            
            running_loss = 0
            running_grad_norm = 0
            log_steps = 0
            start_time = time()

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            
            if dist.get_rank() == 0 and args.checkpoints_total_limit is not None:
                all_checkpoints = []
                if os.path.exists(checkpoint_dir):
                    all_checkpoints = [d for d in os.listdir(checkpoint_dir) 
                                     if os.path.isdir(os.path.join(checkpoint_dir, d))]
                    all_checkpoints = sorted(all_checkpoints, key=lambda x: int(x))
                
                if len(all_checkpoints) + 1 > args.checkpoints_total_limit:
                    num_to_remove = len(all_checkpoints) + 1 - args.checkpoints_total_limit
                    checkpoints_to_remove = all_checkpoints[:num_to_remove]
                    logger.info(f"Removing {len(checkpoints_to_remove)} old checkpoint(s) to maintain limit of {args.checkpoints_total_limit}")
                    for checkpoint_folder in checkpoints_to_remove:
                        remove_path = os.path.join(checkpoint_dir, checkpoint_folder)
                        logger.info(f"Removing old checkpoint: {remove_path}")
                        import shutil
                        shutil.rmtree(remove_path)
            
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_ema_state_dict = model_ema.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_ema_fn = (
                        "consolidated_ema."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_ema_state_dict,
                        os.path.join(checkpoint_path, consolidated_ema_fn),
                    )
            dist.barrier()
            del consolidated_ema_state_dict
            logger.info(f"Saved consolidated_ema to {checkpoint_path}")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}")
        
    dist.barrier()
    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='debugging', required=False)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=False, default='results/debugging/')
    parser.add_argument("--model", type=str, default="NextDiT_2B_GQA_patch2")
    parser.add_argument("--image_size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse", "sdxl", "sd3"], default="ema")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--checkpointing", action="store_true", default=False, help="Enable gradient checkpointing")
    parser.add_argument(
        "--local_diffusers_model_root",
        type=str,
        help="Specify the root directory if diffusers models are to be loaded "
        "from the local filesystem (instead of being automatically "
        "downloaded from the Internet). Useful in environments without "
        "Internet access.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--init_from",
        type=str,
        help="Initialize the model weights from a checkpoint folder. "
        "Compared to --resume, this loads neither the optimizer states "
        "nor the data loader states.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=2.0,
        help="Clip the L2 norm of the gradients to the given value.",
    )
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay for the optimizer.")
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )
    parser.add_argument("--snr_type", type=str, default="uniform")
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help="Maximum number of checkpoints to keep. Older checkpoints will be deleted.",
    )
    
    args = parser.parse_args()
    main(args)