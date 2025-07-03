import argparse
import json
import os
import gc
import random
import socket
import time
from torchvision.transforms.functional import to_pil_image
from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datetime import datetime
import models
from transport import Sampler, create_transport
import torch.multiprocessing as mp

def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # Random choice during training, first element otherwise
            captions.append(random.choice(caption) if is_train else caption[0])
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
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]
    return prompt_embeds, prompt_masks

def none_or_str(value):
    if value == "None":
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--sampling-method",
        type=str,
        default="euler",
        help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq",
    )
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")


def main(rank, args, master_port):
    # Disable gradient computation for sampling
    torch.set_grad_enabled(False)

    # Set process environment variables
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize distributed process group with NCCL backend
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    # Initialize FairScale model parallel group
    import fairscale.nn.model_parallel.initialize as parallel_init
    if parallel_init._MODEL_PARALLEL_GROUP is None:
        parallel_init._MODEL_PARALLEL_GROUP = torch.distributed.new_group(ranks=[torch.distributed.get_rank()])

    # Load model arguments
    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    print('train_args: ', train_args)
    if dist.get_rank() == 0:
        print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

    # Set data type
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    # Initialize tokenizer and text encoder
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", add_eos=True)
    tokenizer.padding_side = "right"
    text_encoder = AutoModel.from_pretrained("google/gemma-2b", torch_dtype=dtype, device_map="cuda").eval()
    cap_feat_dim = text_encoder.config.hidden_size

    # Load VAE model
    vae = AutoencoderKL.from_pretrained(
        (f"stabilityai/sd-vae-ft-{train_args.vae}" if train_args.vae != "sdxl" else "stabilityai/sdxl-vae"),
        torch_dtype=torch.float32,
    ).cuda()

    # Build model and load weights
    model = models.__dict__[train_args.model](
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    model.eval().to("cuda", dtype=dtype)

    if not args.debug:
        if args.ema:
            print("Loading ema model.")
        ckpt = torch.load(
            os.path.join(
                args.ckpt,
                f"consolidated{'_ema' if args.ema else ''}.00-of-01.pth",
            ),
            map_location=lambda storage, loc: storage.cuda(rank)
        )
        model.load_state_dict(ckpt, strict=True)

    sample_folder_dir = args.image_save_path
    current_time = ""
    # Generate seeds list and output directory (rank 0 only)
    if rank == 0:
        seeds_list = get_seeds(args)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.id}_{args.num_sampling_steps}_{args.sampling_method}"
        os.makedirs(os.path.join(sample_folder_dir, current_time, "images"), exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    else:
        seeds_list = None

    dist.barrier()
    # Broadcast seeds_list
    if rank == 0:
        seeds_tensor = torch.tensor(seeds_list, dtype=torch.long, device=torch.cuda.current_device())
        length_tensor = torch.tensor([len(seeds_list)], dtype=torch.long, device=torch.cuda.current_device())
    else:
        seeds_tensor = None
        length_tensor = torch.tensor([0], dtype=torch.long, device=torch.cuda.current_device())

    dist.broadcast(length_tensor, src=0)
    if rank != 0:
        seeds_tensor = torch.empty(length_tensor.item(), dtype=torch.long, device=torch.cuda.current_device())
    dist.broadcast(seeds_tensor, src=0)
    seeds_list = seeds_tensor.cpu().numpy().tolist()

    # Broadcast current_time
    if rank == 0:
        time_str_bytes = str(current_time).encode("utf-8")
        time_len_tensor = torch.tensor([len(time_str_bytes)], dtype=torch.long, device=torch.cuda.current_device())
    else:
        time_str_bytes = b""
        time_len_tensor = torch.tensor([0], dtype=torch.long, device=torch.cuda.current_device())

    dist.broadcast(time_len_tensor, src=0)
    if rank != 0:
        time_str_bytes = bytearray(time_len_tensor.item())
    time_str_tensor = torch.tensor(list(time_str_bytes), dtype=torch.uint8, device=torch.cuda.current_device())
    dist.broadcast(time_str_tensor, src=0)
    current_time = time_str_tensor.cpu().numpy().tobytes().decode("utf-8")

    dist.barrier()

    # Read and broadcast caption list
    captions = []
    if rank == 0:
        with open(args.caption_path, "r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if text:
                    captions.append(text)
        cap_count = len(captions)
        cap_count_tensor = torch.tensor([cap_count], dtype=torch.long, device=torch.cuda.current_device())
    else:
        cap_count_tensor = torch.tensor([0], dtype=torch.long, device=torch.cuda.current_device())

    dist.broadcast(cap_count_tensor, src=0)
    cap_count = cap_count_tensor.item()
    if rank != 0:
        captions = ["" for _ in range(cap_count)]
    # Broadcast captions one by one
    for i in range(cap_count):
        if rank == 0:
            c_str_bytes = captions[i].encode("utf-8")
            c_len_tensor = torch.tensor([len(c_str_bytes)], dtype=torch.long, device=torch.cuda.current_device())
        else:
            c_str_bytes = b""
            c_len_tensor = torch.tensor([0], dtype=torch.long, device=torch.cuda.current_device())
        dist.broadcast(c_len_tensor, src=0)
        if rank != 0:
            c_str_bytes = bytearray(c_len_tensor.item())
        c_str_tensor = torch.tensor(list(c_str_bytes), dtype=torch.uint8, device=torch.cuda.current_device())
        dist.broadcast(c_str_tensor, src=0)
        captions[i] = c_str_tensor.cpu().numpy().tobytes().decode("utf-8")

    dist.barrier()
    # Split caption list by rank
    world_size = dist.get_world_size()
    chunk_size = (cap_count + world_size - 1) // world_size
    start_idx = rank * chunk_size
    end_idx = min((rank + 1) * chunk_size, cap_count)
    sub_captions = captions[start_idx:end_idx]

    # Prepare JSONL file for writing
    local_info_filename = f"data_gpu{rank}.jsonl"
    local_info_path = os.path.join(sample_folder_dir, current_time, local_info_filename)

    with open(local_info_path, "w", encoding="utf-8") as f_jsonl:
        # Start sampling
        with torch.autocast("cuda", dtype):
            for res in args.resolution:
                res_cat, resolution = res.split(":")
                w, h = tuple(map(int, resolution.split("x")))
                latent_w, latent_h = w // 8, h // 8

                for idx, caption in tqdm(
                    enumerate(sub_captions),
                    desc=f"Resolution {res} [GPU {rank}]",
                    disable=(rank != 0)
                ):
                    with torch.no_grad():
                        caption_list = [caption]
                        cap_feats, cap_mask = encode_prompt([caption_list] + [""], text_encoder, tokenizer, 0.0)
                        cap_mask = cap_mask.to(cap_feats.device)
                    
                    model_kwargs = dict(
                        cap_feats=cap_feats,
                        cap_mask=cap_mask,
                        cfg_scale=args.cfg_scale,
                    )
                    if args.proportional_attn:
                        model_kwargs["proportional_attn"] = True
                        model_kwargs["base_seqlen"] = (train_args.image_size // 16) ** 2
                        
                    else:
                        model_kwargs["proportional_attn"] = False
                        model_kwargs["base_seqlen"] = None

                    transport = create_transport(
                        args.path_type,
                        args.prediction,
                        args.loss_weight,
                        args.train_eps,
                        args.sample_eps
                    )
                    sampler = Sampler(transport)
                    sample_fn = sampler.sample_ode(
                        sampling_method=args.sampling_method,
                        num_steps=args.num_sampling_steps,
                        atol=args.atol,
                        rtol=args.rtol,
                        reverse=args.reverse,
                        time_shifting_factor=args.time_shifting_factor,
                    )

                    for seed_i, seed_val in enumerate(seeds_list):
                        torch.random.manual_seed(seed_val)
                        z = torch.randn([1, 4, latent_w, latent_h], device="cuda").to(dtype)
                        z = z.repeat(len(caption_list) * 2, 1, 1, 1)
                        samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1][:1]
                        
                        factor = 0.18215 if train_args.vae != "sdxl" else 0.13025
                        samples = vae.decode(samples / factor).sample
                        samples = (samples + 1.0) / 2.0
                        samples.clamp_(0.0, 1.0)
                        
                        img = to_pil_image(samples[0].float())
                        idx_in_global = start_idx + idx
                        save_filename = f"{idx_in_global}_res={res}_{seed_i}_seed={seed_val}_gpu={rank}.png"
                        save_path = os.path.join(sample_folder_dir, current_time, "images", save_filename)
                        img.save(save_path)
                        relative_file_path = os.path.join("images", save_filename)
                        
                        # Write sampling info to JSONL file
                        item = {
                            "text": caption,
                            "absolute_image_path": save_path,
                            "relative_image_path": relative_file_path,
                            "resolution": resolution,
                            "sampling_method": args.sampling_method,
                            "num_sampling_steps": args.num_sampling_steps,
                            "seed_used": seed_val,
                            "time": current_time,
                            "gpu_rank": rank,
                        }
                        f_jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
                        f_jsonl.flush()
                        
                        gc.collect() 

    gc.collect()
    torch.cuda.empty_cache()

    # Synchronize all processes
    dist.barrier()
    print(f"Rank {rank} finished writing to {local_info_path}")
    dist.barrier()

    # Destroy distributed process group
    dist.destroy_process_group()

def find_free_port(start_port=38003, end_port=60000, retries=50, delay=1):
    for attempt in range(retries):
        for port in range(start_port, end_port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("127.0.0.1", port))
                    print(f"Found free port: {port}")
                    return port
            except OSError:
                continue
        print(f"Retrying to find a free port (Attempt {attempt + 1}/{retries}) after {delay} seconds...")
        time.sleep(delay)
    raise RuntimeError("No free ports available in the specified range")

def get_seeds(args):
    """
    Generate final seed list based on user input and samples_per_caption.
    """
    if not args.seeds:
        if args.samples_per_caption == 1:
            return [42]
        else:
            return [random.randint(0, 2**32 - 1) for _ in range(args.samples_per_caption)]
    else:
        provided_seeds = args.seeds
        num_provided = len(provided_seeds)
        num_required = args.samples_per_caption
        if num_provided == num_required:
            return provided_seeds
        elif num_provided < num_required:
            additional_seeds = [random.randint(0, 2**32 - 1) for _ in range(num_required - num_provided)]
            return provided_seeds + additional_seeds
        else:
            raise ValueError(f"Number of provided seeds ({num_provided}) exceeds samples_per_caption ({num_required}). "
                           f"Please provide no more than {num_required} seeds or adjust samples_per_caption.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="sampling_debugging")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Manual specification of random seeds")
    parser.add_argument("--samples_per_caption", type=int, default=1, help="Number of images to generate per caption")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--precision", type=str, choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.set_defaults(ema=True)
    parser.add_argument("--image_save_path", type=str, default="samples")
    parser.add_argument("--time_shifting_factor", type=float, default=1.0)
    parser.add_argument("--caption_path", type=str, default="test_captions.txt")
    parser.add_argument("--resolution", type=str, default="", nargs="+")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--proportional_attn", type=bool, default=True)
    parser.add_argument("--scaling_method", type=str, default="Time-aware")
    parser.add_argument("--scaling_watershed", type=float, default=0.3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)

    parse_transport_args(parser)
    parse_ode_args(parser)

    args = parser.parse_args()

    master_port = find_free_port()

    # Launch multi-process using mp.spawn
    mp.spawn(
        main,
        nprocs=args.num_gpus,
        join=True,
        args=(args, master_port),
    )