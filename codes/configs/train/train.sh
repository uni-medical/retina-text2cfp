cd path/to_your_repo/codes

CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master_port=38678 train.py \
    --id "debugging"\
    --data_path ./configs/train/data_example.yaml \
    --results_dir ./results/training/ \
    --micro_batch_size  8 \
    --global_batch_size 8 \
    --lr 1e-6 \
    --data_parallel fsdp \
    --max_steps 400000 \
    --ckpt_every 10000 \
    --log_every 1 \
    --precision bf16 \
    --grad_precision fp32 \
    --qk_norm \
    --image_size 512 \
    --vae sdxl \
    --global_seed 0 \
    --num_workers 24 \
    --local_diffusers_model_root None \
    --init_from ./results/model_chkpt/checkpoints/model_id7 \
    --grad_clip 0.5 \
    --wd 0.00001 \
    --snr_type uniform