#!/usr/bin/env bash
# FrameCrafter LoRA training -- low-resolution stage (192x336, 6-to-1).
#
# Fixed context-to-target split: M = 6 input frames, N = 1 target frame
# (num_frames = M + N = 7). This is the cheapest configuration -- recommended
# as a first stage before fine-tuning at full resolution
# (train_480_832_6to1.sh) and, optionally, with variable M-to-N
# (train_480_832_mixed.sh).
#
# Hardware: 8x 48GB GPUs (e.g. A6000s) via DeepSpeed ZeRO-2
# (see model_training/my_config.yaml).

accelerate launch --config_file model_training/my_config.yaml model_training/train.py \
  --dataset_base_path ../DL3DV-10K_960P/1K \
  --dataset_metadata_path ../DL3DV-10K_960P/1K \
  --height 192 \
  --width 336 \
  --num_frames 7 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 160 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/framecrafter-192_336_6to1" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --modify_channels \
  --new_in_dim 420 \
  --gradient_accumulation_steps 1 \
  --initialize_model_on_cpu \
  --individual_encoding \
  --sampling_strategy "prob_random" \
  --wandb_project "framecrafter" \
  --wandb_run_name "192_336_6to1" \
  --num_input_frames 6 \
  # --resume_checkpoint "./models/train/framecrafter-192_336_6to1/epoch-last.safetensors" \
