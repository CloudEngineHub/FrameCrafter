#!/usr/bin/env bash
# FrameCrafter LoRA training -- full-resolution mixed M-to-N (480x832).
#
# Random context-to-target split: --num_input_frames and --num_output_frames
# are BOTH omitted, so every sample draws
#   M in [min_input_frames, num_frames - min_output_frames] = [3, 9]
#   N = num_frames - M
# (num_frames = 10, defaults min_input_frames=3 / min_output_frames=1). This
# stage teaches the model to handle a variable number of input views.
# Typically run after the fixed-split stages; supply the previous checkpoint
# via --resume_checkpoint (see the commented line at the bottom of this file).

accelerate launch --config_file model_training/my_config.yaml model_training/train.py \
  --dataset_base_path ../DL3DV-10K_960P/1K \
  --dataset_metadata_path ../DL3DV-10K_960P/1K \
  --height 480 \
  --width 832 \
  --num_frames 10 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 30 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/framecrafter-480_832_mixed" \
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
  --wandb_run_name "480_832_mixed" \
  # --resume_checkpoint "./models/train/framecrafter-480_832_6to1/epoch-last.safetensors" \
