#!/bin/bash

# =========================================================
# 🏆 最终训练脚本 (Fixed Scheduler Params)
# =========================================================

# 1. 显存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 2. 实验名称
EXP_NAME="effort_universal_diff_margin_fixed_final"

# 3. 启动训练
python train.py \
  --name ${EXP_NAME} \
  --arch CLIP:ViT-L/14_svd \
  --gpu_ids 0 \
  --fix_backbone \
  --use_svd \
  --svd_rank_ratio 0.25 \
  --data_mode wang2020 \
  --wang2020_data_path /root/autodl-tmp/datasets/CNNDetection \
  --batch_size 32 \
  --lr 0.00001 \
  --niter 20 \
  --save_epoch_freq 1 \
  --noise_std 0.02 \
  --loss_freq 10 \
  --data_aug \
  --blur_prob 0.5 \
  --jpg_prob 0.5 \
  --warmup_steps 500
  # 👆【关键修复】必须显式指定，否则 Scheduler 不会初始化！
