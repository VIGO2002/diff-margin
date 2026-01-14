#!/bin/bash

# =========================================================
# 🚀 黄金配置: 不对称 Margin + L1 约束 + 原论文 LR
# 🎯 目标: Glide 大幅提升，其他数据集保持稳定
# =========================================================

# 1. 显存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 2. 实验名称
EXP_NAME="effort_universal_asym_margin_golden_v1"

# 3. 启动训练
# 注意：LR 改为了 0.0002 (原论文参数)，Batch=32
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
  --lr 0.0002 \
  --niter 20 \
  --loss_freq 50 \
  --save_epoch_freq 1 \
  --noise_std 0.02