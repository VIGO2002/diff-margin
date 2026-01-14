#!/bin/bash

# =========================================================
# 🚀 Diff-Margin 训练脚本 (完美复刻成功版增强配置)
# =========================================================

# 1. 显存优化 (保持一致)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 2. 实验名称
EXP_NAME="effort_universal_diff_margin_fixed"

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
  --lr 0.0002 \
  --niter 20 \
  --save_epoch_freq 1 \
  --noise_std 0.02 \
  \
  --loss_freq 10 \
  \
  --data_aug \
  --blur_prob 0.5 \
  --jpg_prob 0.5

# 解析：
# --loss_freq 10     : 改回了 10，和你成功版一致，方便你观察日志。
# --blur_prob 0.5    : 显式写出了 0.5，这和成功版利用 default=0.5 是一模一样的效果。
# --continue_train   : 【已删除】因为这是新实验，不要加这个。
