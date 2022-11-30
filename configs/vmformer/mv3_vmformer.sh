#!/usr/bin/env bash

set -x

python -u main_vm.py \
    --dataset_file vm \
    --epochs 36 \
    --lr 2e-4 \
    --lr_drop 18 30 \
    --batch_size 2 \
    --num_workers 2 \
    --vm_path /home/jiachenl/data/Matting/ \
    --num_queries 1 \
    --num_frames 5 \
    --backbone mv3 \
    --mask_loss_coef 5 \
    --dice_loss_coef 1 \
    --temporal_loss_coef 1 \
    --enc_layers 1 \
    --dec_layers 1 \
    --hidden_dim 256 \
    --num_feature_levels 3 \
    --version v1 \
    --query_temporal weight_sum \
    --fpn_temporal \
    --output_dir outputs/mv3_vmformer \
