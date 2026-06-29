#!/bin/bash

# Training script for CoDuAR

# Number of GPUs to use
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=2

# Pretrained model path (VideoMAE pretrained on Kinetics)
MODEL_PATH='models/vit_b_k710_dl_from_giant.pth'

# Dataset paths to left hand and right hand data
LH_DATA_DIR='data/havid/lh_v0'
RH_DATA_DIR='data/havid/rh_v0'

# Annotation files (compositional format)
LH_TRAIN_ANN="${LH_DATA_DIR}/train_list_compositional.txt"
RH_TRAIN_ANN="${RH_DATA_DIR}/train_list_compositional.txt"
LH_VAL_ANN="${LH_DATA_DIR}/val_list_compositional.txt"
RH_VAL_ANN="${RH_DATA_DIR}/val_list_compositional.txt"

# Number of classes for each compositional element (HAVID dataset)
LH_NUM_VERBS=7
LH_NUM_MANIP_OBJS=26
LH_NUM_TARGET_OBJS=26
LH_NUM_TOOLS=6
RH_NUM_VERBS=7
RH_NUM_MANIP_OBJS=26
RH_NUM_TARGET_OBJS=26
RH_NUM_TOOLS=6

# Output directory
OUTPUT_DIR='./output/coduar'

# ===========================
# HYPERPARAMETERS
# ===========================

# Transformer decoder settings
DECODER_LAYERS=3         # Number of transformer decoder layers
DECODER_HEADS=8          # Number of attention heads
DECODER_DIM=2048         # FFN dimension in decoder
DECODER_DROPOUT=0.1      # Dropout in decoder

# Training settings
BATCH_SIZE=4
EPOCHS=50
WARMUP_EPOCHS=5
LR=1e-3
MIN_LR=1e-6
WEIGHT_DECAY=0.05

# Model settings
DROP_PATH=0.1
HEAD_DROP=0.1

# Video settings
NUM_FRAMES=16
SAMPLING_RATE=4

# Hand adapter settings (from adaptive model)
USE_HAND_ADAPTERS='--use_hand_adapters'  # Enable hand-specific adapters
ADAPTER_DIM=128

# ===========================
# TRAINING
# ===========================

cd "$(dirname "$0")/.." || exit

echo "========================================"
echo "Compositional Transformer Decoder Training"
echo "========================================"
echo "Decoder Layers: ${DECODER_LAYERS}"
echo "Decoder Heads: ${DECODER_HEADS}"
echo "Hand Adapters: ENABLED"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    echo "Training on single GPU..."
    python run_compositional_transformer.py \
        --model vit_base_patch16_224_compositional_dual_transformer \
        --finetune ${MODEL_PATH} \
        --lh_data_dir ${LH_DATA_DIR} \
        --rh_data_dir ${RH_DATA_DIR} \
        --lh_train_ann ${LH_TRAIN_ANN} \
        --rh_train_ann ${RH_TRAIN_ANN} \
        --lh_val_ann ${LH_VAL_ANN} \
        --rh_val_ann ${RH_VAL_ANN} \
        --lh_num_verbs ${LH_NUM_VERBS} \
        --lh_num_manip_objs ${LH_NUM_MANIP_OBJS} \
        --lh_num_target_objs ${LH_NUM_TARGET_OBJS} \
        --lh_num_tools ${LH_NUM_TOOLS} \
        --rh_num_verbs ${RH_NUM_VERBS} \
        --rh_num_manip_objs ${RH_NUM_MANIP_OBJS} \
        --rh_num_target_objs ${RH_NUM_TARGET_OBJS} \
        --rh_num_tools ${RH_NUM_TOOLS} \
        ${USE_HAND_ADAPTERS} \
        --adapter_dim ${ADAPTER_DIM} \
        --decoder_layers ${DECODER_LAYERS} \
        --decoder_heads ${DECODER_HEADS} \
        --decoder_dim ${DECODER_DIM} \
        --decoder_dropout ${DECODER_DROPOUT} \
        --data_set HAVID \
        --imagenet_default_mean_and_std \
        --num_frames ${NUM_FRAMES} \
        --sampling_rate ${SAMPLING_RATE} \
        --num_sample 2 \
        --num_segments 1 \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --min_lr ${MIN_LR} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --weight_decay ${WEIGHT_DECAY} \
        --epochs ${EPOCHS} \
        --drop_path ${DROP_PATH} \
        --head_drop_rate ${HEAD_DROP} \
        --layer_decay 0.75 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --opt_eps 1e-8 \
        --test_num_segment 10 \
        --test_num_crop 3 \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR} \
        --save_ckpt \
        --num_workers 8
else
    # Multi-GPU training with DDP
    echo "Training on ${NUM_GPUS} GPUs..."
    OMP_NUM_THREADS=1 python -m torch.distributed.launch \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=12320 \
        run_compositional_transformer.py \
        --model vit_base_patch16_224_compositional_dual_transformer \
        --finetune ${MODEL_PATH} \
        --lh_data_dir ${LH_DATA_DIR} \
        --rh_data_dir ${RH_DATA_DIR} \
        --lh_train_ann ${LH_TRAIN_ANN} \
        --rh_train_ann ${RH_TRAIN_ANN} \
        --lh_val_ann ${LH_VAL_ANN} \
        --rh_val_ann ${RH_VAL_ANN} \
        --lh_num_verbs ${LH_NUM_VERBS} \
        --lh_num_manip_objs ${LH_NUM_MANIP_OBJS} \
        --lh_num_target_objs ${LH_NUM_TARGET_OBJS} \
        --lh_num_tools ${LH_NUM_TOOLS} \
        --rh_num_verbs ${RH_NUM_VERBS} \
        --rh_num_manip_objs ${RH_NUM_MANIP_OBJS} \
        --rh_num_target_objs ${RH_NUM_TARGET_OBJS} \
        --rh_num_tools ${RH_NUM_TOOLS} \
        ${USE_HAND_ADAPTERS} \
        --adapter_dim ${ADAPTER_DIM} \
        --decoder_layers ${DECODER_LAYERS} \
        --decoder_heads ${DECODER_HEADS} \
        --decoder_dim ${DECODER_DIM} \
        --decoder_dropout ${DECODER_DROPOUT} \
        --data_set HAVID \
        --imagenet_default_mean_and_std \
        --num_frames ${NUM_FRAMES} \
        --sampling_rate ${SAMPLING_RATE} \
        --num_sample 2 \
        --num_segments 1 \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --min_lr ${MIN_LR} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --weight_decay ${WEIGHT_DECAY} \
        --epochs ${EPOCHS} \
        --drop_path ${DROP_PATH} \
        --head_drop_rate ${HEAD_DROP} \
        --layer_decay 0.75 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --opt_eps 1e-8 \
        --test_num_segment 10 \
        --test_num_crop 3 \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR} \
        --save_ckpt \
        --num_workers 8 \
        --dist_eval
fi

echo "========================================"
echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"

