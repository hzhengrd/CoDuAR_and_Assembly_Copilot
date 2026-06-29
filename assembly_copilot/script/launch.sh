#!/bin/bash

# Launch Real-time Inference GUI on a specific video

cd "$(dirname "$0")/.." || exit

# Configuration
VIDEO="/home/hao/Polyphony/data/case_study/single_stream_pt/S0T7V0.mp4"
CHECKPOINT="./output/compositional_transformer_single_stream_case_study/checkpoint-best.pth"

# Model configuration
LH_NUM_VERBS=5
LH_NUM_MANIP_OBJS=12
LH_NUM_TARGET_OBJS=6
LH_NUM_TOOLS=4
RH_NUM_VERBS=5
RH_NUM_MANIP_OBJS=12
RH_NUM_TARGET_OBJS=6
RH_NUM_TOOLS=4

# GPU
export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "Real-time Inference GUI"
echo "========================================"
echo "Video: ${VIDEO}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Processing: 16-frame windows at 15 fps"
echo "========================================"

python realtime_inference_gui.py \
    --video "${VIDEO}" \
    --checkpoint "${CHECKPOINT}" \
    --lh_num_verbs ${LH_NUM_VERBS} \
    --lh_num_manip_objs ${LH_NUM_MANIP_OBJS} \
    --lh_num_target_objs ${LH_NUM_TARGET_OBJS} \
    --lh_num_tools ${LH_NUM_TOOLS} \
    --rh_num_verbs ${RH_NUM_VERBS} \
    --rh_num_manip_objs ${RH_NUM_MANIP_OBJS} \
    --rh_num_target_objs ${RH_NUM_TARGET_OBJS} \
    --rh_num_tools ${RH_NUM_TOOLS}

