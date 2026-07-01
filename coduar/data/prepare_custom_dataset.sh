#!/bin/bash

# =============================================================================
# Single-Stream Dual-Hand Dataset Generation Script
# =============================================================================
# 
# CORRECT APPROACH:
# - Extracts video clips ONCE from single video stream
# - Generates TWO index files (LH and RH) pointing to same clips
# - Saves 50% disk space compared to dual-stream approach
# - Guarantees perfect temporal synchronization
#
# Output structure:
#   output_dir/
#     clips/
#       S0T7V0_ws16_0001.mp4  <- SINGLE copy of each clip
#       S0T7V0_ws24_0002.mp4
#       ...
#     train_list_video_lh.txt  <- LH labels for all clips
#     train_list_video_rh.txt  <- RH labels for all clips
#     train_pairs_index.txt    <- Both labels together
#     train_metadata.json
#
# =============================================================================

# -------------------------
# Configuration
# -------------------------

# Input directories
VIDEO_DIR="path to the custom video directory"
LH_ANNOTATION_DIR="path to the custom left hand annotation directory"
RH_ANNOTATION_DIR="path to the custom right hand annotation directory"

# Label mappings
LH_MAPPING="custom_labels/pt_mapping_list.txt"
RH_MAPPING="custom_labels/pt_mapping_list.txt"  # Usually same for both hands

# Output directory
OUTPUT_DIR="path to the output directory"

# Dataset split (optional - leave empty to process all videos)
TRAIN_SPLIT="path to the train split"
TEST_SPLIT="path to the test split"

# -------------------------
# VideoMAE V2 Configuration
# -------------------------
# Multi-scale windows for training (will be temporally sampled to 16 frames in dataloader)
TRAIN_WINDOW_SIZES="8, 16, 24, 32, 40"

# Single window for testing
TEST_WINDOW_SIZES="16"

# -------------------------
# Sampling Parameters
# -------------------------

# Stride ratio (0.5 = 50% overlap)
# Lower = more clips, higher overlap
# Higher = fewer clips, less overlap
STRIDE_RATIO=1

# Null suppression (target null ratio in final dataset)
# 0.15 = 15% null samples
# Default distribution: ~60% null -> we reduce to 15%
NULL_RATIO=0.15

# Long-tail upsampling
# Bottom 30% of actions by frequency are considered "long-tail"
LONGTAIL_THRESHOLD=0.3

# Long-tail actions are upsampled 3x
LONGTAIL_UPSAMPLE_FACTOR=3.0

# Temporal jitter (frames to randomly shift window boundaries)
# 0 = no jitter, 4 = ±4 frames
TEMPORAL_JITTER=4

# Random seed
SEED=42

# -------------------------
# Training Set
# -------------------------

echo "=========================================="
echo "Generating TRAINING set (single-stream)"
echo "=========================================="

python prepare_dual_hand_single_stream.py \
    --video_dir "$VIDEO_DIR" \
    --lh_annotation_dir "$LH_ANNOTATION_DIR" \
    --rh_annotation_dir "$RH_ANNOTATION_DIR" \
    --output_dir "${OUTPUT_DIR}/videos_train/" \
    --lh_mapping_file "$LH_MAPPING" \
    --rh_mapping_file "$RH_MAPPING" \
    --window_sizes "$TRAIN_WINDOW_SIZES" \
    --stride_ratio $STRIDE_RATIO \
    --null_ratio $NULL_RATIO \
    --longtail_threshold $LONGTAIL_THRESHOLD \
    --longtail_upsample_factor $LONGTAIL_UPSAMPLE_FACTOR \
    --temporal_jitter $TEMPORAL_JITTER \
    --split_list "$TRAIN_SPLIT" \
    --output_prefix "train" \
    --seed $SEED

echo ""
echo "Training set complete!"
echo "Output: ${OUTPUT_DIR}/train_videos/"
echo "  - videos_train/           <- Video clips (SINGLE copy each)"
echo "  - train_list_video_lh.txt  <- Left hand labels"
echo "  - train_list_video_rh.txt  <- Right hand labels"
echo ""

# -------------------------
# Test Set
# -------------------------

echo "=========================================="
echo "Generating TEST set (single-stream)"
echo "=========================================="

python prepare_dual_hand_single_stream.py \
    --video_dir "$VIDEO_DIR" \
    --lh_annotation_dir "$LH_ANNOTATION_DIR" \
    --rh_annotation_dir "$RH_ANNOTATION_DIR" \
    --output_dir "${OUTPUT_DIR}/videos_val/" \
    --lh_mapping_file "$LH_MAPPING" \
    --rh_mapping_file "$RH_MAPPING" \
    --window_sizes "$TEST_WINDOW_SIZES" \
    --stride_ratio 0.25 \
    --null_ratio 1.0 \
    --longtail_threshold 1.0 \
    --longtail_upsample_factor 1.0 \
    --temporal_jitter 0 \
    --split_list "$TEST_SPLIT" \
    --output_prefix "test" \
    --seed $SEED

echo ""
echo "Test set complete!"
echo "Output: ${OUTPUT_DIR}/videos_val/"
echo ""

