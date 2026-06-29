#!/bin/bash

# =============================================================================
# Clean evaluation script for the selected refinement configuration.
#   joint PMI       : LH=0.0, RH=0.0
#   margin gate     : LH=inf, RH=inf
#   bigram PMI      : LH=0.10, RH=0.10
#   refinement top-k: 5
#
# Override paths/hyperparameters with environment variables, e.g.:
#   CHECKPOINT=./output/.../checkpoint-best.pth \
#   LH_DATA_DIR=/path/to/lh_v0 RH_DATA_DIR=/path/to/rh_v0 \
#   bash scripts/evaluate.sh
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-./output/compositional_transformer_v0/checkpoint-best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/compositional_transformer_v0/refinement_best_bigram0.10}"

LH_DATA_DIR="${LH_DATA_DIR:-/home/hao/Polyphony/data/havid_mmaction_extended/lh_v0}"
RH_DATA_DIR="${RH_DATA_DIR:-/home/hao/Polyphony/data/havid_mmaction_extended/rh_v0}"

LH_TRAIN_ANN="${LH_TRAIN_ANN:-${LH_DATA_DIR}/train_list_compositional.txt}"
RH_TRAIN_ANN="${RH_TRAIN_ANN:-${RH_DATA_DIR}/train_list_compositional.txt}"
LH_VAL_ANN="${LH_VAL_ANN:-${LH_DATA_DIR}/val_list_compositional.txt}"
RH_VAL_ANN="${RH_VAL_ANN:-${RH_DATA_DIR}/val_list_compositional.txt}"

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
LH_NUM_VERBS="${LH_NUM_VERBS:-7}"
LH_NUM_MANIP_OBJS="${LH_NUM_MANIP_OBJS:-26}"
LH_NUM_TARGET_OBJS="${LH_NUM_TARGET_OBJS:-26}"
LH_NUM_TOOLS="${LH_NUM_TOOLS:-6}"
RH_NUM_VERBS="${RH_NUM_VERBS:-7}"
RH_NUM_MANIP_OBJS="${RH_NUM_MANIP_OBJS:-26}"
RH_NUM_TARGET_OBJS="${RH_NUM_TARGET_OBJS:-26}"
RH_NUM_TOOLS="${RH_NUM_TOOLS:-6}"

DECODER_LAYERS="${DECODER_LAYERS:-3}"
DECODER_HEADS="${DECODER_HEADS:-8}"
DECODER_DIM="${DECODER_DIM:-2048}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"
ADAPTER_DIM="${ADAPTER_DIM:-128}"

NUM_FRAMES="${NUM_FRAMES:-16}"
SAMPLING_RATE="${SAMPLING_RATE:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MODE="${EVAL_MODE:-validation}"

# ---------------------------------------------------------------------------
# Selected refinement parameters from the sweep
# ---------------------------------------------------------------------------
REFINEMENT_TOP_K="${REFINEMENT_TOP_K:-5}"
LH_PMI_WEIGHT="${LH_PMI_WEIGHT:-0.0}"
RH_PMI_WEIGHT="${RH_PMI_WEIGHT:-0.0}"
LH_PMI_MARGIN="${LH_PMI_MARGIN:-inf}"
RH_PMI_MARGIN="${RH_PMI_MARGIN:-inf}"
LH_PMI_BIGRAM_WEIGHT="${LH_PMI_BIGRAM_WEIGHT:-0.10}"
RH_PMI_BIGRAM_WEIGHT="${RH_PMI_BIGRAM_WEIGHT:-0.10}"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "Evaluate v0 transformer with selected refinement parameters"
echo "========================================================================"
echo "Checkpoint : ${CHECKPOINT}"
echo "LH data    : ${LH_DATA_DIR}"
echo "RH data    : ${RH_DATA_DIR}"
echo "Output     : ${OUTPUT_DIR}"
echo "Python     : ${PYTHON}"
echo "CUDA devs  : ${CUDA_VISIBLE_DEVICES}"
echo "Eval mode  : ${EVAL_MODE}"
echo "Refinement : top_k=${REFINEMENT_TOP_K}"
echo "             joint PMI LH/RH=${LH_PMI_WEIGHT}/${RH_PMI_WEIGHT}"
echo "             margin   LH/RH=${LH_PMI_MARGIN}/${RH_PMI_MARGIN}"
echo "             bigram   LH/RH=${LH_PMI_BIGRAM_WEIGHT}/${RH_PMI_BIGRAM_WEIGHT}"
echo "========================================================================"

MISSING=0
for p in "${CHECKPOINT}" "${LH_TRAIN_ANN}" "${RH_TRAIN_ANN}" "${LH_VAL_ANN}" "${RH_VAL_ANN}"; do
    if [ ! -e "${p}" ]; then
        echo "[error] Missing required path: ${p}" >&2
        MISSING=1
    fi
done
if [ "${MISSING}" = "1" ]; then
    echo "Set CHECKPOINT, LH_DATA_DIR/RH_DATA_DIR, or annotation path variables before running." >&2
    exit 2
fi

"${PYTHON}" evaluate_dual_transformer_with_refinement.py \
    --checkpoint       "${CHECKPOINT}" \
    --model            vit_base_patch16_224_compositional_dual_transformer \
    --lh_data_dir      "${LH_DATA_DIR}" \
    --rh_data_dir      "${RH_DATA_DIR}" \
    --lh_train_ann     "${LH_TRAIN_ANN}" \
    --rh_train_ann     "${RH_TRAIN_ANN}" \
    --lh_val_ann       "${LH_VAL_ANN}" \
    --rh_val_ann       "${RH_VAL_ANN}" \
    --lh_num_verbs       "${LH_NUM_VERBS}" \
    --lh_num_manip_objs  "${LH_NUM_MANIP_OBJS}" \
    --lh_num_target_objs "${LH_NUM_TARGET_OBJS}" \
    --lh_num_tools       "${LH_NUM_TOOLS}" \
    --rh_num_verbs       "${RH_NUM_VERBS}" \
    --rh_num_manip_objs  "${RH_NUM_MANIP_OBJS}" \
    --rh_num_target_objs "${RH_NUM_TARGET_OBJS}" \
    --rh_num_tools       "${RH_NUM_TOOLS}" \
    --decoder_layers   "${DECODER_LAYERS}" \
    --decoder_heads    "${DECODER_HEADS}" \
    --decoder_dim      "${DECODER_DIM}" \
    --decoder_dropout  "${DECODER_DROPOUT}" \
    --use_hand_adapters \
    --adapter_dim      "${ADAPTER_DIM}" \
    --num_frames       "${NUM_FRAMES}" \
    --sampling_rate    "${SAMPLING_RATE}" \
    --refinement_top_k "${REFINEMENT_TOP_K}" \
    --pmi_weight       0.0 \
    --lh_pmi_weight    "${LH_PMI_WEIGHT}" \
    --rh_pmi_weight    "${RH_PMI_WEIGHT}" \
    --pmi_margin       1e9 \
    --lh_pmi_margin    "${LH_PMI_MARGIN}" \
    --rh_pmi_margin    "${RH_PMI_MARGIN}" \
    --pmi_bigram_weight 0.0 \
    --lh_pmi_bigram_weight "${LH_PMI_BIGRAM_WEIGHT}" \
    --rh_pmi_bigram_weight "${RH_PMI_BIGRAM_WEIGHT}" \
    --eval_mode        "${EVAL_MODE}" \
    --batch_size       "${BATCH_SIZE}" \
    --num_workers      "${NUM_WORKERS}" \
    --output_dir       "${OUTPUT_DIR}"

echo "========================================================================"
echo "Evaluation complete. Results saved to: ${OUTPUT_DIR}"
echo "========================================================================"
