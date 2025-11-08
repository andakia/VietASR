#! /usr/bin/bash

set -euo pipefail

# This script fine-tunes a VietASR SSL checkpoint.
# Recommended workflow:
#   1. Run ./prepare_ssl.sh to prepare unsupervised data.
#   2. Train k-means labels with ./scripts/learn_vietASR_kmeans.sh and
#      generate label manifests with ./scripts/extract_vietASR_kmeans.sh.
#   3. Pretrain the SSL encoder via ./scripts/run_ssl.sh.
#   4. Finally, update the variables below (or export overrides) and run this script.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH=${PWD}/zipformer_fbank:${PYTHONPATH:-}

WORLD_SIZE=${WORLD_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-300}
START_EPOCH=${START_EPOCH:-1}
USE_FP16=${USE_FP16:-1}
SAMPLE_RATE=${SAMPLE_RATE:-100}
MANIFEST_DIR=${MANIFEST_DIR:-data/fbank}
BPE_MODEL=${BPE_MODEL:-/mnt/training/VietASR/ASR/data/lang_bpe_2000/bpe.model}
EXP_DIR=${EXP_DIR:-zipformer_fbank/exp-kmeans_ASR_100h-all/exp-epoch-9-tri-stage-100h}
MAX_DURATION=${MAX_DURATION:-1000}
ENABLE_MUSAN=${ENABLE_MUSAN:-0}
ENABLE_SPEC_AUG=${ENABLE_SPEC_AUG:-0}
MASK_BEFORE_CNN=${MASK_BEFORE_CNN:-1}
MASK_PROB=${MASK_PROB:-0.65}
MASK_CHANNEL_PROB=${MASK_CHANNEL_PROB:-0.5}
MASK_CHANNEL_LENGTH=${MASK_CHANNEL_LENGTH:-20}
ACCUM_GRAD=${ACCUM_GRAD:-1}
SEED=${SEED:-1556}
BASE_LR=${BASE_LR:-0.002}
MAX_LR_UPDATE=${MAX_LR_UPDATE:-80000}
PHASE_RATIO=${PHASE_RATIO:-"(0.1, 0.4, 0.5)"}
PRETRAINED_CHECKPOINT_PATH=${PRETRAINED_CHECKPOINT_PATH:-zipformer_fbank/exp/best-train-loss.pt}
FINAL_DOWNSAMPLE=${FINAL_DOWNSAMPLE:-1}
CAUSAL=${CAUSAL:-0}
MASTER_PORT=${MASTER_PORT:-12356}

python zipformer_fbank/finetune.py \
    --world-size "${WORLD_SIZE}" \
    --num-epochs "${NUM_EPOCHS}" \
    --start-epoch "${START_EPOCH}" \
    --use-fp16 "${USE_FP16}" \
    --sample-rate "${SAMPLE_RATE}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --bpe-model "${BPE_MODEL}" \
    --exp-dir "${EXP_DIR}" \
    --max-duration "${MAX_DURATION}" \
    --enable-musan "${ENABLE_MUSAN}" \
    --enable-spec-aug "${ENABLE_SPEC_AUG}" \
    --mask-before-cnn "${MASK_BEFORE_CNN}" \
    --mask-prob "${MASK_PROB}" \
    --mask-channel-prob "${MASK_CHANNEL_PROB}" \
    --mask-channel-length "${MASK_CHANNEL_LENGTH}" \
    --accum-grad "${ACCUM_GRAD}" \
    --seed "${SEED}" \
    --base-lr "${BASE_LR}" \
    --max-lr-update "${MAX_LR_UPDATE}" \
    --phase-ratio "${PHASE_RATIO}" \
    --pretrained-checkpoint-path "${PRETRAINED_CHECKPOINT_PATH}" \
    --final-downsample "${FINAL_DOWNSAMPLE}" \
    --causal "${CAUSAL}" \
    --master-port "${MASTER_PORT}"

