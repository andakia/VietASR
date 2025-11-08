#!/usr/bin/env bash

set -euo pipefail

# Default configuration. Override by exporting the variables before running the
# script or by passing the corresponding --flag from the command line.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EPOCH="${EPOCH:-3}"
AVG="${AVG:-5}"
EXP_DIR="${EXP_DIR:-zipformer_fbank/exp-kmeans_ASR_100h-all/exp-epoch-9-tri-stage-100h}"
MAX_DURATION="${MAX_DURATION:-1000}"
BPE_MODEL="${BPE_MODEL:-/mnt/training/VietASR/ASR/data/lang_bpe_2000/bpe.model}"
DECODING_METHOD="${DECODING_METHOD:-greedy_search}"
MANIFEST_DIR="${MANIFEST_DIR:-/mnt/training/VietASR/ASR/data/fbank}"
USE_AVERAGED_MODEL="${USE_AVERAGED_MODEL:-0}"
FINAL_DOWNSAMPLE="${FINAL_DOWNSAMPLE:-1}"
CUTS_NAME="${CUTS_NAME:-all}"

print_usage() {
  cat <<'EOF'
Usage: ./scripts/decode.sh [options]

Environment variables with defaults can also be exported instead of flags:
  CUDA_VISIBLE_DEVICES  (default: 0)
  EPOCH                 (default: 10)
  AVG                   (default: 5)
  EXP_DIR               (default: zipformer_fbank/exp-kmeans_ASR_100h-all/exp-epoch-9-tri-stage-100h)
  MANIFEST_DIR          (default: /mnt/training/VietASR/ASR/data/fbank)
  BPE_MODEL             (default: /mnt/training/VietASR/ASR/data/lang_bpe_2000/bpe.model)
  DECODING_METHOD       (default: greedy_search)
  USE_AVERAGED_MODEL    (default: 0)
  FINAL_DOWNSAMPLE      (default: 1)
  MAX_DURATION          (default: 1000)
  CUTS_NAME             (default: all)

Options:
  --cuda-visible-devices VALUE
  --epoch VALUE
  --avg VALUE
  --exp-dir PATH
  --manifest-dir PATH
  --bpe-model PATH
  --decoding-method METHOD
  --use-averaged-model {0,1}
  --final-downsample VALUE
  --max-duration VALUE
  --cuts-name NAME
  -h, --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-visible-devices)
      export CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --epoch)
      EPOCH="$2"
      shift 2
      ;;
    --avg)
      AVG="$2"
      shift 2
      ;;
    --exp-dir)
      EXP_DIR="$2"
      shift 2
      ;;
    --manifest-dir)
      MANIFEST_DIR="$2"
      shift 2
      ;;
    --bpe-model)
      BPE_MODEL="$2"
      shift 2
      ;;
    --decoding-method)
      DECODING_METHOD="$2"
      shift 2
      ;;
    --use-averaged-model)
      USE_AVERAGED_MODEL="$2"
      shift 2
      ;;
    --final-downsample)
      FINAL_DOWNSAMPLE="$2"
      shift 2
      ;;
    --max-duration)
      MAX_DURATION="$2"
      shift 2
      ;;
    --cuts-name)
      CUTS_NAME="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

python ./zipformer_fbank/decode.py \
    --epoch "${EPOCH}" \
    --avg "${AVG}" \
    --exp-dir "${EXP_DIR}" \
    --max-duration "${MAX_DURATION}" \
    --bpe-model "${BPE_MODEL}" \
    --decoding-method "${DECODING_METHOD}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --use-averaged-model "${USE_AVERAGED_MODEL}" \
    --final-downsample "${FINAL_DOWNSAMPLE}" \
    --cuts-name "${CUTS_NAME}"

