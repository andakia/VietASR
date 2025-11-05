#!/usr/bin/env bash

# Script to prepare SSL data from multiple volumes
# Usage: ./prepare_ssl_multivolume.sh

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=16
# run step 1 to step 5 by default
stage=1
stop_stage=5

# Configuration for multi-volume data
volume_prefix="/mnt/pr_audio"
start_volume=1
end_volume=1
subset_name="train_multivolume"
lang=Wolof
num_per_split=200000

# Directories
manifest_dir=data/manifest_${subset_name}
fbank_dir=data/ssl_${subset_name}

. shared/parse_options.sh || exit 1

mkdir -p data

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Running prepare_ssl_multivolume.sh"
log "Processing volumes: ${volume_prefix}${start_volume} to ${volume_prefix}${end_volume}"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare manifests from multiple volumes"
  mkdir -p $manifest_dir
  
  if [ ! -e $manifest_dir/.vietASR.done ]; then
    # Process each volume
    for vol_num in $(seq $start_volume $end_volume); do
      vol_path="${volume_prefix}${vol_num}"
      
      if [ ! -d "$vol_path" ]; then
        log "Warning: Volume $vol_path does not exist, skipping..."
        continue
      fi
      
      log "Processing volume: $vol_path"
      python local/vietASR_ssl.py \
        --lang $lang \
        -j $nj \
        $vol_path \
        "vol${vol_num}" \
        $manifest_dir
    done
    
    # Merge all volume manifests
    log "Merging manifests from all volumes..."
    python local/merge_manifests.py \
      --manifest-dir $manifest_dir \
      --output-name $subset_name \
      --num-volumes $end_volume
    
    touch $manifest_dir/.vietASR.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Preprocess VietASR manifest"
  if [ ! -f $fbank_dir/.preprocess.done ]; then
    python3 ./local/preprocess_vietASR_ssl.py \
      --lang $lang \
      --dataset "$subset_name" \
      --src-dir $manifest_dir \
      --tgt-dir $fbank_dir
    touch $fbank_dir/.preprocess.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Split train set into pieces"
  log "Split subset: $subset_name"
  split_dir=$fbank_dir/${subset_name}_split
  if [ ! -f $split_dir/.split.done ]; then
    lhotse split-lazy \
      $fbank_dir/vietASR-ssl_cuts_${subset_name}_raw.jsonl.gz \
      $split_dir \
      $num_per_split
    touch $split_dir/.split.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute features for train set"
  python local/compute_fbank_vietASR_ssl_splits.py parallel \
    --src-dir $fbank_dir \
    --dataset $subset_name
fi

log "Done! Manifests saved to: $manifest_dir"
log "Features saved to: $fbank_dir"
