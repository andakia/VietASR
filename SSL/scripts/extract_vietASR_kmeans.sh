#! /usr/bin/bash

# Update the arguments below to match your environment or export overrides before calling the script.
# The task list must contain pairs of source and target cut paths, one pair per line:
#   /path/to/src_cuts.jsonl.gz /path/to/target_cuts_with_labels.jsonl.gz

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH

task_list=${task_list:-data/ssl_train_multivolume/train_multivolume_split/kmeans_task_list.txt}
km_model=${km_model:-kmeans.pt}
pretrained_dir=${pretrained_dir:-/mnt/training/VietASR/ASR/zipformer/exp}
epoch=${epoch:-20}
avg=${avg:-1}
max_duration=${max_duration:-500}
bpe_model=${bpe_model:-/mnt/training/VietASR/ASR/data/lang_bpe_2000/bpe.model}
checkpoint_type=${checkpoint_type:-ASR}
use_averaged_model=${use_averaged_model:-1}

python -m zipformer_fbank.extract_kmeans_scripts.extract_kmeans \
    --task-list "${task_list}" \
    --model-path "${km_model}" \
    --pretrained-dir "${pretrained_dir}" \
    --epoch "${epoch}" \
    --avg "${avg}" \
    --max-duration "${max_duration}" \
    --bpe-model "${bpe_model}" \
    --checkpoint-type "${checkpoint_type}" \
    --use-averaged-model "${use_averaged_model}"