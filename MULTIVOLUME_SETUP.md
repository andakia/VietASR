# Multi-Volume Data Setup Guide

This guide explains how to prepare your SSL pretraining data from multiple volumes.

## Your Data Structure

```
/mnt/pr_audio1/
├── subdir1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── subdir2/
│   └── ...
└── ...

/mnt/pr_audio2/
└── ...

...

/mnt/pr_audio20/
└── ...
```

## Quick Start

### 1. Prepare SSL Data from All Volumes

Run the multi-volume preparation script:

```bash
cd SSL
./prepare_ssl_multivolume.sh
```

This script will:
- Process all 20 volumes (`/mnt/pr_audio1` to `/mnt/pr_audio20`)
- Create manifests for each volume
- Merge them into a single manifest
- Extract fbank features
- Split the data for training

### 2. Customize Configuration (Optional)

You can modify parameters in `prepare_ssl_multivolume.sh`:

```bash
# Change volume range
start_volume=1
end_volume=20

# Change volume path prefix
volume_prefix="/mnt/pr_audio"

# Change number of parallel jobs
nj=16

# Change number of utterances per split
num_per_split=200000

# Change subset name
subset_name="train_multivolume"
```

### 3. Run Specific Stages

If you want to run only specific stages:

```bash
# Only create manifests (stage 1)
./prepare_ssl_multivolume.sh --stage 1 --stop-stage 1

# Only preprocess (stage 2)
./prepare_ssl_multivolume.sh --stage 2 --stop-stage 2

# Only split data (stage 4)
./prepare_ssl_multivolume.sh --stage 4 --stop-stage 4

# Only compute features (stage 5)
./prepare_ssl_multivolume.sh --stage 5 --stop-stage 5
```

### 4. Process Subset of Volumes

To process only specific volumes, modify the script:

```bash
./prepare_ssl_multivolume.sh --start-volume 1 --end-volume 10
```

## Output Structure

After running the script, you'll have:

```
SSL/data/
├── manifest_train_multivolume/
│   ├── vietASR-ssl_recordings_vol1.jsonl.gz
│   ├── vietASR-ssl_supervisions_vol1.jsonl.gz
│   ├── ...
│   ├── vietASR-ssl_recordings_train_multivolume.jsonl.gz  # Merged
│   └── vietASR-ssl_supervisions_train_multivolume.jsonl.gz  # Merged
└── ssl_train_multivolume/
    ├── vietASR-ssl_cuts_train_multivolume_raw.jsonl.gz
    └── train_multivolume_split/
        ├── vietASR-ssl_cuts_train_multivolume.00000001.jsonl.gz
        ├── vietASR-ssl_cuts_train_multivolume.00000002.jsonl.gz
        └── ...
```

## Next Steps

After data preparation, follow the main README for:

1. **Train k-means model** on a subset of your data
2. **Extract labels** using the k-means model
3. **Pre-train** the SSL model
4. **Fine-tune** on labeled data

## Troubleshooting

### Missing Volumes
If some volumes don't exist, the script will skip them with a warning and continue processing the rest.

### Memory Issues
If you run into memory issues:
- Reduce `nj` (number of parallel jobs)
- Reduce `num_per_split` to create more, smaller splits
- Process volumes in batches by adjusting `start_volume` and `end_volume`

### Disk Space
Make sure you have enough disk space in `SSL/data/` for:
- Manifests (relatively small)
- Feature files (can be large depending on your data size)

## Advanced: Process Volumes in Batches

If you have limited resources, process volumes in batches:

```bash
# Batch 1: volumes 1-5
./prepare_ssl_multivolume.sh --start-volume 1 --end-volume 5 --subset-name train_batch1

# Batch 2: volumes 6-10
./prepare_ssl_multivolume.sh --start-volume 6 --end-volume 10 --subset-name train_batch2

# ... and so on
```

Then you can train on each batch separately or merge them later.
