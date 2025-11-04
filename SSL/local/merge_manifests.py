#!/usr/bin/env python3
"""
Merge manifests from multiple volumes into a single manifest.
"""

import argparse
import logging
from pathlib import Path
from lhotse import RecordingSet, SupervisionSet, CutSet


def merge_manifests(
    manifest_dir: Path,
    output_name: str,
    num_volumes: int,
):
    """
    Merge recording and supervision manifests from multiple volumes.
    
    Args:
        manifest_dir: Directory containing volume manifests
        output_name: Name for the merged output files
        num_volumes: Number of volumes to merge
    """
    manifest_dir = Path(manifest_dir)
    
    logging.info(f"Merging manifests from {num_volumes} volumes...")
    
    all_recordings = []
    all_supervisions = []
    
    # Collect all volume manifests
    for vol_num in range(1, num_volumes + 1):
        recording_path = manifest_dir / f"vietASR-ssl_recordings_vol{vol_num}.jsonl.gz"
        supervision_path = manifest_dir / f"vietASR-ssl_supervisions_vol{vol_num}.jsonl.gz"
        
        if not recording_path.exists():
            logging.warning(f"Recording manifest not found: {recording_path}, skipping...")
            continue
            
        if not supervision_path.exists():
            logging.warning(f"Supervision manifest not found: {supervision_path}, skipping...")
            continue
        
        logging.info(f"Loading volume {vol_num}...")
        recordings = RecordingSet.from_jsonl_lazy(recording_path)
        supervisions = SupervisionSet.from_jsonl_lazy(supervision_path)
        
        all_recordings.append(recordings)
        all_supervisions.append(supervisions)
    
    if not all_recordings:
        raise ValueError("No valid manifests found to merge!")
    
    # Merge all recordings and supervisions
    logging.info("Combining all recordings...")
    merged_recordings = all_recordings[0]
    for recordings in all_recordings[1:]:
        merged_recordings = RecordingSet.from_items(
            list(merged_recordings) + list(recordings)
        )
    
    logging.info("Combining all supervisions...")
    merged_supervisions = all_supervisions[0]
    for supervisions in all_supervisions[1:]:
        merged_supervisions = SupervisionSet.from_segments(
            list(merged_supervisions) + list(supervisions)
        )
    
    # Save merged manifests
    output_recording = manifest_dir / f"vietASR-ssl_recordings_{output_name}.jsonl.gz"
    output_supervision = manifest_dir / f"vietASR-ssl_supervisions_{output_name}.jsonl.gz"
    
    logging.info(f"Saving merged recordings to: {output_recording}")
    merged_recordings.to_file(output_recording)
    
    logging.info(f"Saving merged supervisions to: {output_supervision}")
    merged_supervisions.to_file(output_supervision)
    
    logging.info(f"Merge complete! Total recordings: {len(merged_recordings)}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Merge manifests from multiple volumes")
    parser.add_argument(
        "--manifest-dir",
        type=str,
        required=True,
        help="Directory containing volume manifests"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Name for the merged output files"
    )
    parser.add_argument(
        "--num-volumes",
        type=int,
        required=True,
        help="Number of volumes to merge"
    )
    
    args = parser.parse_args()
    
    merge_manifests(
        manifest_dir=Path(args.manifest_dir),
        output_name=args.output_name,
        num_volumes=args.num_volumes,
    )
