import io
import os
from datasets import Audio, load_dataset
from pathlib import Path
import soundfile as sf

# Note: This script uses soundfile for audio decoding to avoid torchcodec/FFmpeg dependencies

def convert_hf_dataset_to_supervised(
    dataset_name,
    audio_column="audio",
    text_column="sentence",
    output_dir="download",
    samples_per_subdir=None
):
    """
    Simple script to convert Hugging Face dataset to supervised format.
    """
    print(f"Loading dataset: {dataset_name}")
    # Cast audio column to avoid torchcodec by keeping raw files/bytes
    dataset = load_dataset(dataset_name)
    
    # Disable built-in decoding so we can rely on soundfile instead of torchcodec
    for split_name in dataset.keys():
        dataset[split_name] = dataset[split_name].cast_column(
            audio_column, Audio(decode=False)
        )
   
    # Process each split
    for split_name, split_data in dataset.items():
        output_split = "dev" if split_name == "validation" else split_name
       
        print(f"\nProcessing {split_name} split ({len(split_data)} samples)...")
       
        # Determine subdirs
        if samples_per_subdir:
            num_subdirs = (len(split_data) + samples_per_subdir - 1) // samples_per_subdir
        else:
            num_subdirs = 1
       
        for subdir_idx in range(num_subdirs):
            subdir_name = f"data{subdir_idx + 1}"
            split_dir = Path(output_dir) / output_split / subdir_name
            split_dir.mkdir(parents=True, exist_ok=True)
           
            start_idx = subdir_idx * samples_per_subdir if samples_per_subdir else 0
            end_idx = min(start_idx + samples_per_subdir, len(split_data)) if samples_per_subdir else len(split_data)
           
            transcript_path = split_dir / "filename.trans.txt"
           
            with open(transcript_path, "w", encoding="utf-8") as trans_file:
                for idx in range(start_idx, end_idx):
                    sample = split_data[idx]
                   
                    audio_data = sample[audio_column]
                    text = sample[text_column]
                   
                    audio_filename = f"audio_{idx:06d}"
                    wav_path = split_dir / f"{audio_filename}.wav"
                   
                    # Handle audio data. With decode disabled, we read using soundfile.
                    audio_array = None
                    sampling_rate = None

                    if audio_data.get("array") is not None:
                        audio_array = audio_data["array"]
                        sampling_rate = audio_data["sampling_rate"]
                    else:
                        audio_path = audio_data.get("path")
                        audio_bytes = audio_data.get("bytes")

                        if audio_path and os.path.exists(audio_path):
                            audio_array, sampling_rate = sf.read(audio_path, dtype="float32")
                        elif audio_bytes:
                            audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                        else:
                            raise ValueError(
                                f"No readable audio data found for sample index {idx}: "
                                f"path={audio_path}, has_bytes={audio_bytes is not None}"
                            )
                   
                    # Ensure audio array is in correct format
                    # soundfile expects (num_samples,) for mono or (num_samples, channels) for multi-channel
                    if audio_array.ndim > 1 and audio_array.shape[1] == 1:
                        audio_array = audio_array.squeeze()
                   
                    sf.write(str(wav_path), audio_array, sampling_rate)
                   
                    trans_file.write(f"{audio_filename} {text}\n")
                   
                    if (idx - start_idx + 1) % 100 == 0:
                        print(f" [{subdir_name}] Processed {idx - start_idx + 1}/{end_idx - start_idx} samples...")
           
            print(f"✓ Completed {subdir_name}: {end_idx - start_idx} files saved to {split_dir}")
       
        print(f"✓ Completed {split_name} split: {len(split_data)} total files")
   
    print(f"\n✓ Dataset conversion complete! Data saved to: {output_dir}")

if __name__ == "__main__":
    convert_hf_dataset_to_supervised(
        dataset_name="galsenai/wolof-audio-data",
        audio_column="audio",
        text_column="sentence",
        output_dir="download"
    )