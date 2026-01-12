"""
Waymo Open Dataset v2.0.1 - Parquet-based image fetcher with augmentation and metadata.

Reads parquet files from Waymo dataset, applies image augmentations, saves augmented
images to output folders by degradation type, and saves individual metadata JSON per image.

Requires:
- Google Application Default Credentials (ADC):
    gcloud auth application-default login
- pyarrow, pandas, tensorflow (for GCS access)
"""

import argparse
import json
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

import pyarrow.parquet as pq
import pyarrow.fs as pafs
import pandas as pd
import tensorflow as tf

from augment import ImageAugmenter, decode_image, encode_image

BUCKET_BASE = "gs://waymo_open_dataset_v_2_0_1"

# Metadata folders to read from
METADATA_FOLDERS = [
    "stats",
    "vehicle_pose",
    "camera_calibration",
    "lidar_calibration",
    "camera_box",
    "lidar_box",
    "camera_segmentation",
    "lidar_segmentation",
]


def list_parquet_files(bucket_path: str, verbose: bool = False) -> List[str]:
    """List all parquet files in a GCS bucket path."""
    try:
        # Try both patterns: files directly in folder, and in subdirectories
        files = tf.io.gfile.glob(f"{bucket_path}/*.parquet")
        if not files:
            files = tf.io.gfile.glob(f"{bucket_path}/**/*.parquet")
        if verbose:
            print(f"[debug] Found {len(files)} parquet files in {bucket_path}")
        return sorted(files)
    except Exception as e:
        print(f"[error] Failed to list files in {bucket_path}: {e}")
        return []


def read_parquet_file(file_path: str, verbose: bool = False) -> Optional[pd.DataFrame]:
    """Read a single parquet file from GCS."""
    try:
        # Read directly using pyarrow with GCS filesystem
        
        # Parse GCS path
        if file_path.startswith('gs://'):
            bucket, key = file_path[5:].split('/', 1)
            fs = pafs.GcsFileSystem()
            table = pq.read_table(f"{bucket}/{key}", filesystem=fs)
        else:
            table = pq.read_table(file_path)
        
        df = table.to_pandas()
        if verbose:
            print(f"[debug] Read {file_path}: {len(df)} rows, {len(df.columns)} cols")
        return df
    except Exception as e:
        print(f"[warn] Failed to read {file_path}: {e}")
        return None


def fetch_metadata_for_frame(
    segment: str,
    timestamp_micros: int,
    split: str = "training",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fetch metadata for a specific frame from all metadata folders.
    Returns a dict with all available metadata keyed by folder name.
    """
    metadata = {}
    
    for folder in METADATA_FOLDERS:
        folder_path = f"{BUCKET_BASE}/{split}/{folder}"
        try:
            # List files in folder
            files = list_parquet_files(folder_path, verbose=False)
            if not files:
                if verbose:
                    print(f"[debug] No files found in {folder}")
                continue
            
            # Limit files to check per folder (avoid endless loops)
            files_to_check = files[:10]  # Only check first 10 files per folder
            
            if verbose:
                print(f"[debug] Checking {len(files_to_check)} files in {folder}...")
            
            # Read each file and look for matching rows
            for file_idx, file_path in enumerate(files_to_check):
                df = read_parquet_file(file_path, verbose=False)
                if df is None or len(df) == 0:
                    continue
                
                # Try to find matching rows by segment and timestamp
                # Different folders have different key column names
                segment_col = None
                timestamp_col = None
                
                for col in df.columns:
                    if 'segment' in col.lower() or 'context_name' in col.lower():
                        segment_col = col
                    if 'timestamp' in col.lower() and 'micros' in col.lower():
                        timestamp_col = col
                
                if segment_col and timestamp_col:
                    matching_rows = df[(df[segment_col] == segment) & (df[timestamp_col] == timestamp_micros)]
                    if len(matching_rows) > 0:
                        # Store all matching rows' data
                        if folder not in metadata:
                            metadata[folder] = []
                        for _, row in matching_rows.iterrows():
                            metadata[folder].append(row.to_dict())
                        if verbose:
                            print(f"[debug]   Found {len(matching_rows)} rows in {folder}")
                        break  # Found match, move to next folder
                else:
                    # Fallback: try to match by just segment if available
                    if segment_col:
                        matching_rows = df[df[segment_col] == segment]
                        if len(matching_rows) > 0 and folder not in metadata:
                            metadata[folder] = []
                            for _, row in matching_rows.iterrows():
                                metadata[folder].append(row.to_dict())
                            if verbose:
                                print(f"[debug]   Found {len(matching_rows)} rows by segment in {folder}")
                            break  # Found match, move to next folder
        except KeyboardInterrupt:
            print(f"[warn] Interrupted while fetching metadata from {folder}")
            raise
        except Exception as e:
            if verbose:
                print(f"[debug] Error fetching metadata from {folder}: {e}")
            continue
    
    return metadata


def fetch_camera_images(
    split: str = "training",
    max_files: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch camera images from parquet files.
    Returns list of records with image bytes + metadata.
    """
    split_path = f"{BUCKET_BASE}/{split}/camera_image"
    if verbose:
        print(f"[info] Reading camera images from {split_path}...")

    image_files = list_parquet_files(split_path, verbose=verbose)
    if not image_files:
        print(f"[warn] No camera image files found in {split_path}")
        return []

    if max_files:
        image_files = image_files[:max_files]

    results = []
    for file_path in image_files:
        df = read_parquet_file(file_path, verbose=verbose)
        if df is None or len(df) == 0:
            continue

        for idx, row in df.iterrows():
            # Build frame key from key columns
            segment = row.get("key.segment_context_name", "unknown")
            timestamp = row.get("key.frame_timestamp_micros", 0)
            frame_key = f"{segment}_{timestamp}"

            # Get image bytes - column name is "[CameraImageComponent].image"
            image_bytes = None
            if "[CameraImageComponent].image" in df.columns:
                image_bytes = row.get("[CameraImageComponent].image")
            
            if image_bytes is None:
                if verbose:
                    print(f"[warn] No image bytes found in row {idx} of {file_path}")
                continue

            # Extract velocity and pose data
            velocity_x = row.get("[CameraImageComponent].velocity.linear_velocity.x", None)
            velocity_y = row.get("[CameraImageComponent].velocity.linear_velocity.y", None)
            velocity_z = row.get("[CameraImageComponent].velocity.linear_velocity.z", None)
            
            # Build record
            record = {
                "frame_id": frame_key,
                "timestamp_micros": timestamp,
                "segment_context_name": segment,
                "camera_name": row.get("key.camera_name", "unknown"),
                "image_bytes": image_bytes,
                "velocity": {
                    "linear_x": float(velocity_x) if velocity_x is not None else None,
                    "linear_y": float(velocity_y) if velocity_y is not None else None,
                    "linear_z": float(velocity_z) if velocity_z is not None else None,
                },
                "raw_data": row.to_dict(),  # Store all columns for metadata
            }
            results.append(record)

    return results


# Weather filtering removed — metadata lookups were unreliable.
# The function was intentionally removed to avoid runtime hangs and missing matches.
# If needed in future, re-implement with robust metadata caching and error handling.



def apply_degradations_and_save(
    images: List[Dict[str, Any]],
    output_dir: str,
    augmenter: ImageAugmenter,
    image_format: str = "JPEG",
    split: str = "training",
    verbose: bool = False,
) -> List[Path]:
    """
    Apply augmentations to images, save to degradation-specific subfolders,
    and save individual metadata JSON for each image (including all metadata from other folders).
    """
    out_dir = Path(output_dir)
    written: List[Path] = []

    for idx, record in enumerate(images):
        base_name = f"{record['segment_context_name']}_{record['timestamp_micros']}_{idx:03d}"

        # Decode image
        try:
            pil_img = decode_image(record["image_bytes"])
        except Exception as e:
            print(f"[warn] Failed to decode image {base_name}: {e}")
            continue

        # Fetch comprehensive metadata for this frame
        if verbose:
            print(f"[debug] Fetching metadata for {base_name}...")
        frame_metadata = fetch_metadata_for_frame(
            record['segment_context_name'],
            record['timestamp_micros'],
            split=split,
            verbose=verbose
        )

        # Prepare metadata directory
        meta_dir = out_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # Save the clear (original) base image as well
        try:
            clear_dir = out_dir / "clear"
            clear_dir.mkdir(parents=True, exist_ok=True)
            clear_path = clear_dir / f"{base_name}.{image_format.lower()}"
            clear_path.write_bytes(encode_image(pil_img, format=image_format))
            written.append(clear_path)

            clear_meta_path = meta_dir / f"{base_name}_clear.json"
            # Write the full collected metadata once per clear image
            clear_metadata = {
                "generated_at_utc": datetime.utcnow().isoformat() + "Z",
                "dataset_version": "waymo_open_dataset_v_2_0_1",
                "frame_id": record["frame_id"],
                "timestamp_micros": record["timestamp_micros"],
                "segment_context_name": record["segment_context_name"],
                "camera_name": record["camera_name"],
                "velocity": record["velocity"],
                "all_metadata": frame_metadata,
            }
            with open(clear_meta_path, "w") as f:
                json.dump(clear_metadata, f, indent=2, default=str)
            written.append(clear_meta_path)
        except Exception as e:
            print(f"[warn] Failed to save clear image/metadata for {base_name}: {e}")

        # Apply augmentations
        aug_map = augmenter.apply_all(pil_img)

        # Save each augmented version (no per-augmentation metadata files — only one per clear image)
        for aug_name, aug_img in aug_map.items():
            dest_dir = out_dir / aug_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            img_path = dest_dir / f"{base_name}.{image_format.lower()}"
            img_path.write_bytes(encode_image(aug_img, format=image_format))
            written.append(img_path)

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Waymo images, apply augmentations, and save with metadata."
    )
    parser.add_argument("--split", default="training", help="Dataset split (training/validation/test).")
    parser.add_argument("--count", type=int, default=10, help="Max images to fetch.")
    parser.add_argument("--output-dir", default="output", help="Where to write images and metadata.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit parquet files scanned.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for augmentations.")
    parser.add_argument("--random-start", type=int, default=None, help="Random starting point in image list (shuffles images using this seed).")
    parser.add_argument("--verbose", action="store_true", help="Print debug info.")

    args = parser.parse_args()

    print(f"[info] Fetching images from split: {args.split}")
    images = fetch_camera_images(
        split=args.split,
        max_files=args.max_files,
        verbose=args.verbose,
    )

    if not images:
        print("[error] No images found. Check split and credentials.")
        return

    print(f"[info] Fetched {len(images)} images")

    # Weather filtering removed (metadata lookup unreliable)
    if args.verbose:
        print("[info] Skipping weather filtering; using all images fetched from parquet files")

    # Optionally shuffle and pick random subset
    if args.random_start is not None:
        rng = random.Random(args.random_start)
        rng.shuffle(images)
        if args.verbose:
            print(f"[info] Shuffled images using seed {args.random_start}")

    # Limit to requested count
    images = images[:args.count]
    
    # Create augmenter
    augmenter = ImageAugmenter(seed=args.seed)

    # Apply augmentations and save
    paths = apply_degradations_and_save(
        images,
        args.output_dir,
        augmenter=augmenter,
        split=args.split,
        verbose=args.verbose,
    )

    print(f"[success] Wrote {len(paths)} files under {args.output_dir}")
    print(f"  - Augmented images in: rain/, fog/, snow/, motion_blur/")
    print(f"  - Metadata JSON in: metadata/")


if __name__ == "__main__":
    main()
