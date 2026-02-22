"""
Cache a local subset of Waymo parquet files in the same layout as GCS.

The output layout mirrors:
  gs://waymo_open_dataset_v_2_0_1/<split>/<folder>/...

By default this writes to:
  ../dataset/waymo_open_dataset_v_2_0_1/...
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow.fs as pafs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAYMO_UTILS_DIR = PROJECT_ROOT / "waymo-open-dataset-utils"
if WAYMO_UTILS_DIR.exists():
    sys.path.insert(0, str(WAYMO_UTILS_DIR))

from parquet_component_loader import (  # type: ignore  # noqa: E402
    DATASET_BUCKET_NAME,
    DEFAULT_METADATA_COMPONENTS,
    infer_key_columns,
    list_parquet_files,
    read_parquet_table,
    read_schema_names,
    split_gs_path,
)

CLOUD_DATASET_BASE = "gs://waymo_open_dataset_v_2_0_1"
METADATA_FOLDERS = list(DEFAULT_METADATA_COMPONENTS)


def read_table(file_path: str, columns: Optional[List[str]] = None):
    """Read a parquet table from gs:// or local."""
    return read_parquet_table(file_path, columns=columns)


def normalize_dataset_output_root(output_root: Path) -> Path:
    """Ensure output root points to ../dataset (parent of bucket folder)."""
    if output_root.name == DATASET_BUCKET_NAME:
        return output_root.parent
    return output_root


def to_local_path(gcs_path: str, output_root: Path) -> Path:
    """
    Convert gs://waymo_open_dataset_v_2_0_1/... to
    <output_root>/waymo_open_dataset_v_2_0_1/...
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Expected gs:// path, got: {gcs_path}")
    relative = gcs_path[5:]
    return output_root / Path(relative)


def copy_gcs_to_local(gcs_path: str, output_root: Path, overwrite: bool = False) -> Tuple[Path, bool]:
    """Copy one gs:// parquet file into mirrored local path."""
    local_path = to_local_path(gcs_path, output_root)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and not overwrite:
        return local_path, False

    bucket, key = split_gs_path(gcs_path)
    fs = pafs.GcsFileSystem()
    with fs.open_input_stream(f"{bucket}/{key}") as src, open(local_path, "wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    return local_path, True


def collect_camera_subset(
    split: str,
    max_camera_files: Optional[int],
    camera_start: int,
    shuffle_seed: Optional[int],
    verbose: bool = False,
) -> Tuple[List[str], Set[str], Set[str]]:
    """
    Select camera parquet files and extract frame keys from those files.

    Returns:
      - selected camera parquet file paths
      - frame key set as "<segment>|<timestamp>"
      - segment context names
    """
    camera_root = f"{CLOUD_DATASET_BASE}/{split}/camera_image"
    camera_files = list_parquet_files(camera_root, verbose=verbose)
    if not camera_files:
        raise RuntimeError(f"No camera parquet files found in {camera_root}")

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(camera_files)
        if verbose:
            print(f"[debug] Shuffled camera files with seed {shuffle_seed}")

    if camera_start > 0:
        camera_files = camera_files[camera_start:]

    if max_camera_files is not None:
        camera_files = camera_files[:max_camera_files]

    if not camera_files:
        raise RuntimeError("No camera files selected after applying start/max limits")

    frame_keys: Set[str] = set()
    segments: Set[str] = set()

    for file_path in camera_files:
        schema_names = read_schema_names(file_path)
        segment_col, timestamp_col = infer_key_columns(schema_names)
        if segment_col is None:
            print(f"[warn] Missing segment key column in camera file: {file_path}")
            continue

        columns = [segment_col]
        if timestamp_col:
            columns.append(timestamp_col)

        table = read_table(file_path, columns=columns)
        df = table.to_pandas()
        if df.empty:
            continue

        segments.update(df[segment_col].dropna().astype(str).tolist())

        if timestamp_col:
            timestamps = pd.to_numeric(df[timestamp_col], errors="coerce")
            valid_mask = timestamps.notna()
            if valid_mask.any():
                keys = (
                    df.loc[valid_mask, segment_col].astype(str)
                    + "|"
                    + timestamps.loc[valid_mask].astype("int64").astype(str)
                )
                frame_keys.update(keys.tolist())

    return camera_files, frame_keys, segments


def metadata_file_is_related(
    file_path: str,
    frame_keys: Set[str],
    segments: Set[str],
) -> bool:
    """
    Check if a metadata parquet file contains rows matching selected frames.
    """
    schema_names = read_schema_names(file_path)
    segment_col, timestamp_col = infer_key_columns(schema_names)
    if segment_col is None:
        return False

    columns = [segment_col]
    if timestamp_col:
        columns.append(timestamp_col)

    table = read_table(file_path, columns=columns)
    df = table.to_pandas()
    if df.empty:
        return False

    if timestamp_col and frame_keys:
        timestamps = pd.to_numeric(df[timestamp_col], errors="coerce")
        valid_mask = timestamps.notna()
        if valid_mask.any():
            keys = (
                df.loc[valid_mask, segment_col].astype(str)
                + "|"
                + timestamps.loc[valid_mask].astype("int64").astype(str)
            )
            if keys.isin(frame_keys).any():
                return True

    if segments:
        if df[segment_col].dropna().astype(str).isin(segments).any():
            return True

    return False


def find_related_metadata_files(
    split: str,
    frame_keys: Set[str],
    segments: Set[str],
    max_metadata_files_per_folder: Optional[int],
    verbose: bool = False,
) -> Dict[str, List[str]]:
    """
    Scan metadata folders and keep parquet files that contain selected frames.
    """
    related: Dict[str, List[str]] = {}
    for folder in METADATA_FOLDERS:
        folder_root = f"{CLOUD_DATASET_BASE}/{split}/{folder}"
        files = list_parquet_files(folder_root, verbose=verbose)
        if max_metadata_files_per_folder is not None:
            files = files[:max_metadata_files_per_folder]

        matched: List[str] = []
        for idx, file_path in enumerate(files):
            try:
                if metadata_file_is_related(file_path, frame_keys=frame_keys, segments=segments):
                    matched.append(file_path)
            except Exception as exc:
                if verbose:
                    print(f"[debug] Failed metadata scan for {file_path}: {exc}")

            if verbose and (idx + 1) % 100 == 0:
                print(f"[debug] Scanned {idx + 1}/{len(files)} files in {folder}")

        related[folder] = matched
        print(f"[info] Related metadata files in {folder}: {len(matched)}")
    return related


def write_manifest(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a Waymo subset from GCS to local dataset mirror."
    )
    parser.add_argument("--split", default="training", help="Dataset split.")
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent.parent / "dataset"),
        help="Local root folder that will contain waymo_open_dataset_v_2_0_1.",
    )
    parser.add_argument("--max-camera-files", type=int, default=5, help="Number of camera parquet files to copy.")
    parser.add_argument("--camera-start", type=int, default=0, help="Skip this many camera parquet files before selecting.")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Optional camera parquet shuffle seed.")
    parser.add_argument(
        "--max-metadata-files-per-folder",
        type=int,
        default=None,
        help="Limit metadata files scanned per folder (default scans all).",
    )
    parser.add_argument("--manifest-name", default="cache_manifest.json", help="Manifest filename written in output-root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local files.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")

    args = parser.parse_args()
    output_root = normalize_dataset_output_root(Path(args.output_root).resolve())

    print(f"[info] Output root: {output_root}")
    print(f"[info] Split: {args.split}")

    camera_files, frame_keys, segments = collect_camera_subset(
        split=args.split,
        max_camera_files=args.max_camera_files,
        camera_start=args.camera_start,
        shuffle_seed=args.shuffle_seed,
        verbose=args.verbose,
    )
    print(f"[info] Selected camera parquet files: {len(camera_files)}")
    print(f"[info] Extracted frame keys: {len(frame_keys)}")

    related_metadata = find_related_metadata_files(
        split=args.split,
        frame_keys=frame_keys,
        segments=segments,
        max_metadata_files_per_folder=args.max_metadata_files_per_folder,
        verbose=args.verbose,
    )

    copied_camera: List[Path] = []
    copied_metadata: Dict[str, List[Path]] = {folder: [] for folder in METADATA_FOLDERS}

    for camera_file in camera_files:
        local_path, copied = copy_gcs_to_local(camera_file, output_root=output_root, overwrite=args.overwrite)
        if copied:
            copied_camera.append(local_path)

    for folder, files in related_metadata.items():
        for meta_file in files:
            local_path, copied = copy_gcs_to_local(meta_file, output_root=output_root, overwrite=args.overwrite)
            if copied:
                copied_metadata[folder].append(local_path)

    manifest_path = output_root / args.manifest_name
    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_dataset_base": CLOUD_DATASET_BASE,
        "split": args.split,
        "output_root": str(output_root),
        "local_dataset_base": str(output_root / DATASET_BUCKET_NAME),
        "selection": {
            "max_camera_files": args.max_camera_files,
            "camera_start": args.camera_start,
            "shuffle_seed": args.shuffle_seed,
            "max_metadata_files_per_folder": args.max_metadata_files_per_folder,
            "overwrite": args.overwrite,
        },
        "counts": {
            "camera_files_selected": len(camera_files),
            "camera_files_copied": len(copied_camera),
            "frame_keys": len(frame_keys),
            "segments": len(segments),
            "metadata_files_selected": sum(len(v) for v in related_metadata.values()),
            "metadata_files_copied": sum(len(v) for v in copied_metadata.values()),
        },
        "camera_files_selected": camera_files,
        "metadata_files_selected_by_folder": related_metadata,
        "camera_files_copied": [str(p) for p in copied_camera],
        "metadata_files_copied_by_folder": {
            folder: [str(p) for p in paths] for folder, paths in copied_metadata.items()
        },
    }
    write_manifest(manifest_path, manifest)

    print("[success] Local subset cache complete.")
    print(f"  - Local dataset base: {output_root / DATASET_BUCKET_NAME}")
    print(f"  - Camera files copied: {len(copied_camera)}")
    print(f"  - Metadata files copied: {sum(len(v) for v in copied_metadata.values())}")
    print(f"  - Manifest: {manifest_path}")
    print("")
    print("[next] Run local-mode pipeline (default):")
    print("  python generate_dataset.py --split training --count 10 --max-files 3")
    print("[next] Run cloud-mode pipeline:")
    print("  python generate_dataset.py -cloud --split training --count 10 --max-files 3")


if __name__ == "__main__":
    main()
