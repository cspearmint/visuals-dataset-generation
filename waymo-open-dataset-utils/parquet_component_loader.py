"""Waymo parquet IO and frame-keyed metadata table helpers."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from dataframe_utils import merge as merge_dataframes


KEY_SEGMENT = "key.segment_context_name"
KEY_TIMESTAMP = "key.frame_timestamp_micros"
KEY_CAMERA_NAME = "key.camera_name"

DATASET_BUCKET_NAME = "waymo_open_dataset_v_2_0_1"

DEFAULT_METADATA_COMPONENTS = [
    "stats",
    "vehicle_pose",
    "camera_calibration",
    "lidar_calibration",
    "camera_box",
    "lidar_box",
]

SCENE_LEVEL_COMPONENTS = [
    "stats",
    "vehicle_pose",
    "lidar_calibration",
    "lidar_box",
]

CAMERA_LEVEL_COMPONENTS = [
    "camera_calibration",
    "camera_box",
]


def split_gs_path(path: str) -> Tuple[str, str]:
    if not path.startswith("gs://"):
        raise ValueError("Expected gs:// path, got: %s" % path)
    no_scheme = path[5:]
    if "/" not in no_scheme:
        return no_scheme, ""
    bucket, prefix = no_scheme.split("/", 1)
    return bucket, prefix


def resolve_local_dataset_base(dataset_root: Path) -> Path:
    """Accept .../dataset or .../dataset/waymo_open_dataset_v_2_0_1."""
    if dataset_root.name == DATASET_BUCKET_NAME:
        return dataset_root
    return dataset_root / DATASET_BUCKET_NAME


def list_parquet_files(base_path: str, verbose: bool = False) -> List[str]:
    """List parquet files for local path or gs:// prefix."""
    try:
        if base_path.startswith("gs://"):
            bucket, prefix = split_gs_path(base_path)
            fs = pafs.GcsFileSystem()
            selector = pafs.FileSelector("%s/%s" % (bucket, prefix), recursive=True, allow_not_found=True)
            infos = fs.get_file_info(selector)
            files = [
                "gs://%s" % info.path
                for info in infos
                if info.type == pafs.FileType.File and info.path.endswith(".parquet")
            ]
        else:
            root = Path(base_path)
            files = [str(p) for p in root.rglob("*.parquet")] if root.exists() else []
        files = sorted(files)
        if verbose:
            print("[debug] Found %d parquet files in %s" % (len(files), base_path))
        return files
    except Exception as exc:
        if verbose:
            print("[warn] Failed to list parquet files in %s: %s" % (base_path, exc))
        return []


def read_parquet_table(file_path: str, columns: Optional[List[str]] = None):
    """Read one parquet table from local path or gs://."""
    if file_path.startswith("gs://"):
        bucket, key = split_gs_path(file_path)
        fs = pafs.GcsFileSystem()
        return pq.read_table("%s/%s" % (bucket, key), filesystem=fs, columns=columns)
    return pq.read_table(file_path, columns=columns)


def read_parquet_dataframe(
    file_path: str, columns: Optional[List[str]] = None, verbose: bool = False
) -> Optional[pd.DataFrame]:
    try:
        table = read_parquet_table(file_path, columns=columns)
        df = table.to_pandas()
        if verbose:
            print("[debug] Read %s rows=%d cols=%d" % (file_path, len(df), len(df.columns)))
        return df
    except Exception as exc:
        if verbose:
            print("[warn] Failed to read %s: %s" % (file_path, exc))
        return None


def read_schema_names(file_path: str) -> List[str]:
    if file_path.startswith("gs://"):
        bucket, key = split_gs_path(file_path)
        fs = pafs.GcsFileSystem()
        schema = pq.read_schema("%s/%s" % (bucket, key), filesystem=fs)
    else:
        schema = pq.read_schema(file_path)
    return list(schema.names)


def infer_key_columns(column_names: List[str]) -> Tuple[Optional[str], Optional[str]]:
    segment_col = None
    timestamp_col = None
    for name in column_names:
        lname = name.lower()
        if segment_col is None and ("segment" in lname or "context_name" in lname):
            segment_col = name
        if timestamp_col is None and ("timestamp" in lname and "micros" in lname):
            timestamp_col = name
    return segment_col, timestamp_col


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if KEY_SEGMENT in out.columns:
        out[KEY_SEGMENT] = out[KEY_SEGMENT].astype(str)
    if KEY_TIMESTAMP in out.columns:
        out[KEY_TIMESTAMP] = pd.to_numeric(out[KEY_TIMESTAMP], errors="coerce").astype("Int64")
    if KEY_CAMERA_NAME in out.columns:
        out[KEY_CAMERA_NAME] = out[KEY_CAMERA_NAME].astype(str)
    return out


def build_frame_keys_dataframe_from_records(image_records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            KEY_SEGMENT: str(r.get("segment_context_name", "")),
            KEY_TIMESTAMP: r.get("timestamp_micros"),
        }
        for r in image_records
    ]
    if not rows:
        return pd.DataFrame(columns=[KEY_SEGMENT, KEY_TIMESTAMP])
    keys = pd.DataFrame(rows)
    keys = _normalize_keys(keys)
    keys = keys.dropna(subset=[KEY_SEGMENT, KEY_TIMESTAMP]).drop_duplicates()
    return keys


def build_segment_keys_dataframe(frame_keys_df: pd.DataFrame) -> pd.DataFrame:
    if frame_keys_df.empty or KEY_SEGMENT not in frame_keys_df.columns:
        return pd.DataFrame(columns=[KEY_SEGMENT])
    segment_df = frame_keys_df[[KEY_SEGMENT]].dropna().drop_duplicates()
    segment_df[KEY_SEGMENT] = segment_df[KEY_SEGMENT].astype(str)
    return segment_df


def _concat_dataframes(parts: List[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts, ignore_index=True)


def load_component_dataframe(
    dataset_base: str,
    split: str,
    component: str,
    segments: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    root = "%s/%s/%s" % (dataset_base, split, component)
    files = list_parquet_files(root, verbose=verbose)
    if segments:
        segment_tokens = [str(s) for s in segments if s]
        if segment_tokens:
            segment_filtered = []
            for file_path in files:
                file_name = Path(file_path).name
                if any(token in file_name for token in segment_tokens):
                    segment_filtered.append(file_path)
            if segment_filtered:
                files = segment_filtered
            if verbose:
                print(
                    "[debug] Segment-filtered files for %s: %d"
                    % (component, len(files))
                )
    if max_files is not None:
        files = files[:max_files]
    frames = []
    for file_path in files:
        df = read_parquet_dataframe(file_path, verbose=verbose)
        if df is not None and not df.empty:
            frames.append(df)
    joined = _concat_dataframes(frames)
    if verbose:
        print("[debug] Component %s rows=%d files=%d" % (component, len(joined), len(files)))
    return joined


def filter_component_dataframe_to_selection(
    component_df: pd.DataFrame,
    frame_keys_df: pd.DataFrame,
    segment_keys_df: pd.DataFrame,
) -> pd.DataFrame:
    if component_df.empty:
        return component_df

    component_df = _normalize_keys(component_df)
    if KEY_SEGMENT not in component_df.columns:
        return component_df.iloc[0:0]

    has_timestamp = KEY_TIMESTAMP in component_df.columns and not component_df[KEY_TIMESTAMP].isna().all()
    if has_timestamp:
        frame_keys_df = _normalize_keys(frame_keys_df)
        return merge_dataframes(frame_keys_df, component_df, key_prefix="key.")

    segment_keys_df = _normalize_keys(segment_keys_df)
    return merge_dataframes(segment_keys_df, component_df, key_prefix="key.")


def load_filtered_component_tables(
    dataset_base: str,
    split: str,
    components: Iterable[str],
    frame_keys_df: pd.DataFrame,
    max_files_per_component: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    segment_keys_df = build_segment_keys_dataframe(frame_keys_df)
    selected_segments = (
        segment_keys_df[KEY_SEGMENT].dropna().astype(str).tolist()
        if KEY_SEGMENT in segment_keys_df.columns
        else []
    )
    tables = {}
    for component in components:
        raw_df = load_component_dataframe(
            dataset_base=dataset_base,
            split=split,
            component=component,
            segments=selected_segments,
            max_files=max_files_per_component,
            verbose=verbose,
        )
        tables[component] = filter_component_dataframe_to_selection(raw_df, frame_keys_df, segment_keys_df)
        if verbose:
            print("[debug] Filtered component %s rows=%d" % (component, len(tables[component])))
    return tables


class FrameMetadataRepository:
    """Caches frame-level component rows keyed by segment+timestamp."""

    def __init__(self, component_tables: Dict[str, pd.DataFrame], verbose: bool = False):
        self.component_tables = {k: _normalize_keys(v) for k, v in component_tables.items()}
        self.verbose = verbose
        self._cache = {}

    @staticmethod
    def _frame_key(segment_context_name: str, frame_timestamp_micros: Any) -> str:
        return "%s|%s" % (segment_context_name, int(frame_timestamp_micros))

    def _select_rows(self, df: pd.DataFrame, segment_context_name: str, frame_timestamp_micros: int):
        if df.empty or KEY_SEGMENT not in df.columns:
            return []
        mask = df[KEY_SEGMENT].astype(str) == str(segment_context_name)
        if KEY_TIMESTAMP in df.columns:
            mask = mask & (pd.to_numeric(df[KEY_TIMESTAMP], errors="coerce") == int(frame_timestamp_micros))
        return df.loc[mask].to_dict(orient="records")

    def get_frame_components(self, segment_context_name: str, frame_timestamp_micros: int) -> Dict[str, List[Dict[str, Any]]]:
        key = self._frame_key(segment_context_name, frame_timestamp_micros)
        if key in self._cache:
            return self._cache[key]

        payload = {}
        for component_name, df in self.component_tables.items():
            payload[component_name] = self._select_rows(df, segment_context_name, frame_timestamp_micros)
        self._cache[key] = payload
        if self.verbose:
            print("[debug] Prepared component metadata for frame %s" % key)
        return payload
