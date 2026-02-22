# waymo-open-dataset-utils

Local vendored helpers used by this project to access Waymo Open Dataset parquet
components with Waymo-style key joins.

Contents:
- `dataframe_utils.py`: vendored/adapted from `waymo_open_dataset.v2.dataframe_utils`
- `frame_utils.py`: frame-style row extractors for camera image/calibration fields
- `parquet_component_loader.py`: gs/local parquet IO + frame-key filtering + metadata repository

These modules are imported by:
- `visuals_dataset/generate_dataset.py`
- `visuals_dataset/cache_waymo_subset.py`
