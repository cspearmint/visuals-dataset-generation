"""Frame-like extraction helpers for Waymo v2 parquet rows."""

from typing import Any, Dict, Optional


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_frame_key(segment_context_name: str, frame_timestamp_micros: Any) -> str:
    return "%s|%s" % (segment_context_name, _as_int(frame_timestamp_micros, 0))


def build_scene_name(segment_context_name: str, frame_timestamp_micros: Any) -> str:
    return "%s_%s" % (segment_context_name, _as_int(frame_timestamp_micros, 0))


def camera_image_row_to_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract image-level record from a CameraImageComponent parquet row."""
    segment = str(row.get("key.segment_context_name", "unknown"))
    timestamp = _as_int(row.get("key.frame_timestamp_micros", 0), 0)
    camera_name = str(row.get("key.camera_name", "unknown"))

    return {
        "frame_key": build_frame_key(segment, timestamp),
        "frame_id": build_scene_name(segment, timestamp),
        "timestamp_micros": timestamp,
        "segment_context_name": segment,
        "camera_name": camera_name,
        "image_bytes": row.get("[CameraImageComponent].image"),
        "velocity": {
            "linear_x": _as_float(row.get("[CameraImageComponent].velocity.linear_velocity.x")),
            "linear_y": _as_float(row.get("[CameraImageComponent].velocity.linear_velocity.y")),
            "linear_z": _as_float(row.get("[CameraImageComponent].velocity.linear_velocity.z")),
            "angular_x": _as_float(row.get("[CameraImageComponent].velocity.angular_velocity.x")),
            "angular_y": _as_float(row.get("[CameraImageComponent].velocity.angular_velocity.y")),
            "angular_z": _as_float(row.get("[CameraImageComponent].velocity.angular_velocity.z")),
        },
        "camera_timing": {
            "pose_timestamp": _as_float(row.get("[CameraImageComponent].pose_timestamp")),
            "shutter": _as_float(row.get("[CameraImageComponent].rolling_shutter_params.shutter")),
            "camera_trigger_time": _as_float(
                row.get("[CameraImageComponent].rolling_shutter_params.camera_trigger_time")
            ),
            "camera_readout_done_time": _as_float(
                row.get("[CameraImageComponent].rolling_shutter_params.camera_readout_done_time")
            ),
        },
        "camera_pose_transform": row.get("[CameraImageComponent].pose.transform"),
    }


def extract_camera_calibration(calibration_row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw camera_calibration row to compact per-camera payload."""
    if not isinstance(calibration_row, dict):
        return {}

    intrinsic_prefix = "[CameraCalibrationComponent].intrinsic."
    intrinsic = {}
    for key, value in calibration_row.items():
        if key.startswith(intrinsic_prefix):
            intrinsic[key[len(intrinsic_prefix) :]] = value

    return {
        "camera_name": calibration_row.get("key.camera_name"),
        "intrinsic": intrinsic,
        "extrinsic_transform": calibration_row.get("[CameraCalibrationComponent].extrinsic.transform"),
        "rolling_shutter_direction": calibration_row.get("[CameraCalibrationComponent].rolling_shutter_direction"),
        "height": calibration_row.get("[CameraCalibrationComponent].height"),
        "width": calibration_row.get("[CameraCalibrationComponent].width"),
    }

