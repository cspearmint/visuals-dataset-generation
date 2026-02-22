"""
LiDAR-camera association utilities.

Association is derived directly from geometric projection and IoU matching.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def _camera_object_id(row: Dict[str, Any]) -> Optional[str]:
    object_id = row.get("camera_object_id")
    if object_id is None:
        object_id = row.get("key.camera_object_id")
    if object_id is None:
        return None
    return str(object_id)


def _camera_object_type(row: Dict[str, Any]) -> Any:
    if "type" in row:
        return row.get("type")
    return row.get("[CameraBoxComponent].type")


def _camera_object_xyxy(row: Dict[str, Any]) -> Optional[List[float]]:
    box_2d = row.get("box_2d")
    if isinstance(box_2d, dict):
        x1 = _safe_float(box_2d.get("x1"))
        y1 = _safe_float(box_2d.get("y1"))
        x2 = _safe_float(box_2d.get("x2"))
        y2 = _safe_float(box_2d.get("y2"))
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]

        center_x = _safe_float(box_2d.get("center_x"))
        center_y = _safe_float(box_2d.get("center_y"))
        size_x = _safe_float(box_2d.get("size_x"))
        size_y = _safe_float(box_2d.get("size_y"))
        if (
            center_x is not None
            and center_y is not None
            and size_x is not None
            and size_y is not None
            and size_x > 0
            and size_y > 0
        ):
            x1 = center_x - 0.5 * size_x
            y1 = center_y - 0.5 * size_y
            x2 = center_x + 0.5 * size_x
            y2 = center_y + 0.5 * size_y
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]

    center_x = _safe_float(row.get("[CameraBoxComponent].box.center.x"))
    center_y = _safe_float(row.get("[CameraBoxComponent].box.center.y"))
    size_x = _safe_float(row.get("[CameraBoxComponent].box.size.x"))
    size_y = _safe_float(row.get("[CameraBoxComponent].box.size.y"))
    if (
        center_x is None
        or center_y is None
        or size_x is None
        or size_y is None
        or size_x <= 0
        or size_y <= 0
    ):
        return None
    x1 = center_x - 0.5 * size_x
    y1 = center_y - 0.5 * size_y
    x2 = center_x + 0.5 * size_x
    y2 = center_y + 0.5 * size_y
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _xyxy_to_center_size(xyxy: List[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = xyxy
    size_x = x2 - x1
    size_y = y2 - y1
    center_x = x1 + 0.5 * size_x
    center_y = y1 + 0.5 * size_y
    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "size_x": float(size_x),
        "size_y": float(size_y),
    }


def _xyxy_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _camera_calibration_to_projection_params(
    calibration_row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    fu = _safe_float(calibration_row.get("[CameraCalibrationComponent].intrinsic.f_u"))
    fv = _safe_float(calibration_row.get("[CameraCalibrationComponent].intrinsic.f_v"))
    cu = _safe_float(calibration_row.get("[CameraCalibrationComponent].intrinsic.c_u"))
    cv = _safe_float(calibration_row.get("[CameraCalibrationComponent].intrinsic.c_v"))
    if fu is None or fv is None or cu is None or cv is None:
        return None

    transform = calibration_row.get("[CameraCalibrationComponent].extrinsic.transform")
    if not isinstance(transform, (list, tuple, np.ndarray)) or len(transform) != 16:
        return None
    try:
        vehicle_from_camera = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        camera_from_vehicle = np.linalg.inv(vehicle_from_camera)
    except Exception:
        return None

    return {
        "camera_name": str(calibration_row.get("key.camera_name")),
        "fu": fu,
        "fv": fv,
        "cu": cu,
        "cv": cv,
        "k1": _safe_float(calibration_row.get("[CameraCalibrationComponent].intrinsic.k1")) or 0.0,
        "k2": _safe_float(calibration_row.get("[CameraCalibrationComponent].intrinsic.k2")) or 0.0,
        "p1": _safe_float(calibration_row.get("[CameraCalibrationComponent].intrinsic.p1")) or 0.0,
        "p2": _safe_float(calibration_row.get("[CameraCalibrationComponent].intrinsic.p2")) or 0.0,
        "k3": _safe_float(calibration_row.get("[CameraCalibrationComponent].intrinsic.k3")) or 0.0,
        "width": _safe_float(calibration_row.get("[CameraCalibrationComponent].width")),
        "height": _safe_float(calibration_row.get("[CameraCalibrationComponent].height")),
        "camera_from_vehicle": camera_from_vehicle,
    }


def _camera_calibration_payload_to_row(camera_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    intrinsic = payload.get("intrinsic", {})
    if not isinstance(intrinsic, dict):
        intrinsic = {}
    row: Dict[str, Any] = {
        "key.camera_name": str(camera_name),
        "[CameraCalibrationComponent].extrinsic.transform": payload.get("extrinsic_transform"),
        "[CameraCalibrationComponent].rolling_shutter_direction": payload.get(
            "rolling_shutter_direction"
        ),
        "[CameraCalibrationComponent].height": payload.get("height"),
        "[CameraCalibrationComponent].width": payload.get("width"),
    }
    for name in ["f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3"]:
        row["[CameraCalibrationComponent].intrinsic.%s" % name] = intrinsic.get(name)
    return row


def _lidar_box_corners_vehicle_frame(lidar_box_row: Dict[str, Any]) -> Optional[np.ndarray]:
    center_x = _safe_float(lidar_box_row.get("[LiDARBoxComponent].box.center.x"))
    center_y = _safe_float(lidar_box_row.get("[LiDARBoxComponent].box.center.y"))
    center_z = _safe_float(lidar_box_row.get("[LiDARBoxComponent].box.center.z"))
    size_x = _safe_float(lidar_box_row.get("[LiDARBoxComponent].box.size.x"))
    size_y = _safe_float(lidar_box_row.get("[LiDARBoxComponent].box.size.y"))
    size_z = _safe_float(lidar_box_row.get("[LiDARBoxComponent].box.size.z"))
    heading = _safe_float(lidar_box_row.get("[LiDARBoxComponent].box.heading"))
    if (
        center_x is None
        or center_y is None
        or center_z is None
        or size_x is None
        or size_y is None
        or size_z is None
        or heading is None
    ):
        return None
    if size_x <= 0 or size_y <= 0 or size_z <= 0:
        return None

    lx = 0.5 * size_x
    ly = 0.5 * size_y
    lz = 0.5 * size_z
    local = np.asarray(
        [
            [lx, ly, -lz],
            [-lx, ly, -lz],
            [-lx, -ly, -lz],
            [lx, -ly, -lz],
            [lx, ly, lz],
            [-lx, ly, lz],
            [-lx, -ly, lz],
            [lx, -ly, lz],
        ],
        dtype=np.float64,
    )
    c = np.cos(heading)
    s = np.sin(heading)
    rot = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    center = np.asarray([center_x, center_y, center_z], dtype=np.float64)
    return local @ rot.T + center


def _project_vehicle_points_to_image(
    points_vehicle: np.ndarray, params: Dict[str, Any]
) -> Optional[np.ndarray]:
    if points_vehicle.size == 0:
        return None
    camera_from_vehicle = params.get("camera_from_vehicle")
    if camera_from_vehicle is None:
        return None
    n_points = points_vehicle.shape[0]
    points_h = np.concatenate([points_vehicle, np.ones((n_points, 1), dtype=np.float64)], axis=1)
    points_cam = (camera_from_vehicle @ points_h.T).T[:, :3]

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    valid = x > 1e-6
    if not np.any(valid):
        return None

    u_n = -y[valid] / x[valid]
    v_n = -z[valid] / x[valid]
    r2 = u_n * u_n + v_n * v_n
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + params["k1"] * r2 + params["k2"] * r4 + params["k3"] * r6

    u_nd = u_n * radial + 2.0 * params["p1"] * u_n * v_n + params["p2"] * (r2 + 2.0 * u_n * u_n)
    v_nd = v_n * radial + params["p1"] * (r2 + 2.0 * v_n * v_n) + 2.0 * params["p2"] * u_n * v_n
    u = params["fu"] * u_nd + params["cu"]
    v = params["fv"] * v_nd + params["cv"]

    uv = np.stack([u, v], axis=1)
    uv = uv[np.all(np.isfinite(uv), axis=1)]
    if uv.size == 0:
        return None
    return uv


def _project_lidar_box_to_camera_row(
    lidar_box_row: Dict[str, Any], params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    corners = _lidar_box_corners_vehicle_frame(lidar_box_row)
    if corners is None:
        return None
    uv = _project_vehicle_points_to_image(corners, params)
    if uv is None:
        return None

    x1 = float(np.min(uv[:, 0]))
    y1 = float(np.min(uv[:, 1]))
    x2 = float(np.max(uv[:, 0]))
    y2 = float(np.max(uv[:, 1]))

    width = params.get("width")
    height = params.get("height")
    if width is not None and height is not None:
        x1 = max(0.0, min(x1, width - 1.0))
        x2 = max(0.0, min(x2, width - 1.0))
        y1 = max(0.0, min(y1, height - 1.0))
        y2 = max(0.0, min(y2, height - 1.0))
    if x2 <= x1 or y2 <= y1:
        return None

    laser_object_id = lidar_box_row.get("key.laser_object_id")
    if laser_object_id is None:
        return None
    lidar_type = lidar_box_row.get("[LiDARBoxComponent].type")
    size_x = x2 - x1
    size_y = y2 - y1
    center_x = x1 + 0.5 * size_x
    center_y = y1 + 0.5 * size_y
    return {
        "key.camera_name": params["camera_name"],
        "key.laser_object_id": str(laser_object_id),
        "[ProjectedLiDARBoxComponent].box.center.x": center_x,
        "[ProjectedLiDARBoxComponent].box.center.y": center_y,
        "[ProjectedLiDARBoxComponent].box.size.x": size_x,
        "[ProjectedLiDARBoxComponent].box.size.y": size_y,
        "[ProjectedLiDARBoxComponent].type": lidar_type,
    }


def _build_geometry_projected_lidar_rows(
    lidar_rows: List[Dict[str, Any]], calibration_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    projection_params: List[Dict[str, Any]] = []
    for calibration_row in calibration_rows:
        if not isinstance(calibration_row, dict):
            continue
        params = _camera_calibration_to_projection_params(calibration_row)
        if params is not None:
            projection_params.append(params)

    projected_rows: List[Dict[str, Any]] = []
    for lidar_row in lidar_rows:
        if not isinstance(lidar_row, dict):
            continue
        for params in projection_params:
            projected = _project_lidar_box_to_camera_row(lidar_row, params)
            if projected is not None:
                projected_rows.append(projected)
    return projected_rows


def _box_row_to_xyxy(row: Dict[str, Any], prefix: str) -> Optional[List[float]]:
    center_x = _safe_float(row.get("%s.box.center.x" % prefix))
    center_y = _safe_float(row.get("%s.box.center.y" % prefix))
    size_x = _safe_float(row.get("%s.box.size.x" % prefix))
    size_y = _safe_float(row.get("%s.box.size.y" % prefix))
    if center_x is None or center_y is None or size_x is None or size_y is None:
        return None
    if size_x <= 0 or size_y <= 0:
        return None
    x1 = center_x - 0.5 * size_x
    y1 = center_y - 0.5 * size_y
    x2 = center_x + 0.5 * size_x
    y2 = center_y + 0.5 * size_y
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _derive_association_rows_from_projected_boxes(
    camera_objects_by_camera: Dict[str, List[Dict[str, Any]]],
    projected_rows: List[Dict[str, Any]],
    min_iou: float = 0.05,
) -> List[Dict[str, Any]]:
    projected_by_camera: Dict[str, List[Dict[str, Any]]] = {}
    for row in projected_rows:
        if not isinstance(row, dict):
            continue
        cam_name = row.get("key.camera_name")
        laser_object_id = row.get("key.laser_object_id")
        if cam_name is None or laser_object_id is None:
            continue
        projected_by_camera.setdefault(str(cam_name), []).append(row)

    association_rows: List[Dict[str, Any]] = []
    for camera_name, camera_objects in camera_objects_by_camera.items():
        projected_for_camera = projected_by_camera.get(str(camera_name), [])
        if not projected_for_camera:
            continue

        camera_entries = []
        for obj in camera_objects:
            if not isinstance(obj, dict):
                continue
            obj_id = _camera_object_id(obj)
            if obj_id is None:
                continue
            box_xyxy = _camera_object_xyxy(obj)
            if box_xyxy is None:
                continue
            camera_entries.append(
                {
                    "camera_object_id": obj_id,
                    "box_xyxy": box_xyxy,
                    "type": _camera_object_type(obj),
                }
            )

        projected_entries = []
        for row in projected_for_camera:
            box_xyxy = _box_row_to_xyxy(row, "[ProjectedLiDARBoxComponent]")
            if box_xyxy is None:
                continue
            projected_entries.append(
                {
                    "laser_object_id": str(row.get("key.laser_object_id")),
                    "box_xyxy": box_xyxy,
                    "type": row.get("[ProjectedLiDARBoxComponent].type"),
                }
            )

        used_camera = set()
        used_projected = set()
        while True:
            best = None
            for ci, camera_entry in enumerate(camera_entries):
                if ci in used_camera:
                    continue
                for pi, projected_entry in enumerate(projected_entries):
                    if pi in used_projected:
                        continue
                    camera_type = camera_entry.get("type")
                    projected_type = projected_entry.get("type")
                    if camera_type is not None and projected_type is not None and camera_type != projected_type:
                        continue
                    iou = _xyxy_iou(camera_entry["box_xyxy"], projected_entry["box_xyxy"])
                    if best is None or iou > best[0]:
                        best = (iou, ci, pi)
            if best is None:
                break
            best_iou, ci, pi = best
            if best_iou < min_iou:
                break

            used_camera.add(ci)
            used_projected.add(pi)
            camera_entry = camera_entries[ci]
            projected_entry = projected_entries[pi]
            association_rows.append(
                {
                    "key.camera_name": str(camera_name),
                    "key.camera_object_id": camera_entry["camera_object_id"],
                    "key.laser_object_id": projected_entry["laser_object_id"],
                    "_derived_method": "projected_box_iou",
                    "_derived_iou": float(best_iou),
                }
            )
    return association_rows


def _build_camera_lidar_object_map(
    association_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    assoc_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in association_rows:
        if not isinstance(row, dict):
            continue
        camera_name = row.get("key.camera_name")
        camera_object_id = row.get("key.camera_object_id")
        if camera_name is None or camera_object_id is None:
            continue
        cam_key = str(camera_name)
        obj_key = str(camera_object_id)
        assoc_map.setdefault(cam_key, {}).setdefault(obj_key, []).append(row)
    return assoc_map


def _index_lidar_rows_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        laser_object_id = row.get("key.laser_object_id")
        if laser_object_id is None:
            continue
        indexed[str(laser_object_id)] = row
    return indexed


def _index_projected_rows_by_camera_and_lidar(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    indexed: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        camera_name = row.get("key.camera_name")
        laser_object_id = row.get("key.laser_object_id")
        if camera_name is None or laser_object_id is None:
            continue
        indexed.setdefault(str(camera_name), {}).setdefault(str(laser_object_id), []).append(row)
    return indexed


def _build_compact_object(
    camera_object: Dict[str, Any],
    association_mode: str,
    laser_object_ids: List[str],
    best_iou: Optional[float],
    lidar_boxes: List[Dict[str, Any]],
    projected_lidar_boxes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    object_id = _camera_object_id(camera_object)
    object_type = _camera_object_type(camera_object)
    box_xyxy = _camera_object_xyxy(camera_object)
    box_2d: Dict[str, Any] = {}
    if box_xyxy is not None:
        box_2d = _xyxy_to_center_size(box_xyxy)

    detection_difficulty = _safe_float(camera_object.get("[CameraBoxComponent].difficulty_level.detection"))
    tracking_difficulty = _safe_float(camera_object.get("[CameraBoxComponent].difficulty_level.tracking"))
    difficulty = {}
    if detection_difficulty is not None:
        difficulty["detection"] = detection_difficulty
    if tracking_difficulty is not None:
        difficulty["tracking"] = tracking_difficulty

    payload: Dict[str, Any] = {
        "box_2d": box_2d,
        "lidar_association": {
            "status": "matched" if laser_object_ids else "no_match",
            "association_mode": association_mode,
            "laser_object_ids": laser_object_ids,
            "derived_match_iou": best_iou,
            "lidar_boxes": lidar_boxes,
            "projected_lidar_boxes": projected_lidar_boxes,
        },
    }
    if object_id is not None:
        payload["camera_object_id"] = object_id
    if object_type is not None:
        payload["type"] = object_type
    if difficulty:
        payload["difficulty"] = difficulty
    return payload


def enrich_camera_objects_with_lidar(
    camera_objects_by_camera: Dict[str, List[Dict[str, Any]]],
    frame_components: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Enrich camera objects with projection-derived LiDAR matches."""
    lidar_rows = frame_components.get("lidar_box", [])
    calibration_rows = frame_components.get("camera_calibration", [])
    projected_rows = _build_geometry_projected_lidar_rows(
        lidar_rows=lidar_rows,
        calibration_rows=calibration_rows,
    )
    association_rows = _derive_association_rows_from_projected_boxes(
        camera_objects_by_camera, projected_rows
    )
    association_mode = "geometry_projection_iou"

    assoc_by_camera_obj = _build_camera_lidar_object_map(association_rows)
    lidar_by_id = _index_lidar_rows_by_id(lidar_rows)
    projected_by_camera_lidar = _index_projected_rows_by_camera_and_lidar(projected_rows)
    enriched: Dict[str, List[Dict[str, Any]]] = {}
    for camera_name, objects in camera_objects_by_camera.items():
        enriched_objects = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            camera_object_id = _camera_object_id(obj)
            laser_ids: List[str] = []
            assoc_rows_for_obj: List[Dict[str, Any]] = []

            if camera_object_id is not None:
                assoc_rows_for_obj = (
                    assoc_by_camera_obj.get(str(camera_name), {}).get(str(camera_object_id), [])
                )
                for assoc_row in assoc_rows_for_obj:
                    laser_object_id = assoc_row.get("key.laser_object_id")
                    if laser_object_id is not None:
                        laser_ids.append(str(laser_object_id))
            laser_ids = sorted(set(laser_ids))
            lidar_matches = [lidar_by_id[lid] for lid in laser_ids if lid in lidar_by_id]
            projected_matches: List[Dict[str, Any]] = []
            for laser_id in laser_ids:
                projected_matches.extend(
                    projected_by_camera_lidar.get(str(camera_name), {}).get(laser_id, [])
                )

            best_iou = None
            for assoc_row in assoc_rows_for_obj:
                iou_value = _safe_float(assoc_row.get("_derived_iou"))
                if iou_value is None:
                    continue
                if best_iou is None or iou_value > best_iou:
                    best_iou = iou_value

            enriched_objects.append(
                _build_compact_object(
                    camera_object=obj,
                    association_mode=association_mode,
                    laser_object_ids=laser_ids,
                    best_iou=best_iou,
                    lidar_boxes=lidar_matches,
                    projected_lidar_boxes=projected_matches,
                )
            )

        enriched[str(camera_name)] = enriched_objects
    return enriched


class LidarCameraAssociator:
    """Scene-level associator for post-processing metadata JSON files."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @staticmethod
    def load_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(path: Path, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _scene_components(scene_metadata: Dict[str, Any]) -> Dict[str, Any]:
        components = scene_metadata.get("components")
        if isinstance(components, dict):
            return components
        all_metadata = scene_metadata.get("all_metadata")
        if isinstance(all_metadata, dict):
            return all_metadata
        return {}

    def associate_scene(
        self,
        scene_metadata_path: Path,
        image_metadata_paths: List[Path],
        write: bool = True,
    ) -> Dict[str, Any]:
        scene_metadata = self.load_json(scene_metadata_path)
        scene_components = self._scene_components(scene_metadata)
        lidar_rows = scene_components.get("lidar_box", [])
        if not isinstance(lidar_rows, list):
            lidar_rows = []

        updated = 0
        matched_objects = 0
        for image_path in image_metadata_paths:
            image_metadata = self.load_json(image_path)
            camera_name = str(image_metadata.get("camera_name", "unknown"))
            camera_objects = image_metadata.get("Objects", [])
            if not isinstance(camera_objects, list):
                camera_objects = []

            frame_components: Dict[str, List[Dict[str, Any]]] = dict(scene_components)

            camera_calibration = image_metadata.get("camera_calibration", {})
            calibration_rows: List[Dict[str, Any]] = []
            if isinstance(camera_calibration, dict) and camera_calibration:
                calibration_rows.append(
                    _camera_calibration_payload_to_row(camera_name, camera_calibration)
                )

            camera_components = image_metadata.get("camera_components", {})
            if isinstance(camera_components, dict):
                legacy_rows = camera_components.get("camera_calibration", [])
                if isinstance(legacy_rows, list):
                    calibration_rows.extend([r for r in legacy_rows if isinstance(r, dict)])

            if calibration_rows:
                frame_components["camera_calibration"] = calibration_rows

            enriched = enrich_camera_objects_with_lidar(
                {camera_name: camera_objects},
                frame_components=frame_components,
            )
            updated_objects = enriched.get(camera_name, camera_objects)
            image_metadata["Objects"] = updated_objects

            for obj in updated_objects:
                if not isinstance(obj, dict):
                    continue
                status = obj.get("lidar_association", {}).get("status")
                if status == "matched":
                    matched_objects += 1

            if write:
                self.save_json(image_path, image_metadata)
            updated += 1
            if self.verbose:
                print("[debug] Updated lidar association for %s" % image_path.name)

        return {
            "scene_metadata_path": str(scene_metadata_path),
            "image_files_processed": updated,
            "matched_objects": matched_objects,
            "lidar_rows_in_scene": len(lidar_rows),
            "written": write,
        }
