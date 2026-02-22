"""
CLI utility for scene-based LiDAR-camera projection association.

Input contract:
- One scene metadata file under metadata/scene_metadata
- Matching image metadata files under metadata/image_metadata

Output:
- In-place updates to image metadata files with compact lidar association fields.
"""

import argparse
import json
from pathlib import Path
from typing import List

from lidar_camera_association import LidarCameraAssociator


def discover_scene_image_metadata(scene_metadata_path: Path, image_metadata_dir: Path) -> List[Path]:
    """
    Find image metadata files that belong to a given scene metadata file.
    """
    expected_scene_ref = str(Path("scene_metadata") / scene_metadata_path.name)
    matches: List[Path] = []
    for image_path in sorted(image_metadata_dir.glob("*.json")):
        try:
            payload = json.loads(image_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("scene_metadata_file") == expected_scene_ref:
            matches.append(image_path)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiDAR-camera association for one scene.")
    parser.add_argument("--scene-metadata", required=True, help="Path to scene metadata JSON.")
    parser.add_argument("--image-metadata-dir", required=True, help="Directory containing image metadata JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; only print summary.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    scene_path = Path(args.scene_metadata).resolve()
    image_dir = Path(args.image_metadata_dir).resolve()

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene metadata file not found: {scene_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image metadata directory not found: {image_dir}")

    image_paths = discover_scene_image_metadata(scene_path, image_dir)
    if not image_paths:
        print(f"[warn] No image metadata files found for scene: {scene_path.name}")
        return

    associator = LidarCameraAssociator(verbose=args.verbose)
    summary = associator.associate_scene(
        scene_metadata_path=scene_path,
        image_metadata_paths=image_paths,
        write=not args.dry_run,
    )

    print("[success] Lidar association completed.")
    print(f"  - Scene: {scene_path}")
    print(f"  - Image metadata files: {summary['image_files_processed']}")
    print(f"  - Scene lidar rows: {summary['lidar_rows_in_scene']}")
    print(f"  - Write mode: {'enabled' if not args.dry_run else 'dry-run'}")


if __name__ == "__main__":
    main()
