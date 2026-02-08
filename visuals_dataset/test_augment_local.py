"""
Local test for augmentation pipeline using a real image from output/clear.
Does not require Waymo credentials.
"""

from PIL import Image
from pathlib import Path
from augment import ImageAugmenter, encode_image

# Load the existing clear image
clear_img_path = Path("output/clear/10094743350625019937_3420_000_3440_000_1553629313073985_001.jpeg")
if not clear_img_path.exists():
    print(f"[error] Clear image not found at {clear_img_path}")
    print("[info] Available images:")
    for img in Path("output/clear").glob("*"):
        print(f"  - {img.name}")
    exit(1)

test_img = Image.open(clear_img_path).convert("RGB")
print(f"[info] Loaded test image: {clear_img_path} ({test_img.size})")

# Create augmenter
try:
    augmenter = ImageAugmenter(seed=42)
    print("[info] ImageAugmenter loaded successfully")
except Exception as e:
    print(f"[error] Failed to load augmenter: {e}")
    exit(1)

# Apply all augmentations
try:
    aug_map = augmenter.apply_all(test_img)
    print(f"[success] Applied {len(aug_map)} augmentations: {list(aug_map.keys())}")
except Exception as e:
    print(f"[error] Failed to apply augmentations: {e}")
    exit(1)

# Save outputs to existing output structure
output_dir = Path("output")

for aug_name, aug_img in aug_map.items():
    out_dir = output_dir / aug_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "10094743350625019937_3420_000_3440_000_1553629313073985_001.jpeg"
    enc = encode_image(aug_img)
    out_path.write_bytes(enc)
    print(f"[saved] {out_path}")

print(f"\n[success] All augmentations saved to {output_dir}/")
