"""
PyTorch Dataset that streams from a records.jsonl index built by build_index.py.

Each sample is:
    image:  float32 tensor of shape (3, H, W) — full 1920x1280 image, normalised
    coords: float32 tensor of shape (8,)       — [cx_n, cy_n, sw_n, sh_n, fu, fv, cu, cv]
    target: float32 tensor of shape (3,)       — [tx, ty, tz] in ego-vehicle frame (metres)
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

INPUT_KEYS = ["cx_n", "cy_n", "sw_n", "sh_n", "fu", "fv", "cu", "cv"]
TARGET_KEYS = ["tx", "ty", "tz"]

_to_tensor = transforms.Compose([
    transforms.Resize((640, 960)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PositionDataset(Dataset):
    def __init__(self, index_file: str):
        index_path = Path(index_file)
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                "Run: python data/build_index.py"
            )

        self._records = []
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        r = self._records[idx]

        img = Image.open(r["image_path"]).convert("RGB")
        image = _to_tensor(img)

        coords = torch.tensor([r[k] for k in INPUT_KEYS], dtype=torch.float32)
        target = torch.tensor([r[k] for k in TARGET_KEYS], dtype=torch.float32)

        return image, coords, target
