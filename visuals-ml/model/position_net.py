"""
ResNet18 image encoder fused with coordinate vector → 3D position prediction.

Image  (3, 640, 960)  → ResNet18 backbone → AdaptiveAvgPool → 512-d
Coords (8,)           ─────────────────────────────────────────────┐
                                                              concat → 520-d
                                                        MLP: 520 → 256 → 128 → 3
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PositionNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Strip the classification head; keep everything up to the global pool
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # → (B, 512, 1, 1)

        self.head = nn.Sequential(
            nn.Linear(512 + 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        feat = self.pool(self.encoder(image)).flatten(1)   # (B, 512)
        x = torch.cat([feat, coords], dim=1)               # (B, 520)
        return self.head(x)                                # (B, 3)
