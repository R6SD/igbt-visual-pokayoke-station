"""Torchvision detector inference adapter for separate glove/marker classes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

CLASS_NAMES = ["background", "hand_glove", "marker"]


class HandGloveDetector:
    def __init__(self, checkpoint: str | Path, score_threshold: float = 0.70, device: str = "cpu"):
        self.device = torch.device(device)
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if bundle.get("class_names") != CLASS_NAMES:
            raise ValueError(f"Unexpected detector classes {bundle.get('class_names')}; expected {CLASS_NAMES}.")
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=len(CLASS_NAMES))
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device).eval()
        self.score_threshold = score_threshold

    @torch.inference_mode()
    def predict(self, frame_bgr: Any) -> list[dict[str, Any]]:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(self.device)
        pred = self.model([image])[0]
        height, width = rgb.shape[:2]
        result = []
        for box, label, score in zip(pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu(), strict=True):
            score_value = float(score)
            class_id = int(label)
            if score_value < self.score_threshold or class_id not in (1, 2):
                continue
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            result.append({
                "class_id": class_id, "class_name": CLASS_NAMES[class_id], "score": score_value,
                "box_pixels_xyxy": [x1, y1, x2, y2],
                "box": [max(0.0, min(1.0, x1 / width)), max(0.0, min(1.0, y1 / height)),
                        max(0.0, min(1.0, x2 / width)), max(0.0, min(1.0, y2 / height))],
            })
        return result
