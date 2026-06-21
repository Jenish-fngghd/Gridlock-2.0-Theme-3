"""Module 4b(wrong-side) — Wrong-side driving, LEARNED classifier (Tier 2, promoted).

00_master_design.md §3.6 names a geometry rule (trajectory vs lane direction), but that needs
motion history a single still doesn't have. Phase 5 promoted this to a direct MobileNetV3-small
classifier trained on the Wrong-Way OBB dataset (appearance/heading cues baked into the crop,
no h-flip augmentation since heading IS the label signal) — F1=0.9551 held-out, replacing the
geometry abstain. See `train_wrongside.py` / `eval_wrongside.py --weights`.

Runs on a single vehicle crop (from the shared detection pass) — works on stills, no tracking
needed. Graceful: missing checkpoint/deps -> model_unavailable, never raises.
"""
from __future__ import annotations

from pathlib import Path

_CKPT = Path(__file__).resolve().parents[2] / "checkpoints" / "wrongside" / "v1" / "model.pt"


class WrongSideModule:
    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None):
        self.ckpt_path = Path(checkpoint) if checkpoint else _CKPT
        self.model = None
        self.tfm = None
        self.classes: dict[int, str] = {0: "right-side", 1: "wrong-side"}
        self.device = device
        self._unavailable_reason = ""
        self._load()

    def _load(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            if not self.ckpt_path.exists():
                self._unavailable_reason = f"checkpoint not found: {self.ckpt_path}"
                return
            blob = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            backbone = blob.get("backbone", "mobilenet_v3_small")
            model = (models.mobilenet_v3_large() if "large" in backbone
                     else models.mobilenet_v3_small())
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
            model.load_state_dict(blob["state_dict"])
            model.eval()
            self.classes = blob.get("classes", self.classes)
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(dev)
            self.device = dev
            size = blob.get("input", 128)
            self.tfm = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        except Exception as e:  # noqa: BLE001
            self._unavailable_reason = f"{type(e).__name__}: {str(e)[:140]}"
            self.model = None

    def classify(self, vehicle_crop) -> dict:
        """vehicle_crop: numpy BGR | PIL | path. Returns {label, confidence, model_unavailable}."""
        if self.model is None:
            return {"model_unavailable": True, "label": "unknown", "confidence": 0.0,
                    "note": f"wrong-side classifier unavailable: {self._unavailable_reason}"}
        try:
            import torch
            from PIL import Image
            pil = self._to_pil(vehicle_crop)
            x = self.tfm(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1)[0]
            idx = int(probs.argmax().item())
            return {"model_unavailable": False, "label": self.classes.get(idx, str(idx)),
                    "confidence": round(float(probs[idx]), 4),
                    "wrong_side": self.classes.get(idx) == "wrong-side"}
        except Exception as e:  # noqa: BLE001
            return {"model_unavailable": True, "label": "unknown", "confidence": 0.0,
                    "note": f"inference error: {type(e).__name__}: {e}"}

    @staticmethod
    def _to_pil(crop):
        from PIL import Image
        if isinstance(crop, (str, Path)):
            return Image.open(str(crop)).convert("RGB")
        if hasattr(crop, "mode"):
            return crop.convert("RGB")
        import numpy as np
        arr = np.asarray(crop)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]  # assume BGR -> RGB
        return Image.fromarray(arr)
