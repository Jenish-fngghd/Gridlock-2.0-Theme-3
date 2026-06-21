"""Module 4a(seatbelt) — Seatbelt, two-stage LEARNED pipeline (§3.5, Phase 5 fine-tune).

Stage 1: YOLOv11n windshield detector (`train_windshield_detector.py`, mAP@0.5=0.995).
Stage 2: MobileNetV3-large belt classifier on the windshield crop (`train_seatbelt.py` v4,
         cosine-LR, no_seatbelt F1=0.8048 GT-crop / 0.8082 end-to-end with real Stage-1 boxes).

Replaces the old zero-shot `not_testable` stub now that both checkpoints exist. Graceful:
missing checkpoint/deps -> model_unavailable, never raises. 'mobile' class (out of the 7
mandated violations) is intentionally not surfaced here.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "checkpoints"
_DET_CKPT = _ROOT / "windshield" / "v1" / "weights" / "best.pt"
_CLF_CKPT = _ROOT / "seatbelt" / "v4" / "model.pt"


class SeatbeltModule:
    def __init__(self, det_checkpoint: str | Path | None = None,
                 clf_checkpoint: str | Path | None = None,
                 det_conf: float = 0.3, device: str | None = None):
        self.det_path = Path(det_checkpoint) if det_checkpoint else _DET_CKPT
        self.clf_path = Path(clf_checkpoint) if clf_checkpoint else _CLF_CKPT
        self.det_conf = det_conf
        self.detector = None
        self.clf = None
        self.tfm = None
        self.classes: dict[int, str] = {0: "seatbelt", 1: "no_seatbelt"}
        self.device = device
        self._unavailable_reason = ""
        self._load()

    def _load(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            from ultralytics import YOLO

            if not self.det_path.exists():
                self._unavailable_reason = f"windshield checkpoint not found: {self.det_path}"
                return
            if not self.clf_path.exists():
                self._unavailable_reason = f"belt classifier checkpoint not found: {self.clf_path}"
                return

            self.detector = YOLO(str(self.det_path))

            blob = torch.load(self.clf_path, map_location="cpu", weights_only=False)
            backbone = blob.get("backbone", "mobilenet_v3_large")
            clf = (models.mobilenet_v3_large() if "large" in backbone
                   else models.mobilenet_v3_small())
            clf.classifier[-1] = nn.Linear(clf.classifier[-1].in_features, 2)
            clf.load_state_dict(blob["state_dict"])
            clf.eval()
            self.classes = blob.get("classes", self.classes)
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.clf = clf.to(dev)
            self.device = dev
            size = blob.get("input", 224)
            self.tfm = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        except Exception as e:  # noqa: BLE001
            self._unavailable_reason = f"{type(e).__name__}: {str(e)[:140]}"
            self.detector = None
            self.clf = None

    def analyze(self, image) -> dict:
        """image: numpy BGR (cv2) | path. Runs windshield detect -> belt classify per crop.

        Returns {model_unavailable, windshields: [{bbox, label, confidence, no_seatbelt}]}.
        """
        if self.detector is None or self.clf is None:
            return {"model_unavailable": True, "windshields": [],
                    "note": f"seatbelt pipeline unavailable: {self._unavailable_reason}"}
        try:
            import cv2
            import torch
            from PIL import Image

            img = cv2.imread(image) if isinstance(image, (str, Path)) else image
            if img is None:
                return {"model_unavailable": True, "windshields": [], "note": "unreadable image"}
            res = self.detector.predict(img, conf=self.det_conf, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
            out = []
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                crop = img[max(0, y1):y2, max(0, x1):x2]
                pil = Image.fromarray(crop[:, :, ::-1])
                x = self.tfm(pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    probs = torch.softmax(self.clf(x), dim=1)[0]
                idx = int(probs.argmax().item())
                label = self.classes.get(idx, str(idx))
                out.append({"bbox": [x1, y1, x2, y2], "label": label,
                            "confidence": round(float(probs[idx]), 4),
                            "no_seatbelt": label == "no_seatbelt"})
            return {"model_unavailable": False, "windshields": out,
                    "note": "yolo11n-windshield + mobilenetv3l-belt (fine-tuned, e2e F1=0.8082)"}
        except Exception as e:  # noqa: BLE001
            return {"model_unavailable": True, "windshields": [],
                    "note": f"inference error: {type(e).__name__}: {e}"}
