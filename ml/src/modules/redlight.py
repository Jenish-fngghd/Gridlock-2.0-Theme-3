"""Module 4b(red-light, event) — LSTM trajectory classifier (§3.8, Tier 2 promoted).

Classifies a tracked vehicle's full trajectory (sequence of (cx,cy,w,h) boxes across frames) as
ran-red-light vs not, replacing the geometry rule's motion requirement with a learned sequence
model (cross_f1=0.9000 held-out, split-by-video). See `train_redlight.py`.

Needs a multi-frame trajectory (video/clip input) — single stills cannot use this module; the
pipeline should route still images straight to `geo_note: abstained` instead. Graceful:
missing checkpoint/deps -> model_unavailable, never raises.
"""
from __future__ import annotations

from pathlib import Path

_CKPT = Path(__file__).resolve().parents[2] / "checkpoints" / "redlight" / "v1" / "model.pt"


class RedLightModule:
    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None):
        self.ckpt_path = Path(checkpoint) if checkpoint else _CKPT
        self.model = None
        self.seq_t = 32
        self.classes: dict[int, str] = {0: "no_cross", 1: "cross"}
        self.device = device
        self._unavailable_reason = ""
        self._load()

    def _load(self) -> None:
        try:
            import torch
            import torch.nn as nn

            if not self.ckpt_path.exists():
                self._unavailable_reason = f"checkpoint not found: {self.ckpt_path}"
                return
            blob = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            in_dim, hidden = blob.get("in_dim", 6), blob.get("hidden", 48)

            class TrajLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
                    self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 2))

                def forward(self, x):
                    out, (hn, _cn) = self.lstm(x)
                    return self.head(hn[-1])

            model = TrajLSTM()
            model.load_state_dict(blob["state_dict"])
            model.eval()
            self.classes = blob.get("classes", self.classes)
            self.seq_t = blob.get("seq_t", 32)
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(dev)
            self.device = dev
        except Exception as e:  # noqa: BLE001
            self._unavailable_reason = f"{type(e).__name__}: {str(e)[:140]}"
            self.model = None

    def _resample(self, frames: list[list[float]]) -> list[list[float]]:
        """frames: [[cx,cy,w,h], ...] (any length >=2) -> fixed [seq_t,6] with velocity feats."""
        t = self.seq_t
        L = len(frames)
        seq, prev = [], None
        for i in range(t):
            idx = round(i * (L - 1) / (t - 1)) if t > 1 else 0
            cx, cy, w, h = frames[idx][:4]
            dcx, dcy = (0.0, 0.0) if prev is None else (cx - prev[0], cy - prev[1])
            seq.append([cx, cy, w, h, dcx, dcy])
            prev = (cx, cy)
        return seq

    def classify(self, trajectory: list[list[float]]) -> dict:
        """trajectory: list of [cx,cy,w,h] per frame for ONE tracked vehicle (>=3 frames)."""
        if self.model is None:
            return {"model_unavailable": True, "label": "unknown", "confidence": 0.0,
                    "note": f"redlight classifier unavailable: {self._unavailable_reason}"}
        if len(trajectory) < 3:
            return {"model_unavailable": False, "label": "unknown", "confidence": 0.0,
                    "note": "trajectory too short (<3 frames); needs video/clip input"}
        try:
            import torch
            seq = self._resample(trajectory)
            x = torch.tensor([seq], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1)[0]
            idx = int(probs.argmax().item())
            return {"model_unavailable": False, "label": self.classes.get(idx, str(idx)),
                    "confidence": round(float(probs[idx]), 4),
                    "ran_red_light": self.classes.get(idx) == "cross"}
        except Exception as e:  # noqa: BLE001
            return {"model_unavailable": True, "label": "unknown", "confidence": 0.0,
                    "note": f"inference error: {type(e).__name__}: {e}"}
