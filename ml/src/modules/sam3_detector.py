"""SAM-3 open-vocabulary detector — client for the isolated `.venv-sam3` worker subprocess.

Why a subprocess: the official `facebookresearch/sam3` package (the proven-working path —
see `how_to_segment_images_with_segment_anything_3.ipynb`) hard-pins `numpy<2`, which conflicts
with the main pipeline's numpy>=2 (required by rfdetr/transformers/ultralytics). The earlier
attempt to load SAM-3 via `transformers.Sam3Model` segfaulted (the Hub checkpoint's config
declares `Sam3VideoModel`, a mismatched architecture) — so we don't use that path at all.

This client lazily spawns `sam3_worker.py` under `.venv-sam3`, keeps it warm (loads the ~3.2GB
checkpoint ONCE), and talks newline-delimited JSON over its stdin/stdout. Used sparingly — only
for the crops that actually need open-vocab reasoning (helmet-state on rider crops, plate
localization on violator crops) — never the full image, to keep cost down (SAM-3 is ~14x slower
than RF-DETR per the Phase-5 benchmark).

Gated model: needs HF_TOKEN in env (read by the worker via huggingface_hub's standard auth).
Graceful: any failure -> model_unavailable, never raises and never crashes the main process.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.modules.detection import Detection, DetectionResult

def _default_venv_python() -> Path:
    root = Path(__file__).resolve().parents[3] / ".venv-sam3"
    win = root / "Scripts" / "python.exe"
    return win if win.exists() else root / "bin" / "python"


_VENV_PY = _default_venv_python()
_WORKER_MODULE = "src.modules.sam3_worker"
_BOOT_TIMEOUT = 180.0   # first call loads a ~3.2GB checkpoint
_CALL_TIMEOUT = 60.0


class SAM3Detector:
    """Drop-in replacement for the old transformers-based detector, same public API."""

    def __init__(self, threshold: float = 0.3, repo_root: str | Path | None = None):
        self.threshold = threshold
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self.device = "unknown"
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._unavailable_reason = ""
        self._booted = False

    # ---- lifecycle ----------------------------------------------------

    def _ensure_started(self) -> bool:
        if self._proc is not None and self._proc.poll() is None and self._booted:
            return True
        if not _VENV_PY.exists():
            self._unavailable_reason = (
                f"SAM-3 venv not found at {_VENV_PY}. Run setup: "
                f"python -m venv .venv-sam3 && .venv-sam3/Scripts/pip install -e vendor/sam3"
            )
            return False
        try:
            env = os.environ.copy()
            self._proc = subprocess.Popen(
                [str(_VENV_PY), "-u", "-m", _WORKER_MODULE],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(self.repo_root), env=env, text=True, bufsize=1,
            )
            boot_line = self._readline_with_timeout(_BOOT_TIMEOUT)
            if boot_line is None:
                stderr = self._proc.stderr.read(4000) if self._proc.stderr else ""
                self._unavailable_reason = f"worker boot timed out/no output. stderr: {stderr[:300]}"
                self._kill()
                return False
            boot = json.loads(boot_line)
            if not boot.get("ok"):
                self._unavailable_reason = f"worker boot failed: {boot.get('error', '?')}"
                self._kill()
                return False
            self.device = boot.get("device", "unknown")
            self._booted = True
            return True
        except Exception as e:  # noqa: BLE001
            self._unavailable_reason = f"{type(e).__name__}: {str(e)[:200]}"
            self._kill()
            return False

    def _readline_with_timeout(self, timeout: float) -> str | None:
        result: list[str | None] = [None]

        def _read():
            try:
                result[0] = self._proc.stdout.readline()  # type: ignore[union-attr]
            except Exception:
                result[0] = None

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive() or not result[0]:
            return None
        return result[0].strip()

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None
        self._booted = False

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
            except Exception:
                pass
        self._kill()

    # ---- inference ------------------------------------------------------

    def detect_concept(self, image, concept: str, threshold: float | None = None) -> DetectionResult:
        """Single-concept convenience wrapper around detect_concepts()."""
        out = self.detect_concepts(image, [concept], threshold=threshold)
        return out.get(concept, DetectionResult(model_unavailable=True, note="no result"))

    def detect_concepts(self, image, concepts: list[str],
                        threshold: float | None = None) -> dict[str, DetectionResult]:
        """Run MULTIPLE concept prompts on ONE image in a single worker call (shared vision
        encoding — cheaper than calling detect_concept() per concept)."""
        thr = self.threshold if threshold is None else threshold
        with self._lock:
            if not self._ensure_started():
                note = f"SAM-3 unavailable: {self._unavailable_reason}"
                return {c: DetectionResult(model_unavailable=True, note=note) for c in concepts}
            try:
                b64 = self._encode_image(image)
                req = {"id": str(time.time()), "image_b64": b64, "prompts": concepts, "threshold": thr}
                self._proc.stdin.write(json.dumps(req) + "\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
                line = self._readline_with_timeout(_CALL_TIMEOUT)
                if line is None:
                    self._unavailable_reason = "worker call timed out (process may have died)"
                    self._kill()
                    note = f"SAM-3 unavailable: {self._unavailable_reason}"
                    return {c: DetectionResult(model_unavailable=True, note=note) for c in concepts}
                resp = json.loads(line)
                if not resp.get("ok"):
                    note = f"SAM-3 inference error: {resp.get('error', '?')}"
                    return {c: DetectionResult(model_unavailable=True, note=note) for c in concepts}
                out: dict[str, DetectionResult] = {}
                for concept, items in resp.get("results", {}).items():
                    dets = [Detection(xyxy=tuple(it["box"]), confidence=it["score"],
                                      class_id=-1, class_name=concept) for it in items]
                    out[concept] = DetectionResult(detections=dets, note=f"sam3:{concept} ({self.device})")
                return out
            except Exception as e:  # noqa: BLE001
                note = f"SAM-3 inference error: {type(e).__name__}: {e}"
                return {c: DetectionResult(model_unavailable=True, note=note) for c in concepts}

    @staticmethod
    def _encode_image(image) -> str:
        from PIL import Image
        if isinstance(image, (str, Path)):
            pil = Image.open(str(image)).convert("RGB")
        elif hasattr(image, "mode"):
            pil = image.convert("RGB")
        else:
            import numpy as np
            arr = np.asarray(image)
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1]  # assume BGR -> RGB
            pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("ascii")
