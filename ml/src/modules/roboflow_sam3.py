"""Roboflow-hosted SAM-3 client — open-vocabulary concept segmentation over the serverless API.

This replaces the local `.venv-sam3` subprocess worker (which needed a GPU and the official
package's CUDA-only build). Roboflow hosts SAM-3 and handles GPU provisioning, so from our side
it's just an HTTP call: image + text prompts -> per-concept boxes. One request can carry MANY
prompts (shared vision encoding), so a whole image's entity set is one call (~2-4s).

SAM-3 segments concrete *concepts* (it does NOT reason about negation/attributes — "person
without helmet" just segments persons). So we prompt nouns ("motorcycle", "helmet", "stop line")
and apply the violation rules ourselves (see sam3_violations.py, ported from the reference
notebook how_to_segment_images_with_segment_anything_3.ipynb).

Endpoint: serverless.roboflow.com/sam3/concept_segment. Needs ROBOFLOW_API_KEY. Graceful:
missing key / network error / busy-lock-exhausted -> model_unavailable, never raises.
"""
from __future__ import annotations

import base64
import os
import time


_URL = "https://serverless.roboflow.com/sam3/concept_segment"


class RoboflowSAM3:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0,
                 default_conf: float = 0.5, retries: int = 4):
        primary = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        fallback = os.environ.get("ROBOFLOW_API_KEY_FALLBACK", "")
        # Second key tried only when the primary fails (rate limit / quota exhausted / any
        # non-200) -- keeps the pipeline running on a free/limited key without manual swapping.
        self.api_keys = [k for k in dict.fromkeys([primary, fallback]) if k]
        self.timeout = timeout
        self.default_conf = default_conf
        self.retries = retries

    def available(self) -> bool:
        return bool(self.api_keys)

    def detect_many(self, image, prompts: list[str], conf: float | None = None) -> dict:
        """One call, many concepts. Returns {prompt: [{'box': [x1,y1,x2,y2], 'conf': float}, ...]}.
        On any failure (across all configured keys) returns {prompt: []} for every prompt plus a
        private '_unavailable' flag."""
        thr = self.default_conf if conf is None else conf
        empty = {p: [] for p in prompts}
        if not self.api_keys or not prompts:
            empty["_unavailable"] = True
            return empty
        try:
            import requests

            b64 = self._encode(image)
            body = {"image": {"type": "base64", "value": b64},
                    "prompts": [{"type": "text", "text": p} for p in prompts]}
            resp = None
            last_note = ""
            for key in self.api_keys:
                for _ in range(self.retries):
                    resp = requests.post(f"{_URL}?api_key={key}", json=body, timeout=self.timeout)
                    if resp.status_code == 200:
                        break
                    if "lock" in resp.text.lower() or "try again" in resp.text.lower():
                        time.sleep(5)  # model-manager warming up; brief backoff
                        continue
                    break  # non-recoverable on this key (e.g. rate limit) -- try next key
                if resp is not None and resp.status_code == 200:
                    break
                last_note = f"HTTP {resp.status_code}: {resp.text[:120]}" if resp is not None \
                    else "busy-lock retries exhausted"
            if resp is None or resp.status_code != 200:
                empty["_unavailable"] = True
                empty["_note"] = last_note or "all keys exhausted"
                return empty

            out: dict = {p: [] for p in prompts}
            for pr in resp.json().get("prompt_results", []):
                idx = pr.get("prompt_index", 0)
                prompt = prompts[idx] if 0 <= idx < len(prompts) else str(idx)
                dets = []
                for pred in pr.get("predictions", []):
                    c = pred.get("confidence", 0.0)
                    if c < thr:
                        continue
                    box = self._box_from_masks(pred.get("masks", []))
                    if box is not None:
                        dets.append({"box": box, "conf": round(float(c), 4)})
                out[prompt] = dets
            return out
        except Exception as e:  # noqa: BLE001
            empty["_unavailable"] = True
            empty["_note"] = f"{type(e).__name__}: {str(e)[:120]}"
            return empty

    def detect(self, image, prompt: str, conf: float | None = None) -> list:
        return self.detect_many(image, [prompt], conf=conf).get(prompt, [])

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _box_from_masks(masks: list) -> list | None:
        """SAM-3 returns polygon masks (list of [x,y] points). Derive an axis-aligned bbox."""
        xs, ys = [], []
        for poly in masks:
            for pt in poly:
                xs.append(pt[0]); ys.append(pt[1])
        if not xs:
            return None
        return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]

    @staticmethod
    def _encode(image) -> str:
        from pathlib import Path
        if isinstance(image, (str, Path)):
            return base64.b64encode(Path(image).read_bytes()).decode()
        import cv2
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode()
