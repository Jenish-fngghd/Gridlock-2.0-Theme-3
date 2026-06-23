"""Module 6 — VLM verify (00_master_design.md §6, confidence-cascade escalation only).

Calls a hosted vision-language model for violations the confidence cascade flagged `needs_vlm`
— now BOTH the `human_review` band (tiebreaker on uncertain cases) AND the `auto_confirm` band
(an agreement-gate cross-check before any auto-challan; see pipeline._run_vlm_verification).
It is still never called on every image — only on the rare frames that produced an actual
violation candidate, so the cheap local models (RF-DETR/MobileNet/TrOCR/SAM-3-crop/HSV) do the
bulk of the work and the VLM is a second opinion that must AGREE before a violation auto-confirms.

Provider: NVIDIA NIM (build.nvidia.com) — OpenAI-compatible chat-completions API with an
image-capable model (meta/llama-3.2-11b-vision-instruct by default). Needs NVIDIA_API_KEY in
env (never hard-coded). Graceful: missing key/network failure -> model_unavailable, never raises.
"""
from __future__ import annotations

import base64
import json
import os

_NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"


def _encode_image(image) -> str:
    """image: numpy BGR | PIL | path -> base64-encoded JPEG string."""
    import io
    from pathlib import Path

    from PIL import Image

    if isinstance(image, (str, Path)):
        pil = Image.open(str(image)).convert("RGB")
    elif hasattr(image, "mode"):
        pil = image.convert("RGB")
    else:
        import numpy as np
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]  # BGR -> RGB
        pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_VIOLATION_QUESTIONS = {
    "helmet": "Is the motorcycle rider in this image NOT wearing a helmet?",
    "no_helmet": "Is the motorcycle rider in this image NOT wearing a helmet?",
    "triple_riding": "Are there 3 or more people riding on this one motorcycle/bike?",
    "seatbelt": "Is the driver/occupant in this image NOT wearing a seatbelt?",
    "no_seatbelt": "Is the driver/occupant in this image NOT wearing a seatbelt?",
    "wrong_side": "Is this vehicle driving on the wrong side of the road (facing oncoming traffic)?",
    "red_light": "Is this vehicle crossing a red traffic light?",
    "illegal_parking": "Is this vehicle illegally parked (e.g. in a no-parking zone)?",
    "stop_line": "Is this vehicle stopped beyond the marked stop line?",
}


class VLMVerifier:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 45.0):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.model = model or os.environ.get("NVIDIA_VLM_MODEL", _DEFAULT_MODEL)
        self.timeout = timeout
        self._unavailable_reason = "" if self.api_key else "NVIDIA_API_KEY not set"

    def available(self) -> bool:
        return bool(self.api_key)

    def verify(self, crop, violation_type: str, extra_context: str = "") -> dict:
        """Ask the VLM to confirm/deny a flagged violation on a crop. Returns:
        {model_unavailable, confirmed: bool|None, vlm_confidence: float, caption: str}
        """
        if not self.api_key:
            return {"model_unavailable": True, "confirmed": None, "vlm_confidence": 0.0,
                    "caption": "", "note": self._unavailable_reason}
        try:
            import requests

            b64 = _encode_image(crop)
            question = _VIOLATION_QUESTIONS.get(
                violation_type, f"Does this image show a '{violation_type}' traffic violation?")
            prompt = (
                f"{question} {extra_context}\n"
                "Answer in strict JSON only, no markdown: "
                '{"confirmed": true/false, "confidence": 0.0-1.0, "caption": "one short sentence"}'
            )
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                "max_tokens": 150,
                "temperature": 0.1,
            }
            resp = requests.post(
                _NIM_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload, timeout=self.timeout,
            )
            if resp.status_code != 200:
                return {"model_unavailable": True, "confirmed": None, "vlm_confidence": 0.0,
                        "caption": "", "note": f"NIM HTTP {resp.status_code}: {resp.text[:160]}"}
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = self._parse_json_response(content)
            return {"model_unavailable": False,
                    "confirmed": parsed.get("confirmed"),
                    "vlm_confidence": round(float(parsed.get("confidence", 0.5)), 3),
                    "caption": parsed.get("caption", content.strip()[:200]),
                    "model": self.model}
        except Exception as e:  # noqa: BLE001
            return {"model_unavailable": True, "confirmed": None, "vlm_confidence": 0.0,
                    "caption": "", "note": f"{type(e).__name__}: {str(e)[:160]}"}

    @staticmethod
    def _parse_json_response(content: str) -> dict:
        text = content.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
        low = text.lower()
        return {"confirmed": "true" in low and "false" not in low.split("true")[0][-20:],
                "confidence": 0.5, "caption": text[:200]}
