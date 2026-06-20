"""Module 4c — ROI-gated ANPR.

00_master_design.md §3.7 / 06_... rows 16–19: plate detect (RF-DETR) → rectify/SR → PP-OCRv5
→ Indian-format syntax validator, run ONLY on flagged violators.

Reality on this machine (Phase 0):
- **Plate detection is not available zero-shot** — COCO has no license-plate class, so RF-DETR
  can't localize plates without fine-tuning (CCPD/Indian-LP). Flagged `needs_finetune`. For OCR
  evaluation we can still use the GT plate crop (CCPD encodes the plate box in the filename).
- **OCR engine** is auto-selected from whatever is installed: PaddleOCR → EasyOCR → TrOCR
  (transformers). None are installed by default here, so OCR degrades to `model_unavailable`.
  CCPD is Chinese plates (needs a CN model); Indian-LP is Latin (TrOCR/EasyOCR viable).
- **Indian-format validator** always runs on any recognized text (regex, no deps).
"""
from __future__ import annotations

import re

# Indian plate formats: old `SS NN L(L) NNNN`, HSRP, and BH-series `NN BH NNNN LL`.
INDIAN_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),   # MH12AB1234 / DL8CAF5031
    re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),            # 22BH1234AA (BH-series)
]
STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB", "AN",
}


def validate_indian(text: str) -> dict:
    t = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    fmt_ok = any(p.match(t) for p in INDIAN_PATTERNS)
    state_ok = t[:2] in STATE_CODES if len(t) >= 2 else False
    return {"normalized": t, "format_valid": fmt_ok, "state_code_valid": state_ok}


# Character-confusion maps for syntax-aware correction (only applied where the plate
# structure demands a specific class — letters in the state/series zones, digits in the
# RTO/number zones). OCR commonly swaps these visually-similar glyphs.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5",
             "B": "8", "G": "6", "A": "4", "T": "7"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "7": "T"}


def correct_plate_text(text: str) -> str:
    """Coerce a noisy OCR string toward the Indian plate grammar
    [2 letters][1-2 digits][1-3 letters][4 digits], fixing class-ambiguous chars by position.

    SAFE: the correction is accepted only if it produces a *format-valid* plate; otherwise the
    raw normalized string is returned unchanged (never makes a valid plate invalid, never
    fabricates structure on a too-short/too-long string).
    """
    s = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    n = len(s)
    if 8 <= n <= 11:  # parseable length (else: only one row read, or noise — leave alone)
        # Try the canonical 2-digit-RTO parse FIRST (Indian RTO codes are overwhelmingly
        # 2-digit, zero-padded), then 1-digit. Return the first parse that yields a valid plate.
        # This also re-segments coincidentally-valid-but-mis-split reads like MH2OCS9817->MH20CS9817.
        for rto_len in (2, 1):
            rest = s[2 + rto_len:]
            if len(rest) < 5:
                continue
            series, num = rest[:-4], rest[-4:]
            if not (1 <= len(series) <= 3):
                continue
            head = "".join(_TO_LETTER.get(c, c) for c in s[:2])           # state -> letters
            rto = "".join(_TO_DIGIT.get(c, c) for c in s[2:2 + rto_len])  # RTO -> digits
            ser = "".join(_TO_LETTER.get(c, c) for c in series)          # series -> letters
            number = "".join(_TO_DIGIT.get(c, c) for c in num)           # number -> digits
            cand = head + rto + ser + number
            if validate_indian(cand)["format_valid"]:
                return cand
    return s  # unchanged if no valid structural parse exists


def merge_text_regions(regions: list) -> tuple[str, float]:
    """Merge multi-line OCR output into one reading-order string.

    regions: list of (bbox_4pts, text, conf). Indian plates are often 2-row; EasyOCR returns
    one region per line, so we group regions into rows (by vertical position) and concatenate
    top-to-bottom, left-to-right — recovering plates where only one row was read before.
    """
    items = []
    for bbox, text, conf in regions:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append({"text": text, "conf": float(conf), "y": min(ys), "x": min(xs),
                      "h": max(ys) - min(ys)})
    if not items:
        return "", 0.0
    items.sort(key=lambda r: (r["y"], r["x"]))
    heights = sorted(r["h"] for r in items)
    med_h = heights[len(heights) // 2] or 1.0
    rows, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it["y"] - cur[-1]["y"]) <= 0.6 * med_h:
            cur.append(it)
        else:
            rows.append(cur)
            cur = [it]
    rows.append(cur)
    out, confs = "", []
    for row in rows:
        row.sort(key=lambda r: r["x"])
        out += "".join(r["text"] for r in row)
        confs += [r["conf"] for r in row]
    return out, (sum(confs) / len(confs) if confs else 0.0)


class ANPRModule:
    def __init__(self, ocr_engine: str = "auto"):
        # Lazy: don't load/download an OCR engine until recognize() is actually called.
        # (Zero-shot plate detection is unavailable, so the demo never OCRs — no need to pay
        #  the model download at init.)
        self._pref = ocr_engine
        self._loaded = False
        self.engine_name = None
        self.engine = None

    def _ensure_engine(self) -> None:
        if not self._loaded:
            self._load_ocr(self._pref)
            self._loaded = True

    def _load_ocr(self, which: str) -> None:
        order = (["paddleocr", "easyocr", "trocr"] if which == "auto" else [which])
        for name in order:
            try:
                if name == "paddleocr":
                    # Disable oneDNN before paddle imports — the PIR+oneDNN CPU path raises
                    # NotImplementedError(ConvertPirAttribute2RuntimeAttribute) on paddle 3.x.
                    import os
                    os.environ.setdefault("FLAGS_use_mkldnn", "0")
                    from paddleocr import PaddleOCR
                    # PaddleOCR 3.x (PP-OCRv5/v6) renamed params; try 3.x init then 2.x.
                    try:
                        self.engine = PaddleOCR(use_textline_orientation=True, lang="en",
                                                enable_mkldnn=False)
                        self._paddle_api = "3x"
                    except TypeError:
                        try:
                            self.engine = PaddleOCR(use_textline_orientation=True, lang="en")
                            self._paddle_api = "3x"
                        except Exception:
                            self.engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                            self._paddle_api = "2x"
                elif name == "easyocr":
                    import easyocr
                    self.engine = easyocr.Reader(["en"], gpu=False, verbose=False)
                elif name == "trocr":
                    from transformers import (TrOCRProcessor,
                                              VisionEncoderDecoderModel)
                    proc = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
                    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
                    self.engine = ("trocr", proc, model)
                else:
                    continue
                self.engine_name = name
                return
            except Exception:
                continue
        self.engine_name = None  # nothing available

    def plate_detection_available(self) -> bool:
        return False  # zero-shot: no plate class. Needs fine-tune (CCPD/Indian-LP).

    def recognize(self, plate_crop) -> dict:
        """plate_crop: numpy/PIL/path of a cropped plate. Returns {text, confidence, engine}."""
        self._ensure_engine()
        if self.engine_name is None:
            return {"model_unavailable": True, "text": "", "confidence": 0.0,
                    "engine": None,
                    "note": "no OCR engine installed (paddleocr/easyocr/trocr). ANPR not_testable."}
        try:
            text, conf = self._run_engine(plate_crop)
            corrected = correct_plate_text(text)
            val = validate_indian(corrected)
            return {"model_unavailable": False, "text": corrected, "raw_text": text,
                    "confidence": round(conf, 4), "engine": self.engine_name,
                    "indian_validation": val}
        except Exception as e:  # noqa: BLE001
            return {"model_unavailable": True, "text": "", "confidence": 0.0,
                    "engine": self.engine_name, "note": f"ocr error: {type(e).__name__}: {e}"}

    def _run_engine(self, crop) -> tuple[str, float]:
        if self.engine_name == "easyocr":
            import numpy as np
            res = self.engine.readtext(np.asarray(self._to_rgb(crop)))
            if not res:
                return "", 0.0
            # merge ALL detected regions top-to-bottom (handles 2-row Indian plates)
            return merge_text_regions(res)
        if self.engine_name == "paddleocr":
            import numpy as np
            arr = np.asarray(self._to_rgb(crop))
            regions = self._paddle_regions(arr)
            if not regions:
                return "", 0.0
            # merge multi-line top-to-bottom, same as EasyOCR (handles 2-row Indian plates)
            return merge_text_regions(regions)
        if self.engine_name == "trocr":
            _, proc, model = self.engine
            pix = proc(images=self._to_rgb(crop), return_tensors="pt").pixel_values
            ids = model.generate(pix)
            txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
            return txt, 0.5  # TrOCR gives no calibrated score
        return "", 0.0

    def _paddle_regions(self, arr) -> list:
        """Normalize PaddleOCR output (2.x .ocr() OR 3.x .predict()) to [(box4pts, text, score)]."""
        regions = []
        # --- try 3.x predict() first ---
        if getattr(self, "_paddle_api", "2x") == "3x" and hasattr(self.engine, "predict"):
            try:
                res = self.engine.predict(arr)
                for item in (res or []):
                    d = item if isinstance(item, dict) else getattr(item, "json", None) or {}
                    # OCRResult exposes dict-like access in 3.x
                    texts = d.get("rec_texts") if isinstance(d, dict) else None
                    if texts is None:
                        try:
                            texts = item["rec_texts"]
                            scores = item["rec_scores"]
                            polys = item.get("rec_polys", item.get("dt_polys"))
                        except Exception:
                            texts = None
                    else:
                        scores = d.get("rec_scores", [1.0] * len(texts))
                        polys = d.get("rec_polys", d.get("dt_polys"))
                    if texts:
                        for i, t in enumerate(texts):
                            box = polys[i] if polys is not None and i < len(polys) else [[0, 0], [1, 0], [1, 1], [0, 1]]
                            box = [[float(p[0]), float(p[1])] for p in box]
                            regions.append((box, t, float(scores[i]) if i < len(scores) else 1.0))
                if regions:
                    return regions
            except Exception:
                pass
        # --- fall back to 2.x .ocr() ---
        try:
            res = self.engine.ocr(arr, cls=True)
            line = (res or [[]])[0] or []
            for box, (text, score) in line:
                regions.append(([[float(p[0]), float(p[1])] for p in box], text, float(score)))
        except Exception:
            pass
        return regions

    @staticmethod
    def _to_rgb(crop):
        from PIL import Image
        if isinstance(crop, str):
            return Image.open(crop).convert("RGB")
        if hasattr(crop, "mode"):
            return crop.convert("RGB")
        import numpy as np
        arr = np.asarray(crop)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]  # BGR->RGB
        return Image.fromarray(arr)
