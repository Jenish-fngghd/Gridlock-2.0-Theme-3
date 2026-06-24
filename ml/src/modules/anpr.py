"""Module 4c — ROI-gated ANPR.

00_master_design.md §3.3 / 06_... rows 16–19: plate detect (RF-DETR) → rectify/SR → PP-OCRv5
→ Indian-format syntax validator, run ONLY on flagged violators.

Engine ladder (auto-select or force via `ocr_engine` kwarg):
  paddleocr        — PP-OCRv5/v6 mobile (primary; fastest)
  paddleocr_server — PP-OCRv4 server weights (larger, more accurate)
  easyocr          — CRAFT + CRNN backup (GPU if available)
  trocr            — microsoft/trocr-small-printed
  trocr_base       — microsoft/trocr-base-printed (better quality)
  trocr_large      — microsoft/trocr-large-printed (best quality, needs GPU)
  trocr_ft         — locally fine-tuned TrOCR from checkpoints/anpr/trocr_ft/
  doctr            — mindee/doctr (strong open-source alternative)
  ensemble         — runs ALL available engines + votes for best result

Preprocessing pipeline (enabled by default when `preprocess=True`):
  1. Upscale 4× bicubic (small plates → bigger chars → better OCR)
  2. CLAHE on L-channel (uneven lighting compensation)
  3. Unsharp-mask sharpening (enhances character edges)
  4. Auto-invert dark background plates (black bg with white text → invert)

TTA (test-time augmentation, optional):
  Runs the engine on original + 3 augmented views, votes by Levenshtein similarity.
"""
from __future__ import annotations

import re
from pathlib import Path

# ─── Indian plate patterns ──────────────────────────────────────────────────
INDIAN_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),   # MH12AB1234 / DL8CAF5031
    re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),            # 22BH1234AA (BH-series)
]
STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB", "AN",
}

# Expanded character-confusion maps (position-aware correction)
_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
    "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4", "T": "7",
    "E": "3", "F": "7", "J": "1", "U": "0", "V": "7",
}
_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B",
    "6": "G", "4": "A", "7": "T", "3": "E",
}

# Common OCR junk prefixes/suffixes on Indian plates
_STRIP_PATTERNS = [
    re.compile(r"^IND[A-Z]?", re.I),                 # "IND" header on HSRP
    re.compile(r"VALID(ITY)?[\d/\-]*$", re.I),        # "VALID TILL 12/2026"
    re.compile(r"\bINDIA\b", re.I),                   # "INDIA" printed on plate border
    re.compile(r"[\s\.\-]+"),                         # spaces, dots, dashes
]


def _strip_junk(s: str) -> str:
    """Remove common non-plate tokens printed on Indian plates."""
    for pat in _STRIP_PATTERNS:
        s = pat.sub("", s)
    return s.upper()


def validate_indian(text: str) -> dict:
    t = re.sub(r"[^A-Z0-9]", "", _strip_junk(text or ""))
    fmt_ok = any(p.match(t) for p in INDIAN_PATTERNS)
    state_ok = t[:2] in STATE_CODES if len(t) >= 2 else False
    return {"normalized": t, "format_valid": fmt_ok, "state_code_valid": state_ok}


def correct_plate_text(text: str) -> str:
    """Coerce a noisy OCR string toward the Indian plate grammar.

    Safe: the correction is applied only if it yields a valid plate;
    otherwise the raw normalized string is returned unchanged.
    """
    s = re.sub(r"[^A-Z0-9]", "", _strip_junk(text or ""))
    n = len(s)
    if 8 <= n <= 11:
        for rto_len in (2, 1):
            rest = s[2 + rto_len:]
            if len(rest) < 5:
                continue
            series, num = rest[:-4], rest[-4:]
            if not (1 <= len(series) <= 3):
                continue
            head = "".join(_TO_LETTER.get(c, c) for c in s[:2])
            rto = "".join(_TO_DIGIT.get(c, c) for c in s[2:2 + rto_len])
            ser = "".join(_TO_LETTER.get(c, c) for c in series)
            number = "".join(_TO_DIGIT.get(c, c) for c in num)
            cand = head + rto + ser + number
            if validate_indian(cand)["format_valid"]:
                return cand
    # BH-series: NNBHNNNNLL
    if 10 <= n <= 12 and "BH" in s:
        idx = s.index("BH")
        if idx == 2:
            num_part = s[4:8]
            ll = s[8:]
            rn = "".join(_TO_DIGIT.get(c, c) for c in num_part)
            rl = "".join(_TO_LETTER.get(c, c) for c in ll)
            cand = s[:2] + "BH" + rn + rl
            if validate_indian(cand)["format_valid"]:
                return cand
    return s


def merge_text_regions(regions: list) -> tuple[str, float]:
    """Merge multi-line OCR output into one reading-order string (handles 2-row Indian plates)."""
    items = []
    for bbox, text, conf in regions:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append({"text": text, "conf": float(conf), "y": min(ys), "x": min(xs),
                      "x2": max(xs), "h": max(ys) - min(ys)})
    if not items:
        return "", 0.0
    # Drop boxes that substantially overlap an already-kept, higher-confidence box -- a
    # duplicate/partial re-read of the same text (a detector artifact on a noisy/reflective
    # crop), not a second group of characters. Naive concatenation of both is exactly how a
    # 10-char plate turns into 13+ chars of "overprediction".
    items.sort(key=lambda r: -r["conf"])
    kept: list[dict] = []
    for it in items:
        dup = False
        for k in kept:
            overlap = max(0.0, min(it["x2"], k["x2"]) - max(it["x"], k["x"]))
            narrower = min(it["x2"] - it["x"], k["x2"] - k["x"])
            same_row = abs(it["y"] - k["y"]) <= 0.6 * (it["h"] or 1)
            if same_row and narrower > 0 and overlap / narrower > 0.5:
                dup = True
                break
        if not dup:
            kept.append(it)
    items = kept
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


# ─── Preprocessing ──────────────────────────────────────────────────────────

def preprocess_crop(arr, upscale: int = 4, clahe: bool = True,
                    sharpen: bool = True, invert_dark: bool = True):
    """Upscale + CLAHE + sharpen a BGR plate crop. Returns BGR numpy array.

    upscale=4 turns a typical 40×120px plate into 160×480px — well within
    PaddleOCR / TrOCR's optimal input range. CLAHE fixes uneven lighting.
    Sharpening enhances character edges blurred by compression/motion.
    Auto-invert flips dark-background plates (white-on-black) to light-background
    so OCR engines (mostly trained on dark-on-white) can read them.
    """
    import cv2
    import numpy as np

    if arr is None or arr.size == 0:
        return arr
    h, w = arr.shape[:2]
    # Upscale
    if upscale > 1:
        arr = cv2.resize(arr, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
    # CLAHE on L-channel (preserves hue/saturation)
    if clahe and arr.ndim == 3:
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_ch = cl.apply(l_ch)
        arr = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    # Unsharp-mask sharpening
    if sharpen:
        blurred = cv2.GaussianBlur(arr, (0, 0), 2)
        arr = cv2.addWeighted(arr, 1.6, blurred, -0.6, 0)
    # Auto-invert: if mean of dark-gray channel dominates, flip (dark-bg plates)
    if invert_dark:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
        if float(gray.mean()) < 100:  # dark background
            arr = cv2.bitwise_not(arr)
    return arr


def tight_crop_to_text(arr):
    """Use morphology to find the tightest bounding box containing text within the plate crop.

    Useful when the GT box is much larger than the actual plate region.
    Returns the cropped sub-array (or original if detection fails).
    """
    import cv2
    import numpy as np

    if arr is None or arr.size == 0:
        return arr
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr.copy()
    # Threshold + morph-close to group text characters
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return arr
    H, W = arr.shape[:2]
    # Pick the largest contour whose aspect ratio looks like a plate
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < W * 0.3 or h < H * 0.1:
            continue
        area = w * h
        if best is None or area > best[4]:
            best = (x, y, w, h, area)
    if best is None:
        return arr
    x, y, w, h = best[:4]
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    sub = arr[y0:y1, x0:x1]
    return sub if sub.size > 0 else arr


# ─── ANPR Module ────────────────────────────────────────────────────────────

_TROCR_FT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "anpr" / "trocr_ft"


def _materialize_meta_buffers(model) -> None:
    """Newer transformers can leave TrOCR's positional-embedding reference buffer on the meta
    device (it's never part of any checkpoint's state dict, so low_cpu_mem_usage=False alone
    doesn't always materialize it -- confirmed still crashing on 5.12.x). It's only ever used as
    a dtype/device reference (model code does `.to(other_tensor.device)` on it), so replacing it
    with zeros is safe. Without this, the first forward pass raises "Cannot copy out of meta
    tensor; no data!" -- same issue hit during trocr_ft training, fixed there the same way."""
    import torch
    for name, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        *parents, leaf = name.split(".")
        owner = model
        for p in parents:
            owner = getattr(owner, p)
        old = getattr(owner, leaf)
        owner.register_buffer(leaf, torch.zeros(old.shape, dtype=old.dtype))

    # TrOCR's sinusoidal positional embedding keeps its weight tensor as a plain attribute
    # (`self.weights = self.get_embedding(...)` in __init__, not a registered buffer/parameter),
    # so it's invisible to named_buffers() above and the checkpoint loader never touches it. If
    # __init__ ran under from_pretrained's meta-device fast-init context, that attribute is a meta
    # tensor forever, regardless of low_cpu_mem_usage. get_embedding is a pure function of shape
    # (sinusoidal, no learned data), so just re-running it on the real device fixes it.
    for mod in model.modules():
        w = getattr(mod, "weights", None)
        if isinstance(w, torch.Tensor) and w.is_meta and hasattr(mod, "get_embedding"):
            mod.weights = mod.get_embedding(w.shape[0], mod.embedding_dim, mod.padding_idx)


class ANPRModule:
    """ANPR module supporting multiple OCR engines with preprocessing and ensemble mode."""

    def __init__(self, ocr_engine: str = "auto",
                 preprocess: bool = True, upscale: int = 4,
                 tta: bool = False, tight_crop: bool = False):
        self._pref = ocr_engine
        self._preprocess = preprocess
        self._upscale = upscale
        self._tta = tta
        self._tight_crop = tight_crop
        self._loaded: dict[str, object] = {}   # engine_name -> engine obj
        self._active_engine: str | None = None
        self.engine_name: str | None = None
        self.engine: object = None
        self._paddle_api = "2x"

    def _ensure_engine(self) -> None:
        if self.engine_name is None:
            self._load_ocr(self._pref)

    def _load_ocr(self, which: str) -> None:
        if which == "ensemble":
            # Try to load all known engines; ensemble logic is in recognize_all()
            for e in ("paddleocr", "easyocr", "trocr_large", "trocr_base", "trocr", "doctr"):
                self._try_load_single(e)
            # Pick first available as the "primary" for backward-compat recognize()
            for e in ("paddleocr", "easyocr", "trocr_large", "trocr_base", "trocr", "doctr"):
                if e in self._loaded:
                    self.engine = self._loaded[e]
                    self.engine_name = e
                    return
        elif which == "auto":
            order = ["paddleocr", "easyocr", "trocr_base", "trocr", "doctr"]
            for e in order:
                if self._try_load_single(e):
                    self.engine = self._loaded[e]
                    self.engine_name = e
                    return
        else:
            if self._try_load_single(which):
                self.engine = self._loaded[which]
                self.engine_name = which
                return
            # The specifically-requested engine failed to load (e.g. a corrupted/incomplete
            # local checkpoint file) -- fall back to the auto-detect ladder instead of leaving
            # ANPR fully disabled. `recognize()`'s "engine" field always reports which engine
            # actually ran, so this is never silent about what produced a given reading.
            order = ["paddleocr", "easyocr", "trocr_base", "trocr", "doctr"]
            for e in order:
                if self._try_load_single(e):
                    self.engine = self._loaded[e]
                    self.engine_name = e
                    return

    def _try_load_single(self, name: str) -> bool:
        if name in self._loaded:
            return True
        try:
            obj = self._build_engine(name)
            if obj is not None:
                self._loaded[name] = obj
                return True
        except Exception:
            pass
        return False

    def _build_engine(self, name: str):
        import os
        if name in ("paddleocr", "paddleocr_server"):
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            from paddleocr import PaddleOCR
            kwargs_3x = {"use_textline_orientation": True, "lang": "en",
                         "enable_mkldnn": False}
            if name == "paddleocr_server":
                kwargs_3x["ocr_version"] = "PP-OCRv4"  # larger server model
            try:
                eng = PaddleOCR(**kwargs_3x)
                self._paddle_api = "3x"
            except TypeError:
                try:
                    eng = PaddleOCR(use_textline_orientation=True, lang="en")
                    self._paddle_api = "3x"
                except Exception:
                    kw = {"use_angle_cls": True, "lang": "en", "show_log": False}
                    if name == "paddleocr_server":
                        kw["ocr_version"] = "PP-OCRv4"
                    eng = PaddleOCR(**kw)
                    self._paddle_api = "2x"
            return ("paddle", eng)

        if name == "easyocr":
            import easyocr
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False
            eng = easyocr.Reader(["en"], gpu=gpu, verbose=False)
            return ("easyocr", eng)

        if name in ("trocr", "trocr_base", "trocr_large"):
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            hf_id = {
                "trocr": "microsoft/trocr-small-printed",
                "trocr_base": "microsoft/trocr-base-printed",
                "trocr_large": "microsoft/trocr-large-printed",
            }[name]
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False
            proc = TrOCRProcessor.from_pretrained(hf_id)
            mdl = VisionEncoderDecoderModel.from_pretrained(hf_id, low_cpu_mem_usage=False)
            _materialize_meta_buffers(mdl)
            if gpu:
                mdl = mdl.cuda()
            mdl.eval()
            return ("trocr", proc, mdl, gpu)

        if name == "trocr_ft":
            ckpt = _TROCR_FT_PATH
            if not ckpt.exists():
                return None
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False
            proc = TrOCRProcessor.from_pretrained(str(ckpt))
            mdl = VisionEncoderDecoderModel.from_pretrained(str(ckpt), low_cpu_mem_usage=False)
            _materialize_meta_buffers(mdl)
            if gpu:
                mdl = mdl.cuda()
            mdl.eval()
            return ("trocr", proc, mdl, gpu)

        if name == "doctr":
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            mdl = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn",
                                pretrained=True)
            try:
                import torch
                if torch.cuda.is_available():
                    mdl = mdl.cuda()
            except Exception:
                pass
            return ("doctr", mdl)

        return None

    def plate_detection_available(self) -> bool:
        return False

    # ── Plate localization (SAM-3, ROI-gated to violators) ─────────────────

    @staticmethod
    def locate_plate_sam3(vehicle_crop, sam3, threshold: float = 0.3) -> dict | None:
        """Find a license-plate box within a vehicle crop using SAM-3 open-vocab detection.

        RF-DETR/COCO has no plate class, so this fills exactly that gap — called ONLY on
        confirmed/candidate violators' vehicle crops (§4c ROI-gated ANPR: never OCR every car).
        `sam3` is a `RoboflowSAM3` client (see modules/roboflow_sam3.py): `.detect(image, prompt,
        conf=...)` returns `[{"box": [x1,y1,x2,y2], "conf": float}, ...]`.
        Returns {bbox, confidence} in the vehicle_crop's own pixel coordinates, or None if no
        plate found / SAM-3 unavailable.
        """
        if sam3 is None or not sam3.available():
            return None
        try:
            dets = sam3.detect(vehicle_crop, "license plate", conf=threshold)
            if not dets:
                return None
            best = max(dets, key=lambda d: d["conf"])
            return {"bbox": [round(v, 1) for v in best["box"]], "confidence": round(best["conf"], 3)}
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def crop_from_bbox(vehicle_crop, bbox):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        return vehicle_crop[y1:y2, x1:x2]

    # ── Public API ────────────────────────────────────────────────────────

    def recognize(self, plate_crop) -> dict:
        """Run the primary engine (or ensemble if configured) on a plate crop."""
        self._ensure_engine()
        if not self._loaded:
            return {"model_unavailable": True, "text": "", "confidence": 0.0, "engine": None,
                    "note": "no OCR engine installed. ANPR not_testable."}
        if self._pref == "ensemble":
            return self.recognize_ensemble(plate_crop)
        return self._recognize_with(self.engine_name, plate_crop)

    def recognize_ensemble(self, plate_crop) -> dict:
        """Run all loaded engines, vote for the best result by plate validity + similarity."""
        self._ensure_engine()
        results = {}
        for eng_name in list(self._loaded.keys()):
            try:
                r = self._recognize_with(eng_name, plate_crop)
                if not r.get("model_unavailable"):
                    results[eng_name] = r
            except Exception:
                pass
        if not results:
            return {"model_unavailable": True, "text": "", "confidence": 0.0,
                    "engine": "ensemble", "note": "all engines failed"}
        best_key = self._vote_best(results)
        r = results[best_key].copy()
        r["engine"] = f"ensemble({best_key})"
        r["all_engines"] = {k: v.get("text", "") for k, v in results.items()}
        return r

    def recognize_all_engines(self, plate_crop) -> dict[str, dict]:
        """Run every loaded engine independently — useful for ablation comparison."""
        self._ensure_engine()
        out = {}
        for eng_name in list(self._loaded.keys()):
            try:
                out[eng_name] = self._recognize_with(eng_name, plate_crop)
            except Exception as e:
                out[eng_name] = {"model_unavailable": True, "text": "", "confidence": 0.0,
                                  "engine": eng_name, "note": str(e)}
        return out

    # ── Internal ──────────────────────────────────────────────────────────

    def _prepare_crop(self, plate_crop):
        """Convert input to BGR numpy, apply preprocessing pipeline."""
        import numpy as np
        arr = self._to_bgr_np(plate_crop)
        if self._tight_crop:
            arr = tight_crop_to_text(arr)
        if self._preprocess:
            arr = preprocess_crop(arr, upscale=self._upscale)
        return arr

    def _recognize_with(self, eng_name: str, plate_crop) -> dict:
        arr = self._prepare_crop(plate_crop)
        if self._tta:
            text, conf = self._run_tta(eng_name, arr)
        else:
            text, conf = self._run_single(eng_name, arr)
        corrected = correct_plate_text(text)
        val = validate_indian(corrected)
        return {"model_unavailable": False, "text": corrected, "raw_text": text,
                "confidence": round(conf, 4), "engine": eng_name, "indian_validation": val}

    def _run_single(self, eng_name: str, arr) -> tuple[str, float]:
        obj = self._loaded.get(eng_name)
        if obj is None:
            return "", 0.0
        kind = obj[0]
        if kind == "paddle":
            return self._run_paddle(obj[1], arr)
        if kind == "easyocr":
            return self._run_easyocr(obj[1], arr)
        if kind == "trocr":
            return self._run_trocr(*obj[1:], arr)
        if kind == "doctr":
            return self._run_doctr(obj[1], arr)
        return "", 0.0

    def _run_tta(self, eng_name: str, arr) -> tuple[str, float]:
        """Run OCR on 4 augmented views and vote by Levenshtein similarity."""
        import cv2
        import numpy as np

        views = [arr]
        # Slight rotation variants
        h, w = arr.shape[:2]
        for angle in (-5, 5):
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rot = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            views.append(rot)
        # Bilateral filter (preserves edges, reduces noise)
        views.append(cv2.bilateralFilter(arr, 9, 75, 75))

        texts = []
        for v in views:
            try:
                t, c = self._run_single(eng_name, v)
                tc = correct_plate_text(t)
                texts.append((tc, c))
            except Exception:
                pass
        if not texts:
            return "", 0.0
        return self._tta_vote(texts)

    @staticmethod
    def _tta_vote(texts: list[tuple[str, float]]) -> tuple[str, float]:
        """Pick the text that has lowest total Levenshtein distance to all others."""
        if len(texts) == 1:
            return texts[0]

        def lev(a, b):
            if a == b:
                return 0
            if not a:
                return len(b)
            if not b:
                return len(a)
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                cur = [i]
                for j, cb in enumerate(b, 1):
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
                prev = cur
            return prev[-1]

        # A view that found nothing is an abstention, not a vote for "blank plate" -- don't let
        # it sit in the same pool as real readings (it would otherwise just average everything
        # toward agreeing-with-nothing). Only fall back to it if every view came up empty.
        nonempty = [t for t in texts if t[0]]
        if not nonempty:
            return texts[0]

        # Prefer format-valid result; among ties, pick lowest total Lev distance
        valid_texts = [t for t in nonempty if validate_indian(t[0])["format_valid"]]
        pool = valid_texts if valid_texts else nonempty
        if not valid_texts:
            # No view produced a plate-shaped string -- before trusting any of them, require
            # that at least one other view came close. A reading that's idiosyncratic to a
            # single rotated/filtered view (no other view agrees, even loosely) is far more
            # likely an artifact of that specific augmentation than the actual plate text; this
            # is exactly how a blurry crop the base view correctly found nothing on turns into a
            # confidently-reported, overpredicted garbage string. Prefer abstaining over that.
            def agreement(t):
                return sum(1 for t2 in texts if t2[0] and lev(t[0], t2[0]) <= max(2, len(t[0]) // 4))
            supported = [t for t in pool if agreement(t) >= 2]
            pool = supported if supported else []
            if not pool:
                return "", 0.0
        best_t, best_c = pool[0]
        best_score = sum(lev(pool[0][0], t[0]) for t in pool)
        for t, c in pool[1:]:
            score = sum(lev(t, t2[0]) for t2 in pool)
            if score < best_score or (score == best_score and c > best_c):
                best_t, best_c, best_score = t, c, score
        return best_t, best_c

    @staticmethod
    def _vote_best(results: dict) -> str:
        """Among ensemble results, pick the engine whose text is most trustworthy.

        Priority: (1) valid Indian plate format, (2) highest confidence.
        Tie-break: lowest total Levenshtein distance to other results.
        """
        def lev(a, b):
            if a == b:
                return 0
            if not a:
                return len(b)
            if not b:
                return len(a)
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                cur = [i]
                for j, cb in enumerate(b, 1):
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
                prev = cur
            return prev[-1]

        valid_keys = [k for k, v in results.items()
                      if v.get("indian_validation", {}).get("format_valid")]
        pool_keys = valid_keys if valid_keys else list(results.keys())
        texts = {k: results[k].get("text", "") for k in pool_keys}
        confs = {k: results[k].get("confidence", 0.0) for k in pool_keys}

        best_key = pool_keys[0]
        best_score = sum(lev(texts[pool_keys[0]], texts[k]) for k in pool_keys)
        for k in pool_keys[1:]:
            score = sum(lev(texts[k], texts[k2]) for k2 in pool_keys)
            if score < best_score or (score == best_score and confs[k] > confs[best_key]):
                best_key, best_score = k, score
        return best_key

    # ── Engine runners ────────────────────────────────────────────────────

    def _run_paddle(self, eng, arr) -> tuple[str, float]:
        import numpy as np
        regions = self._paddle_regions_from(eng, arr)
        if not regions:
            return "", 0.0
        return merge_text_regions(regions)

    def _paddle_regions_from(self, eng, arr) -> list:
        regions = []
        if self._paddle_api == "3x" and hasattr(eng, "predict"):
            try:
                res = eng.predict(arr)
                for item in (res or []):
                    d = item if isinstance(item, dict) else {}
                    try:
                        d = dict(item)
                    except Exception:
                        pass
                    texts = d.get("rec_texts")
                    scores = d.get("rec_scores", [])
                    polys = d.get("rec_polys") or d.get("dt_polys")
                    if texts is None:
                        try:
                            texts = item["rec_texts"]
                            scores = item["rec_scores"]
                            polys = item.get("rec_polys", item.get("dt_polys"))
                        except Exception:
                            texts = None
                    if texts:
                        for i, t in enumerate(texts):
                            box = polys[i] if polys is not None and i < len(polys) else [[0, 0], [1, 0], [1, 1], [0, 1]]
                            box = [[float(p[0]), float(p[1])] for p in box]
                            regions.append((box, t, float(scores[i]) if i < len(scores) else 1.0))
                if regions:
                    return regions
            except Exception:
                pass
        try:
            res = eng.ocr(arr, cls=True)
            line = (res or [[]])[0] or []
            for box, (text, score) in line:
                regions.append(([[float(p[0]), float(p[1])] for p in box], text, float(score)))
        except Exception:
            pass
        return regions

    def _run_easyocr(self, eng, arr) -> tuple[str, float]:
        import numpy as np
        # EasyOCR expects RGB
        rgb = arr[:, :, ::-1] if arr.ndim == 3 else arr
        res = eng.readtext(rgb)
        if not res:
            return "", 0.0
        return merge_text_regions(res)

    def _run_trocr(self, proc, mdl, gpu, arr) -> tuple[str, float]:
        import torch
        from PIL import Image
        # BGR -> RGB PIL
        rgb = arr[:, :, ::-1] if arr.ndim == 3 else arr
        pil = Image.fromarray(rgb)
        pix = proc(images=pil, return_tensors="pt").pixel_values
        if gpu:
            pix = pix.cuda()
        # Indian plates top out around 10-12 chars (BH-series longest); 32 tokens of headroom
        # let the decoder run past the actual plate and hallucinate extra digits/letters on a
        # noisy crop instead of stopping. 16 is generous over the real max while curbing that.
        with torch.no_grad():
            ids = mdl.generate(pix, max_new_tokens=16)
        txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
        return txt, 0.5

    def _run_doctr(self, mdl, arr) -> tuple[str, float]:
        import numpy as np
        from doctr.io import DocumentFile
        # doctr expects HWC uint8 RGB
        rgb = arr[:, :, ::-1] if arr.ndim == 3 else np.stack([arr] * 3, axis=2)
        result = mdl([rgb])
        words = []
        confs = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        words.append(word.value)
                        confs.append(word.confidence)
        if not words:
            return "", 0.0
        text = "".join(words)
        conf = sum(confs) / len(confs)
        return text, float(conf)

    @staticmethod
    def _to_bgr_np(crop):
        """Convert any input (np BGR, PIL, path) to numpy BGR uint8."""
        import numpy as np
        from PIL import Image
        if isinstance(crop, (str, Path)):
            img = Image.open(str(crop)).convert("RGB")
            return np.asarray(img)[:, :, ::-1].copy()
        if hasattr(crop, "mode"):  # PIL
            return np.asarray(crop.convert("RGB"))[:, :, ::-1].copy()
        arr = np.asarray(crop)
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=2)
        if arr.shape[2] == 4:
            return arr[:, :, :3]
        return arr
