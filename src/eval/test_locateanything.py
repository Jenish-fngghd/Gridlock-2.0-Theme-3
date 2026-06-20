"""Track B (Lightning/H200) — LocateAnything-3B zero-shot violation grounding.

Tests whether NVIDIA's open-vocab grounding VLM (nvidia/LocateAnything-3B) can detect our
traffic violations directly from natural-language phrases, with NO fine-tuning — a single
foundation model vs. our specialized per-violation modules (the §3.10 "foundation-model data
engine" question, and J1's teacher/verifier role).

Cannot run on the 4 GB laptop (model needs ~12 GB). RUN ON LIGHTNING (H200).
License: NVIDIA non-commercial → teacher/baseline only, NOT shippable (consistent with J1).

What it does: for each image, prompt the model with violation phrases and save the returned
boxes + an annotated image. Phrases probe every paradigm:
  - helmet:        "motorcyclist without helmet", "person wearing helmet on motorcycle"
  - triple riding: "three people riding one motorcycle"
  - ANPR:          "license plate"
  - wrong-side / parking are context-dependent; we still probe "vehicle facing wrong direction"

Setup on Lightning:
    pip install "transformers>=4.49" accelerate pillow torch
Run:
    python -m src.eval.test_locateanything --images datasets/idd-detection/IDD_Detection/JPEGImages \
        --limit 40 --out results/locateanything
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ID = "nvidia/LocateAnything-3B"

# Violation probes (phrase grounding). Edit freely — this is the whole point of open-vocab.
PROMPTS = {
    "helmet_no": "motorcyclist without helmet",
    "helmet_yes": "person wearing a helmet on a motorcycle",
    "triple_riding": "three people riding on one motorcycle",
    "license_plate": "license plate",
    "auto_rickshaw": "auto rickshaw",
    "wrong_side": "vehicle facing the wrong direction",
}


def load_model():
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="auto")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, proc, tok


def _ground(model, proc, tok, image, phrase):
    """Best-effort call into LocateAnything's grounding API. The repo ships task-specific
    convenience methods; their exact names can shift, so we try the documented patterns and
    fall back gracefully. Returns list of [x1,y1,x2,y2] (and optionally scores)."""
    # Pattern 1: a high-level detect/ground method on the model
    for meth in ("detect", "ground", "locate", "predict"):
        fn = getattr(model, meth, None)
        if callable(fn):
            try:
                out = fn(image, phrase)
                boxes = _extract_boxes(out)
                if boxes is not None:
                    return boxes, f"model.{meth}"
            except Exception:
                pass
    # Pattern 2: chat-style with the processor (VLM prompt -> parse boxes from text/json)
    try:
        prompt = f"Locate all instances matching: {phrase}. Return boxes."
        inputs = proc(images=image, text=prompt, return_tensors="pt")
        import torch
        if torch.cuda.is_available():
            inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=512)
        text = proc.batch_decode(gen, skip_special_tokens=True)[0]
        boxes = _extract_boxes(text)
        return (boxes or []), "generate"
    except Exception as e:  # noqa: BLE001
        return [], f"unsupported_api: {type(e).__name__}: {str(e)[:80]}"


def _extract_boxes(out):
    """Coerce various return shapes (dict / objects / json-in-text) into [[x1,y1,x2,y2],...]."""
    import re
    if out is None:
        return None
    if isinstance(out, dict):
        for k in ("boxes", "bboxes", "pred_boxes"):
            if k in out:
                return [list(map(float, b))[:4] for b in out[k]]
    if hasattr(out, "xyxy"):
        return [list(map(float, b))[:4] for b in out.xyxy]
    if isinstance(out, str):
        # find json arrays of 4 numbers
        nums = re.findall(r"\[?\s*(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)[,\s]+"
                          r"(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)\s*\]?", out)
        if nums:
            return [[float(a), float(b), float(c), float(d)] for a, b, c, d in nums]
    return None


def run(images_dir: Path, limit: int, out_dir: Path) -> dict:
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(list(images_dir.rglob("*.jpg")))[:limit]
    if not frames:
        return {"error": f"no images under {images_dir}"}
    model, proc, tok = load_model()
    api_used = set()
    records = []
    for fp in frames:
        image = Image.open(fp).convert("RGB")
        per = {"image": fp.name, "detections": {}}
        for tag, phrase in PROMPTS.items():
            boxes, api = _ground(model, proc, tok, image, phrase)
            api_used.add(api)
            per["detections"][tag] = {"phrase": phrase, "n_boxes": len(boxes), "boxes": boxes[:20]}
        records.append(per)
        _annotate_and_save(fp, image, per, out_dir)
    (out_dir / "locateanything_results.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    return {"model": MODEL_ID, "images": len(frames), "api_paths_used": sorted(api_used),
            "out": str(out_dir), "prompts": PROMPTS,
            "note": "Inspect annotated images + json. Non-commercial license -> teacher/baseline only."}


def _annotate_and_save(fp, image, per, out_dir):
    try:
        from PIL import ImageDraw
        vis = image.copy()
        d = ImageDraw.Draw(vis)
        colors = {"helmet_no": "red", "triple_riding": "orange", "license_plate": "yellow",
                  "wrong_side": "magenta", "auto_rickshaw": "cyan", "helmet_yes": "green"}
        for tag, info in per["detections"].items():
            for b in info["boxes"]:
                d.rectangle(b, outline=colors.get(tag, "white"), width=3)
                d.text((b[0], b[1]), tag, fill=colors.get(tag, "white"))
        vis.save(out_dir / f"la_{fp.stem}.jpg")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="results/locateanything")
    args = ap.parse_args()
    result = run(Path(args.images), args.limit, Path(args.out))
    if "error" in result:
        print("ERROR:", result["error"])
        return 1
    print(json.dumps(result, indent=2))
    print("\n[NOTE] If api_paths_used shows 'unsupported_api', open the model card's example "
          "code on HF and adapt _ground() to its exact method names — APIs for brand-new VLMs "
          "shift. The phrase list (PROMPTS) is the part you tune for each violation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
