"""Track B (Lightning/H200) — LocateAnything-3B zero-shot test on ALL tasks.

Tests whether NVIDIA's open-vocab grounding VLM (nvidia/LocateAnything-3B) can detect EVERY
mandated violation + the India-specific detection classes + license plates, zero-shot, from
natural-language phrases. One foundation model vs. our whole stack of specialized modules
(the §3.10 data-engine question; J1's teacher/verifier role).

Hardware: needs ~12 GB VRAM (A100/H200). bf16. Will OOM on the 4 GB laptop — RUN ON LIGHTNING.
License: NVIDIA non-commercial → teacher/baseline only, never shipped.

Output per run:
  - results/locateanything/<img>_annot.jpg   (boxes colored per task)
  - results/locateanything/locateanything_results.json  (per-image, per-task boxes)
  - results/locateanything/SUMMARY.md         (per-task hit-rate table — the headline)

Setup on Lightning:
    pip install "transformers>=4.49" accelerate pillow torch
Run:
    python -m src.eval.test_locateanything \
        --images datasets/idd-detection/IDD_Detection/JPEGImages --limit 60 \
        --out results/locateanything
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.eval.violation_prompts import MANDATED, VIOLATION_PROMPTS

MODEL_ID = "nvidia/LocateAnything-3B"


def _progress(msg):
    print(msg, flush=True)


def load_model():
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _progress(f"[locateanything] loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    _progress(f"[locateanything] loaded {MODEL_ID} ({dtype}, cuda={torch.cuda.is_available()})")
    return model, proc, tok


def _ground(model, proc, tok, image, phrase):
    """Best-effort grounding across the documented API patterns. Returns ([boxes], api_tag)."""
    import torch
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
    try:
        prompt = f"Locate all instances matching: {phrase}. Return boxes."
        inputs = proc(images=image, text=prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=512)
        text = proc.batch_decode(gen, skip_special_tokens=True)[0]
        return (_extract_boxes(text) or []), "generate"
    except Exception as e:  # noqa: BLE001
        return [], f"unsupported_api:{type(e).__name__}"


def _extract_boxes(out):
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
        nums = re.findall(r"(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)[,\s]+"
                          r"(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)", out)
        if nums:
            return [[float(a), float(b), float(c), float(d)] for a, b, c, d in nums]
    return None


def run(images_dir: Path, limit: int, out_dir: Path) -> dict:
    from PIL import Image, ImageDraw
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(images_dir.rglob("*.jpg"))[:limit]
    if not frames:
        return {"error": f"no images under {images_dir}"}
    total_phrases = sum(len(v[1]) for v in VIOLATION_PROMPTS.values())
    _progress(f"[locateanything] images={len(frames)} tasks={len(VIOLATION_PROMPTS)} "
              f"phrases/image={total_phrases} total_phrase_calls={len(frames) * total_phrases}")
    model, proc, tok = load_model()

    api_used = set()
    # task -> images-with-a-hit
    hits = {k: 0 for k in VIOLATION_PROMPTS}
    total_boxes = {k: 0 for k in VIOLATION_PROMPTS}
    records = []
    skipped = []
    t0 = time.time()
    for i, fp in enumerate(frames, 1):
        _progress(f"[locateanything] image {i}/{len(frames)}: {fp.name}")
        try:
            image = Image.open(fp).convert("RGB")
        except Exception as e:  # noqa: BLE001
            skipped.append({"image": str(fp), "error": f"{type(e).__name__}: {e}"})
            _progress(f"[locateanything] skipped unreadable image: {fp} ({type(e).__name__})")
            continue
        vis = image.copy()
        draw = ImageDraw.Draw(vis)
        per = {"image": fp.name, "tasks": {}}
        for task_i, (task, (label, phrases, color)) in enumerate(VIOLATION_PROMPTS.items(), 1):
            if task_i == 1 or task_i % 4 == 0:
                _progress(f"[locateanything]   task {task_i}/{len(VIOLATION_PROMPTS)}: {task}")
            boxes = []
            for ph in phrases:                       # union over phrasings (best recall)
                b, api = _ground(model, proc, tok, image, ph)
                api_used.add(api)
                boxes.extend(b)
            per["tasks"][task] = {"label": label, "n": len(boxes), "boxes": boxes[:25]}
            if boxes:
                hits[task] += 1
                total_boxes[task] += len(boxes)
                for bx in boxes:
                    try:
                        draw.rectangle(bx, outline=color, width=3)
                        draw.text((bx[0], bx[1]), task, fill=color)
                    except Exception:
                        pass
        records.append(per)
        vis.save(out_dir / f"{fp.stem}_annot.jpg")
        if i % 10 == 0:
            _progress(f"[locateanything] {i}/{len(frames)} images")

    elapsed = time.time() - t0
    (out_dir / "locateanything_results.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    if skipped:
        (out_dir / "skipped_images.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    summary = {
        "model": MODEL_ID, "images": len(records), "requested_images": len(frames),
        "skipped_images": len(skipped), "seconds": round(elapsed, 1),
        "img_per_sec": round(len(records) / max(elapsed, 1e-6), 3),
        "api_paths_used": sorted(api_used),
        "per_task_hit_rate": {k: {"images_with_hit": hits[k], "of": len(records),
                                  "total_boxes": total_boxes[k],
                                  "label": VIOLATION_PROMPTS[k][0]} for k in VIOLATION_PROMPTS},
    }
    _write_summary(summary, out_dir)
    return summary


def _write_summary(s, out_dir):
    L = [f"# LocateAnything-3B — zero-shot capability on ALL tasks\n",
         f"- Model: {s['model']} · images: {s['images']} · {s['img_per_sec']} img/s\n",
         "## Per-task hit rate (images where the model returned ≥1 box)\n",
         "| Task | Images w/ hit | Total boxes |", "|---|---|---|"]
    for k, v in s["per_task_hit_rate"].items():
        tag = " ⭐(mandated)" if k in MANDATED else ""
        L.append(f"| {v['label']}{tag} | {v['images_with_hit']}/{v['of']} | {v['total_boxes']} |")
    L.append("\n> Hit-rate = does it fire on the right concept. With no GT this is a capability "
             "signal, not accuracy. Inspect `*_annot.jpg` to judge correctness. A high hit-rate on "
             "a violation → candidate to auto-label + distill into our fast detector (data engine).")
    if any("unsupported_api" in a for a in s["api_paths_used"]):
        L.append("\n⚠️ Some calls hit `unsupported_api` — adapt `_ground()` to the model card's exact "
                 "method names (VLM APIs drift); the prompt set stays the same.")
    (out_dir / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="results/locateanything")
    args = ap.parse_args()
    result = run(Path(args.images), args.limit, Path(args.out))
    if "error" in result:
        print("ERROR:", result["error"]); return 1
    print(json.dumps(result["per_task_hit_rate"], indent=2))
    print(f"\n[locateanything] SUMMARY -> {args.out}/SUMMARY.md  ({result['img_per_sec']} img/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
