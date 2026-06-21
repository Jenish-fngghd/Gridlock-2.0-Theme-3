"""Track B (Lightning/H200) — SAM 3 zero-shot test on ALL tasks.

SAM 3 (Meta, Nov 2025) does open-vocab detection+segmentation from short noun-phrase concepts,
returning a mask + id for every matching instance. We probe EVERY mandated violation + the
India-specific detection classes + license plates, and report a per-task hit-rate (plus clean
instance masks usable for evidence crops).

Hardware: H100/H200 class. RUN ON LIGHTNING.

Backends (try whichever installs cleanly):
  A) ultralytics SAM3 wrapper:  pip install "ultralytics>=8.4"
  B) HF facebook/sam3:          pip install "transformers>=4.49"

Output:
  results/sam3/SUMMARY.md  (per-task hit rate)
  results/sam3/sam3_results.json
  results/sam3/<img>_<task>.jpg  (mask overlays for hits)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.eval.violation_prompts import MANDATED, VIOLATION_PROMPTS


def _progress(msg):
    print(msg, flush=True)


def _slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s)[:30]


def _instances_from_hf_output(out, image_size, score_threshold=0.25, min_mask_area=80):
    """Return mask-derived boxes from a best-effort SAM3 HF output parse."""
    import numpy as np
    import torch
    from PIL import Image

    masks = getattr(out, "pred_masks", None)
    if masks is None:
        return []

    if isinstance(masks, (list, tuple)):
        masks = masks[0]
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().float().cpu()
    else:
        masks = torch.as_tensor(masks).float()

    while masks.ndim > 3:
        masks = masks[0]
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        return []

    scores = None
    for name in ("scores", "pred_scores", "objectness_scores", "confidence_scores"):
        v = getattr(out, name, None)
        if v is not None:
            scores = v.detach().float().cpu().flatten() if hasattr(v, "detach") else torch.as_tensor(v).float().flatten()
            break

    width, height = image_size
    instances = []
    for idx, mask in enumerate(masks):
        if scores is not None and idx < len(scores) and float(scores[idx]) < score_threshold:
            continue
        if float(mask.min()) < 0.0 or float(mask.max()) > 1.0:
            mask = mask.sigmoid()
        binary = (mask.numpy() > 0.5).astype("uint8")
        if binary.shape != (height, width):
            binary_img = Image.fromarray(binary * 255).resize((width, height), Image.Resampling.NEAREST)
            binary = (np.array(binary_img) > 0).astype("uint8")
        ys, xs = np.where(binary > 0)
        if len(xs) < min_mask_area:
            continue
        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        instances.append({"box": box, "area": int(len(xs)), "mask": binary})
    return instances


def _save_hf_overlay(image, instances, out_path, label):
    from PIL import ImageDraw

    vis = image.convert("RGBA")
    overlay = ImageDraw.Draw(vis, "RGBA")
    for inst in instances:
        box = inst["box"]
        mask = inst["mask"]
        ys, xs = mask.nonzero()
        if len(xs):
            for x, y in zip(xs[::8], ys[::8]):
                overlay.point((int(x), int(y)), fill=(255, 0, 0, 80))
        overlay.rectangle(box, outline=(255, 0, 0, 255), width=3)
        overlay.text((box[0], box[1]), label, fill=(255, 255, 255, 255))
    vis.convert("RGB").save(out_path)


def run_ultralytics(frames, out_dir):
    from ultralytics import SAM
    weights = Path("sam3.pt")
    if not weights.exists():
        raise FileNotFoundError(
            "sam3.pt was not found in the current directory. "
            "Download/copy the SAM 3 checkpoint there, or rerun with --backend hf."
        )
    _progress("[sam3] loading ultralytics SAM3...")
    model = SAM(str(weights))
    _progress("[sam3] ultralytics SAM3 loaded")
    records = []
    hits = {k: 0 for k in VIOLATION_PROMPTS}
    totals = {k: 0 for k in VIOLATION_PROMPTS}
    for i, fp in enumerate(frames, 1):
        _progress(f"[sam3] image {i}/{len(frames)}: {fp.name}")
        per = {"image": fp.name, "tasks": {}}
        for task_i, (task, (label, phrases, _color)) in enumerate(VIOLATION_PROMPTS.items(), 1):
            if task_i == 1 or task_i % 4 == 0:
                _progress(f"[sam3]   task {task_i}/{len(VIOLATION_PROMPTS)}: {task}")
            n_task = 0
            for ph in phrases:
                try:
                    res = model(str(fp), prompt=ph, verbose=False)
                    r0 = res[0]
                    n = len(r0.boxes) if getattr(r0, "boxes", None) is not None else 0
                    n_task += int(n)
                    if n:
                        r0.save(filename=str(out_dir / f"{fp.stem}_{task}_{_slug(ph)}.jpg"))
                except Exception as e:  # noqa: BLE001
                    per["tasks"].setdefault(task, {})["error"] = f"{type(e).__name__}:{str(e)[:60]}"
            per["tasks"].setdefault(task, {})["label"] = label
            per["tasks"][task]["n"] = n_task
            if n_task:
                hits[task] += 1; totals[task] += n_task
        records.append(per)
        if i % 10 == 0:
            _progress(f"[sam3] {i}/{len(frames)} images")
    return records, hits, totals


def run_hf(frames, out_dir, score_threshold=0.25, min_mask_area=80):
    import torch
    from PIL import Image
    from PIL import ImageFile
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as e:
        raise ImportError(
            "This transformers install does not expose Sam3Model/Sam3Processor. "
            "Use a separate SAM3 environment and install transformers from a build "
            "that includes SAM3, or use --backend ultralytics with sam3.pt present."
        ) from e
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _progress("[sam3] loading HF processor/model...")
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3", dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    _progress(f"[sam3] HF facebook/sam3 loaded ({dtype})")
    records = []
    hits = {k: 0 for k in VIOLATION_PROMPTS}
    totals = {k: 0 for k in VIOLATION_PROMPTS}
    for i, fp in enumerate(frames, 1):
        _progress(f"[sam3] image {i}/{len(frames)}: {fp.name}")
        try:
            image = Image.open(fp).convert("RGB")
        except Exception as e:  # noqa: BLE001
            _progress(f"[sam3] skipped unreadable image: {fp} ({type(e).__name__})")
            continue
        per = {"image": fp.name, "tasks": {}}
        for task_i, (task, (label, phrases, _color)) in enumerate(VIOLATION_PROMPTS.items(), 1):
            if task_i == 1 or task_i % 4 == 0:
                _progress(f"[sam3]   task {task_i}/{len(VIOLATION_PROMPTS)}: {task}")
            n_task = 0
            boxes = []
            for ph in phrases:
                try:
                    inputs = proc(images=image, text=ph, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = model(**inputs)
                    instances = _instances_from_hf_output(
                        out, image.size, score_threshold=score_threshold, min_mask_area=min_mask_area)
                    if instances:
                        _save_hf_overlay(
                            image, instances, out_dir / f"{fp.stem}_{task}_{_slug(ph)}.jpg", task)
                    n_task += len(instances)
                    boxes.extend(inst["box"] for inst in instances)
                except Exception as e:  # noqa: BLE001
                    per["tasks"].setdefault(task, {})["error"] = f"{type(e).__name__}:{str(e)[:60]}"
            per["tasks"].setdefault(task, {})["label"] = label
            per["tasks"][task]["n"] = n_task
            per["tasks"][task]["boxes"] = boxes[:25]
            if n_task:
                hits[task] += 1; totals[task] += n_task
        records.append(per)
        if i % 10 == 0:
            _progress(f"[sam3] {i}/{len(frames)} images")
    return records, hits, totals


def run(images_dir: Path, limit: int, out_dir: Path, backend: str,
        score_threshold: float = 0.25, min_mask_area: int = 80) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(images_dir.rglob("*.jpg"))[:limit]
    if not frames:
        return {"error": f"no images under {images_dir}"}
    total_phrases = sum(len(v[1]) for v in VIOLATION_PROMPTS.values())
    _progress(f"[sam3] backend={backend} images={len(frames)} tasks={len(VIOLATION_PROMPTS)} "
              f"phrases/image={total_phrases} total_phrase_calls={len(frames) * total_phrases}")
    t0 = time.time()
    if backend == "ultralytics":
        records, hits, totals = run_ultralytics(frames, out_dir)
    else:
        records, hits, totals = run_hf(
            frames, out_dir, score_threshold=score_threshold, min_mask_area=min_mask_area)
    elapsed = time.time() - t0
    (out_dir / "sam3_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {"model": "SAM 3", "backend": backend, "images": len(frames),
               "seconds": round(elapsed, 1), "img_per_sec": round(len(frames) / max(elapsed, 1e-6), 3),
               "per_task_hit_rate": {k: {"images_with_hit": hits[k], "of": len(frames),
                                         "total_instances": totals[k],
                                         "label": VIOLATION_PROMPTS[k][0]} for k in VIOLATION_PROMPTS}}
    _write_summary(summary, out_dir)
    return summary


def _write_summary(s, out_dir):
    L = [f"# SAM 3 — zero-shot capability on ALL tasks\n",
         f"- Model: {s['model']} ({s['backend']}) · images: {s['images']} · {s['img_per_sec']} img/s\n",
         "## Per-task hit rate (images with ≥1 matching instance)\n",
         "| Task | Images w/ hit | Total instances |", "|---|---|---|"]
    for k, v in s["per_task_hit_rate"].items():
        tag = " ⭐(mandated)" if k in MANDATED else ""
        L.append(f"| {v['label']}{tag} | {v['images_with_hit']}/{v['of']} | {v['total_instances']} |")
    L.append("\n> Hit-rate = does the concept fire (no GT → capability signal, not accuracy). Inspect "
             "the saved mask overlays. High hit-rate on a violation/India-class → auto-label + distill "
             "into our fast detector (data engine).")
    (out_dir / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="results/sam3")
    ap.add_argument("--backend", default="ultralytics", choices=["ultralytics", "hf"])
    ap.add_argument("--score-threshold", type=float, default=0.25)
    ap.add_argument("--min-mask-area", type=int, default=80)
    args = ap.parse_args()
    result = run(Path(args.images), args.limit, Path(args.out), args.backend,
                 score_threshold=args.score_threshold, min_mask_area=args.min_mask_area)
    if "error" in result:
        print("ERROR:", result["error"]); return 1
    print(json.dumps(result["per_task_hit_rate"], indent=2))
    print(f"\n[sam3] SUMMARY -> {args.out}/SUMMARY.md  ({result['img_per_sec']} img/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
