"""Track B (Lightning/H200) — SAM 3 zero-shot open-vocabulary violation concepts.

SAM 3 (Meta, Nov 2025) does open-vocabulary detection+segmentation from short noun-phrase
"concept" prompts, returning a mask + id for EVERY matching instance. We test whether concept
prompts can flag our violations zero-shot (and give clean instance masks for evidence crops).

Cannot run on the 4 GB laptop. RUN ON LIGHTNING (H200).

Two supported backends (try whichever installs cleanly on Lightning):
  A) Ultralytics SAM3 wrapper (simplest):  pip install "ultralytics>=8.4"
  B) Meta's official `sam3` package / HF `facebook/sam3` (most faithful).

Concept prompts probe each paradigm:
  helmet:        "motorcyclist without helmet" / "helmet"
  triple riding: "motorcycle with three riders"
  detection:     "auto rickshaw", "cycle rickshaw"  (the India-specific COCO-gap classes)
  ANPR:          "license plate"

Run:
    python -m src.eval.test_sam3 --images datasets/idd-detection/IDD_Detection/JPEGImages \
        --limit 40 --out results/sam3 --backend ultralytics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONCEPTS = [
    "motorcyclist without helmet",
    "helmet",
    "motorcycle with three riders",
    "auto rickshaw",
    "cycle rickshaw",
    "license plate",
    "person",
]


def run_ultralytics(frames, concepts, out_dir):
    from ultralytics import SAM
    model = SAM("sam3.pt")  # ultralytics auto-downloads the SAM3 checkpoint
    records = []
    for fp in frames:
        per = {"image": fp.name, "concepts": {}}
        for c in concepts:
            try:
                res = model(str(fp), prompt=c)          # text/concept prompt
                r0 = res[0]
                n = len(r0.boxes) if getattr(r0, "boxes", None) is not None else 0
                per["concepts"][c] = {"n_instances": int(n)}
                if n:
                    r0.save(filename=str(out_dir / f"sam3_{fp.stem}_{_slug(c)}.jpg"))
            except Exception as e:  # noqa: BLE001
                per["concepts"][c] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
        records.append(per)
    return records


def run_hf(frames, concepts, out_dir):
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor  # names per HF facebook/sam3 card
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    records = []
    for fp in frames:
        image = Image.open(fp).convert("RGB")
        per = {"image": fp.name, "concepts": {}}
        for c in concepts:
            try:
                inputs = proc(images=image, text=c, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                masks = getattr(out, "pred_masks", None)
                n = 0 if masks is None else int(masks.shape[0])
                per["concepts"][c] = {"n_instances": n}
            except Exception as e:  # noqa: BLE001
                per["concepts"][c] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
        records.append(per)
    return records


def _slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s)[:30]


def run(images_dir: Path, limit: int, out_dir: Path, backend: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(list(images_dir.rglob("*.jpg")))[:limit]
    if not frames:
        return {"error": f"no images under {images_dir}"}
    if backend == "ultralytics":
        records = run_ultralytics(frames, CONCEPTS, out_dir)
    else:
        records = run_hf(frames, CONCEPTS, out_dir)
    (out_dir / "sam3_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    # aggregate hit rate per concept
    agg = {}
    for c in CONCEPTS:
        hits = sum(1 for r in records if r["concepts"].get(c, {}).get("n_instances", 0) > 0)
        agg[c] = {"images_with_hit": hits, "of": len(frames)}
    return {"model": "SAM 3", "backend": backend, "images": len(frames),
            "concept_hit_rate": agg, "out": str(out_dir),
            "note": "Inspect masks/boxes. Concept prompts are the tuning surface for each violation."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="results/sam3")
    ap.add_argument("--backend", default="ultralytics", choices=["ultralytics", "hf"])
    args = ap.parse_args()
    result = run(Path(args.images), args.limit, Path(args.out), args.backend)
    if "error" in result:
        print("ERROR:", result["error"]); return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
