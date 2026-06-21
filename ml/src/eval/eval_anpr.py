"""Phase 3/5 — ANPR evaluation with full ablation suite.

Runs every improvement strategy independently + combined, prints a comparison table,
and saves per-config results to results/ + run_history.csv.

GT source: Indian-LP (sirishan) — VOC XML with plate box AND plate text as <name>.
Metrics: plate exact-match accuracy, CER, 1-NED, Indian-format valid rate.

Quick run (single config):
  python -m src.eval.eval_anpr --limit 500 --config preprocessed --engine paddleocr

Full ablation (all configs, recommended for Lightning):
  python -m src.eval.eval_anpr --limit 1000 --mode ablate

Fine-tune checkpoint eval:
  python -m src.eval.eval_anpr --limit 1000 --config trocr_ft
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from src.modules.anpr import ANPRModule, preprocess_crop, validate_indian
from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)

INDIAN_ROOT = REPO_ROOT / "datasets" / "Indian LP" / "sirishan"


def levenshtein(a: str, b: str) -> int:
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


def norm(s: str) -> str:
    import re
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def load_indian(limit: int) -> list[tuple[str, list, str]]:
    out = []
    for dp, _d, files in os.walk(INDIAN_ROOT):
        for f in sorted(files):
            if not f.endswith(".xml"):
                continue
            xml = os.path.join(dp, f)
            try:
                root = ET.parse(xml).getroot()
            except Exception:
                continue
            stem = f[:-4]
            img = None
            for cand in (stem, stem + ".jpeg", stem + ".jpg", stem + ".png"):
                p = os.path.join(dp, cand)
                if os.path.exists(p):
                    img = p
                    break
            if img is None:
                fn = root.findtext("filename")
                if fn and os.path.exists(os.path.join(dp, fn)):
                    img = os.path.join(dp, fn)
            if img is None:
                continue
            obj = root.find("object")
            if obj is None:
                continue
            text = (obj.findtext("name") or "").strip()
            bb = obj.find("bndbox")
            if bb is None or not text:
                continue
            box = [float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0")),
                   float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0"))]
            out.append((img, box, text))
            if limit and len(out) >= limit:
                return out
    return out


# ─── Config definitions ─────────────────────────────────────────────────────

# Each config: (engine, preprocess, upscale, tta, tight_crop, description)
ABLATION_CONFIGS = {
    "baseline": {
        "engine": "paddleocr", "preprocess": False, "upscale": 1,
        "tta": False, "tight_crop": False,
        "desc": "PaddleOCR PP-OCRv5 mobile, no preprocessing (current best: 0.449)",
    },
    "preprocess_only": {
        "engine": "paddleocr", "preprocess": True, "upscale": 4,
        "tta": False, "tight_crop": False,
        "desc": "PaddleOCR + 4× upscale + CLAHE + sharpen + auto-invert",
    },
    "preprocess_tta": {
        "engine": "paddleocr", "preprocess": True, "upscale": 4,
        "tta": True, "tight_crop": False,
        "desc": "PaddleOCR + preprocessing + TTA (4-view vote)",
    },
    "preprocess_tight": {
        "engine": "paddleocr", "preprocess": True, "upscale": 4,
        "tta": False, "tight_crop": True,
        "desc": "PaddleOCR + preprocessing + morphology tight-crop",
    },
    "ppocr_server": {
        "engine": "paddleocr_server", "preprocess": True, "upscale": 4,
        "tta": False, "tight_crop": False,
        "desc": "PP-OCRv4 server model (larger) + preprocessing",
    },
    "ppocr_server_tta": {
        "engine": "paddleocr_server", "preprocess": True, "upscale": 4,
        "tta": True, "tight_crop": False,
        "desc": "PP-OCRv4 server + preprocessing + TTA",
    },
    "easyocr_preprocess": {
        "engine": "easyocr", "preprocess": True, "upscale": 4,
        "tta": False, "tight_crop": False,
        "desc": "EasyOCR (GPU) + preprocessing",
    },
    "easyocr_tta": {
        "engine": "easyocr", "preprocess": True, "upscale": 4,
        "tta": True, "tight_crop": False,
        "desc": "EasyOCR + preprocessing + TTA",
    },
    "trocr_base": {
        "engine": "trocr_base", "preprocess": True, "upscale": 4,
        "tta": False, "tight_crop": False,
        "desc": "TrOCR-base-printed (HF zero-shot) + preprocessing",
    },
    "trocr_large": {
        "engine": "trocr_large", "preprocess": True, "upscale": 4,
        "tta": False, "tight_crop": False,
        "desc": "TrOCR-large-printed (HF zero-shot) + preprocessing",
    },
    "trocr_large_tta": {
        "engine": "trocr_large", "preprocess": True, "upscale": 4,
        "tta": True, "tight_crop": False,
        "desc": "TrOCR-large-printed + preprocessing + TTA",
    },
    "trocr_ft": {
        "engine": "trocr_ft", "preprocess": True, "upscale": 4,
        "tta": False, "tight_crop": False,
        "desc": "TrOCR fine-tuned on Indian-LP (checkpoints/anpr/trocr_ft) + preprocessing",
    },
    "trocr_ft_tta": {
        "engine": "trocr_ft", "preprocess": True, "upscale": 4,
        "tta": True, "tight_crop": False,
        "desc": "TrOCR fine-tuned + preprocessing + TTA",
    },
    "doctr": {
        "engine": "doctr", "preprocess": True, "upscale": 4,
        "tta": False, "tight_crop": False,
        "desc": "DocTR (db_resnet50+crnn_vgg16_bn) + preprocessing",
    },
    "ensemble_all": {
        "engine": "ensemble", "preprocess": True, "upscale": 4,
        "tta": False, "tight_crop": False,
        "desc": "Ensemble: PaddleOCR + EasyOCR + TrOCR-large + vote",
    },
    "ensemble_tta": {
        "engine": "ensemble", "preprocess": True, "upscale": 4,
        "tta": True, "tight_crop": False,
        "desc": "Ensemble (all engines) + preprocessing + TTA — expected best",
    },
}


# ─── Core eval ──────────────────────────────────────────────────────────────

def eval_config(samples: list, config_name: str, config: dict) -> dict:
    """Run one config over all samples, return metrics dict."""
    try:
        import cv2
    except Exception as e:
        return {"error": f"opencv unavailable: {e}", "config": config_name}

    anpr = ANPRModule(
        ocr_engine=config["engine"],
        preprocess=config["preprocess"],
        upscale=config["upscale"],
        tta=config["tta"],
        tight_crop=config["tight_crop"],
    )
    anpr._ensure_engine()
    if anpr.engine_name is None and not anpr._loaded:
        return {"error": f"engine {config['engine']} not available",
                "config": config_name, "model_unavailable": True,
                "samples_available": len(samples)}

    log(f"[anpr] config={config_name} engine={anpr.engine_name or config['engine']}")

    exact = exact_raw = 0
    ned_sum = 0.0
    cer_num = cer_den = 0
    fmt_valid = 0
    n = 0
    examples = []

    for img_path, box, gt in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue

        rec = anpr.recognize(crop)
        pred = norm(rec.get("text", ""))
        pred_raw = norm(rec.get("raw_text", ""))
        g = norm(gt)
        d = levenshtein(pred, g)
        if pred == g:
            exact += 1
        if pred_raw == g:
            exact_raw += 1
        ned_sum += d / max(len(g), 1)
        cer_num += d
        cer_den += len(g)
        if validate_indian(pred)["format_valid"]:
            fmt_valid += 1
        n += 1
        if len(examples) < 10:
            examples.append({"gt": g, "raw": pred_raw, "pred": pred, "edit": d})
        if n % 100 == 0:
            log(f"   ...{n}/{len(samples)}")

    if n == 0:
        return {"error": "no readable crops", "config": config_name}

    return {
        "config": config_name,
        "desc": config["desc"],
        "engine": anpr.engine_name or config["engine"],
        "preprocess": config["preprocess"],
        "upscale": config["upscale"],
        "tta": config["tta"],
        "tight_crop": config["tight_crop"],
        "samples": n,
        "plate_exact_match_acc": round(exact / n, 4),
        "plate_exact_match_acc_raw": round(exact_raw / n, 4),
        "CER": round(cer_num / max(cer_den, 1), 4),
        "1-NED": round(1.0 - ned_sum / n, 4),
        "indian_format_valid_rate": round(fmt_valid / n, 4),
        "examples": examples,
    }


def run_ablation(samples: list, configs_to_run: list[str] | None = None) -> dict:
    """Run multiple configs, return all results keyed by config name."""
    keys = configs_to_run or list(ABLATION_CONFIGS.keys())
    results = {}
    for k in keys:
        cfg = ABLATION_CONFIGS[k]
        log(f"\n[anpr] ═══ Config: {k} ═══")
        log(f"[anpr]   {cfg['desc']}")
        r = eval_config(samples, k, cfg)
        results[k] = r
        if "error" not in r:
            log(f"[anpr]   exact={r['plate_exact_match_acc']}  CER={r['CER']}  "
                f"1-NED={r['1-NED']}")
        else:
            log(f"[anpr]   SKIPPED: {r['error']}")
    return results


# ─── Reporting ──────────────────────────────────────────────────────────────

def write_ablation_report(results: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_anpr_ablation_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# ANPR Ablation Study (run {run_id})\n",
         f"- Dataset: Indian-LP sirishan (GT plate box + text)",
         f"- Configs tested: {len(results)}\n",
         "## Results Summary\n",
         "| Config | Exact Acc | CER | 1-NED | Fmt Valid | Engine | PP | ×up | TTA | TightCrop |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    # Sort by exact match acc
    sorted_keys = sorted(
        [k for k, v in results.items() if "error" not in v],
        key=lambda k: results[k]["plate_exact_match_acc"], reverse=True
    )
    failed = [k for k, v in results.items() if "error" in v]
    baseline_acc = results.get("baseline", {}).get("plate_exact_match_acc", 0.449)

    for k in sorted_keys:
        v = results[k]
        delta = v["plate_exact_match_acc"] - baseline_acc
        delta_str = f"(+{delta:.3f})" if delta >= 0 else f"({delta:.3f})"
        best_tag = " ⭐BEST" if k == sorted_keys[0] else ""
        L.append(
            f"| **{k}**{best_tag} | **{v['plate_exact_match_acc']}** {delta_str} "
            f"| {v['CER']} | {v['1-NED']} | {v['indian_format_valid_rate']} "
            f"| {v['engine']} | {'✓' if v['preprocess'] else '✗'} "
            f"| {v['upscale']}× | {'✓' if v['tta'] else '✗'} "
            f"| {'✓' if v['tight_crop'] else '✗'} |"
        )

    if failed:
        L.append(f"\n**Skipped (engine unavailable):** {', '.join(failed)}")

    # Best config examples
    if sorted_keys:
        best_k = sorted_keys[0]
        best_v = results[best_k]
        L.append(f"\n## Best config: `{best_k}` — exact acc {best_v['plate_exact_match_acc']}")
        L.append(f"\n> {best_v['desc']}\n")
        L.append("### Examples (gt · raw → corrected)\n")
        for e in best_v.get("examples", []):
            L.append(f"- `{e['gt']}`  ·  `{e.get('raw', '')}` → `{e['pred']}`  (edit {e['edit']})")

    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def write_single_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_anpr_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# ANPR eval — {result.get('config', 'unknown')} (run {run_id})\n"]
    if "error" in result:
        L.append(f"**{'NOT TESTABLE' if result.get('model_unavailable') else 'ERROR'}:** {result['error']}")
        rp.write_text("\n".join(L), encoding="utf-8")
        return rp
    L.append(f"- Dataset: Indian-LP sirishan  ·  Config: {result['config']}  ·  N={result['samples']}")
    L.append(f"- Engine: {result['engine']}  ·  Preprocess: {result['preprocess']} ×{result['upscale']}")
    L.append(f"- TTA: {result['tta']}  ·  TightCrop: {result['tight_crop']}")
    L.append(f"\n> {result.get('desc', '')}\n")
    L.append("## Metrics\n")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Exact acc (raw) | {result['plate_exact_match_acc_raw']} |")
    L.append(f"| **Exact acc (corrected)** | **{result['plate_exact_match_acc']}** |")
    L.append(f"| CER | {result['CER']} |")
    L.append(f"| 1-NED | {result['1-NED']} |")
    L.append(f"| Format-valid rate | {result['indian_format_valid_rate']} |")
    L.append("\n### Examples (gt · raw → corrected)\n")
    for e in result.get("examples", []):
        L.append(f"- `{e['gt']}`  ·  `{e.get('raw', '')}` → `{e['pred']}`  (edit {e['edit']})")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="ANPR eval — single config or full ablation")
    ap.add_argument("--mode", choices=["single", "ablate"], default="single",
                    help="'single' evaluates one config; 'ablate' runs all configs + comparison table")
    ap.add_argument("--config", default="preprocess_only",
                    choices=list(ABLATION_CONFIGS.keys()),
                    help="Config to use in --mode single (default: preprocess_only)")
    ap.add_argument("--engine", default=None,
                    help="Override engine for --mode single (e.g. paddleocr, ensemble, trocr_large)")
    ap.add_argument("--limit", type=int, default=500,
                    help="Max samples (1741 available; 1000 recommended for ablation)")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Subset of configs to run in ablate mode (default: all)")
    args = ap.parse_args()

    run_id = new_run_id()
    samples = load_indian(args.limit)
    if not samples:
        log(f"[eval_anpr] ERROR: no Indian-LP samples found under {INDIAN_ROOT}")
        return 1
    log(f"[eval_anpr] loaded {len(samples)} samples  mode={args.mode}")

    if args.mode == "ablate":
        results = run_ablation(samples, configs_to_run=args.configs)
        rp = write_ablation_report(results, run_id)
        write_run_log("phase5", "anpr_ablation", run_id,
                      {"configs": list(results.keys()),
                       "best": max((r for r in results.values() if "error" not in r),
                                   key=lambda r: r["plate_exact_match_acc"],
                                   default={})})
        # Log best result to run_history
        good = [r for r in results.values() if "error" not in r]
        if good:
            best = max(good, key=lambda r: r["plate_exact_match_acc"])
            append_run_history({
                "run_id": run_id, "phase": "phase5", "module": "anpr",
                "dataset": "Indian-LP", "model": best["engine"],
                "metric": "plate_exact_acc",
                "value": best["plate_exact_match_acc"],
                "target": "vs SOTA", "pass_fail": "finetuned",
                "note": f"ablation BEST={best['config']} CER={best['CER']} N={best['samples']}",
            })
        log(f"\n[eval_anpr] Ablation complete → {rp.name}")
        # Print summary table to stdout
        if good:
            log("\n" + "─" * 70)
            log(f"{'Config':<25} {'Exact':>8} {'CER':>8} {'1-NED':>8}")
            log("─" * 70)
            for r in sorted(good, key=lambda r: r["plate_exact_match_acc"], reverse=True):
                log(f"{r['config']:<25} {r['plate_exact_match_acc']:>8.4f} "
                    f"{r['CER']:>8.4f} {r['1-NED']:>8.4f}")
            log("─" * 70)

    else:  # single
        cfg = dict(ABLATION_CONFIGS[args.config])
        if args.engine:
            cfg["engine"] = args.engine
        result = eval_config(samples, args.config, cfg)
        rp = write_single_report(result, run_id)
        if "error" in result:
            log(f"[eval_anpr] {result['error']}")
            return 1
        log(f"[eval_anpr] exact={result['plate_exact_match_acc']} "
            f"CER={result['CER']} 1-NED={result['1-NED']} N={result['samples']}")
        write_run_log("phase5", "anpr", run_id, result)
        append_run_history({
            "run_id": run_id, "phase": "phase5", "module": "anpr",
            "dataset": "Indian-LP", "model": result["engine"],
            "metric": "plate_exact_acc",
            "value": result["plate_exact_match_acc"],
            "target": "vs SOTA", "pass_fail": "finetuned",
            "note": f"config={args.config} CER={result['CER']} N={result['samples']}",
        })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
