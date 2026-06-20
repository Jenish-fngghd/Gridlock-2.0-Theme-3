"""Phase 3 — ANPR evaluation.

Two GT sources:
  * Indian-LP (sirishan) — VOC XML with the plate box AND the plate text as <name>
    (e.g. KA19TR02). Latin script → EasyOCR/TrOCR can read it. This is the design's actual
    target domain (§3.7), so it's the primary ANPR baseline.
  * CCPD — filename-encoded GT (Chinese plates) → needs a CN OCR; reported as available-but-
    needs-CN-engine if no Chinese OCR is present.

We isolate OCR quality by cropping the GT plate box (plate *detection* is a separate, fine-tune
step — zero-shot RF-DETR has no plate class, §3.7). Metrics per §7: plate exact-match accuracy,
CER (character error rate), 1-NED (normalized edit distance), + Indian-format validation rate.

Run:  python -m src.eval.eval_anpr --dataset indian --limit 300
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from src.modules.anpr import ANPRModule, validate_indian
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
    """Returns [(img_path, [x1,y1,x2,y2], plate_text), ...] from sirishan XML."""
    out = []
    for dp, _d, files in os.walk(INDIAN_ROOT):
        for f in files:
            if not f.endswith(".xml"):
                continue
            xml = os.path.join(dp, f)
            try:
                root = ET.parse(xml).getroot()
            except Exception:
                continue
            # image path: xml name minus .xml, try common ext variants in same dir
            stem = f[:-4]
            img = None
            for cand in (stem, stem + ".jpeg", stem + ".jpg", stem + ".png"):
                p = os.path.join(dp, cand)
                if os.path.exists(p):
                    img = p
                    break
            if img is None:
                # filename field
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


def run_indian(limit: int, ocr_engine: str = "auto") -> dict:
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        return {"error": f"opencv unavailable: {e}"}
    samples = load_indian(limit)
    if not samples:
        return {"error": f"no Indian-LP XML/plate-text GT found under {INDIAN_ROOT}"}
    log(f"[eval_anpr] {len(samples)} Indian-LP plates with GT text")

    anpr = ANPRModule(ocr_engine=ocr_engine)
    anpr._ensure_engine()
    if anpr.engine_name is None:
        return {"error": "no OCR engine installed (easyocr/paddleocr/trocr). ANPR not_testable.",
                "samples_available": len(samples), "model_unavailable": True}
    log(f"[eval_anpr] OCR engine: {anpr.engine_name}")

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
        pred = norm(rec.get("text", ""))          # corrected + multi-line-merged
        pred_raw = norm(rec.get("raw_text", ""))  # before syntax correction
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
        return {"error": "no readable crops"}
    return {
        "dataset": "Indian-LP sirishan (GT plate box+text)", "engine": anpr.engine_name,
        "samples": n,
        "plate_exact_match_acc": round(exact / n, 4),
        "plate_exact_match_acc_raw": round(exact_raw / n, 4),
        "CER": round(cer_num / max(cer_den, 1), 4),
        "1-NED": round(1.0 - ned_sum / n, 4),
        "indian_format_valid_rate": round(fmt_valid / n, 4),
        "improvements": "multi-line merge + syntax-aware correction (no fine-tune)",
        "examples": examples,
        "note": ("OCR scored on GT plate crops (isolates OCR from plate-detection, which is a "
                 "separate fine-tune step — zero-shot has no plate class). PaddleOCR (PP-OCRv5/v6, the "
                 "design's primary) runs in a Python-3.12 venv (.venv-paddle) since paddle has no "
                 "Py3.14 wheel; EasyOCR is the named backup. oneDNN disabled (FLAGS_use_mkldnn=0) to "
                 "avoid a paddle-3.x CPU PIR bug."),
    }


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_anpr_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# ANPR eval (run {run_id})\n"]
    if "error" in result:
        L.append(f"**{'NOT TESTABLE' if result.get('model_unavailable') else 'ERROR'}:** {result['error']}")
        if "samples_available" in result:
            L.append(f"\n- GT samples ready: {result['samples_available']} (install an OCR engine to score)")
        rp.write_text("\n".join(L), encoding="utf-8")
        return rp
    L.append(f"- Dataset: {result['dataset']}  ·  OCR: {result['engine']}  ·  N={result['samples']}")
    L.append(f"- Post-processing: {result.get('improvements','—')}\n")
    L.append("## Quantitative (OCR on GT crops)\n")
    L.append("| Metric | Value | §7 reference |")
    L.append("|---|---|---|")
    L.append(f"| Plate exact-match acc (raw) | {result.get('plate_exact_match_acc_raw','—')} | before post-proc |")
    L.append(f"| **Plate exact-match acc (final)** | **{result['plate_exact_match_acc']}** | report vs CCPD/Indian SOTA |")
    L.append(f"| CER | {result['CER']} | lower better |")
    L.append(f"| 1-NED | {result['1-NED']} | higher better |")
    L.append(f"| Indian-format valid rate | {result['indian_format_valid_rate']} | — |")
    L.append("\n### Examples (gt · raw → corrected)\n")
    for e in result["examples"]:
        raw = e.get("raw", "")
        L.append(f"- `{e['gt']}`  ·  `{raw}` → `{e['pred']}`  (edit {e['edit']})")
    L.append(f"\n> {result['note']}")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["indian"], default="indian")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--ocr", default="auto", choices=["auto", "paddleocr", "easyocr", "trocr"],
                    help="force a specific OCR engine (e.g. paddleocr for PP-OCRv5)")
    args = ap.parse_args()
    run_id = new_run_id()
    result = run_indian(args.limit, ocr_engine=args.ocr)
    write_run_log("phase3", "anpr", run_id, result)
    rp = write_report(result, run_id)
    if "error" in result:
        log(f"[eval_anpr] {result['error']}")
        append_run_history({"run_id": run_id, "phase": "phase3", "module": "anpr",
                            "dataset": "Indian-LP", "model": "ocr", "metric": "plate_acc",
                            "value": "not_testable", "target": "-", "pass_fail": "blocked",
                            "note": result["error"][:60]})
        return 1
    log(f"[eval_anpr] plate_acc={result['plate_exact_match_acc']} CER={result['CER']} "
        f"1-NED={result['1-NED']} (report: {rp.name})")
    append_run_history({"run_id": run_id, "phase": "phase3", "module": "anpr",
                        "dataset": "Indian-LP", "model": result["engine"], "metric": "plate_exact_acc",
                        "value": result["plate_exact_match_acc"], "target": "vs SOTA",
                        "pass_fail": "baseline", "note": f"CER={result['CER']} N={result['samples']}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
