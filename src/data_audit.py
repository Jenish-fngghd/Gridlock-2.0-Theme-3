"""Phase 0b/0c — Dataset integrity & violation-coverage audit.

Inspects the ACTUAL on-disk structure of every dataset under ./datasets/, confirms
whether the format matches what each loader will expect, and reports a status of
ready / partial / not_ready. Then cross-references the 11 datasets against the 7
mandated violation types (+ detection + ANPR) for Phase 0c coverage.

Stdlib-only by design (cv2 / lxml are not installed on this machine — see Phase 0a),
so this runs regardless of the ML stack state.

Run:  python -m src.data_audit
Outputs:
  logs/dataset_audit_report.md
  logs/phase0/data_audit_<run_id>.json
  one line appended to results/run_history.csv
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)

DATASETS = REPO_ROOT / "datasets"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Resolved on-disk roots (folder names contain spaces — keep them exact).
ROOTS = {
    "IDD": DATASETS / "idd-detection" / "IDD_Detection",
    "BDD100K": DATASETS / "BDD100K",
    "CCPD": DATASETS / "CCPD",
    "LISA": DATASETS / "Red Light" / "LISA Traffic Light Dataset",
    "IndianLP": DATASETS / "Indian LP",
    "UA-DETRAC": DATASETS / "DETRAC",
    "AICC": DATASETS / "Helmet & Triple Riding",
    "Seatbelt": DATASETS / "seat belt detection" / "seat_belt and mobile.v2i.yolov8-obb",
    "ISLab-PVD": DATASETS / "Illegal Parking" / "IS_labPVD",
    "RunningRedlight": DATASETS / "Red Light" / "namnv78_RunningRedlight",
    "WrongWay": DATASETS / "Wrong Side Driving" / "Wrong Way Driving Detection.v1i.yolov8-obb",
}


# ----------------------------- fast filesystem helpers -----------------------------
def walk_count(root: Path, exts: set[str], cap: int | None = None) -> tuple[int, list[str]]:
    """Recursively count files whose suffix is in `exts`. Returns (count, up-to-5 samples)."""
    n = 0
    samples: list[str] = []
    if not root.exists():
        return 0, []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in exts:
                n += 1
                if len(samples) < 5:
                    samples.append(f)
                if cap and n >= cap:
                    return n, samples
    return n, samples


def count_by_ext(root: Path, top_n: int = 8) -> dict[str, int]:
    hist: dict[str, int] = {}
    if not root.exists():
        return hist
    for _dp, _dirs, files in os.walk(root):
        for f in files:
            ext = Path(f).suffix.lower() or "<none>"
            hist[ext] = hist.get(ext, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1])[:top_n])


# ----------------------------- per-dataset audits -----------------------------
def audit_idd(root: Path) -> dict:
    jpg, _ = walk_count(root / "JPEGImages", IMG_EXTS)
    xml, _ = walk_count(root / "Annotations", {".xml"})
    imagesets = {s: (root / f"{s}.txt").exists() for s in ("train", "val", "test")}
    sample_ok = None
    # validate one VOC XML parses
    for dp, _d, files in os.walk(root / "Annotations"):
        x = [f for f in files if f.endswith(".xml")]
        if x:
            try:
                ET.parse(os.path.join(dp, x[0])).getroot().find("object")
                sample_ok = True
            except Exception as e:  # noqa: BLE001
                sample_ok = f"parse_error: {e}"
            break
    status = "ready" if (jpg > 0 and xml > 0) else "not_ready"
    return {"status": status, "images": jpg, "annotations_xml": xml,
            "imagesets": imagesets, "voc_xml_parses": sample_ok,
            "note": "VOC XML; JPEGImages/+Annotations/ pairing; ImageSets present"}


def audit_bdd(root: Path) -> dict:
    hist = count_by_ext(root)
    imgs, _ = walk_count(root, IMG_EXTS, cap=1)
    only_checksums = (imgs == 0) and (hist.get(".md5", 0) > 0 or all(
        k in {".md5", "<none>"} for k in hist))
    status = "not_ready" if (imgs == 0) else "ready"
    return {"status": status, "ext_histogram": hist, "images_found": imgs,
            "note": ("⚠️ Only .md5 checksum files present — no images/labels. SKIP this run, "
                     "no placeholder." if only_checksums else "images present")}


def decode_ccpd(name: str) -> dict | None:
    """CCPD filename: area-tilt-bbox-vertices-LPindices-brightness-blur. 7 dash fields."""
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    try:
        lp_idx = [int(x) for x in parts[4].split("_")]
        return {"fields": len(parts), "lp_index_count": len(lp_idx), "lp_indices": lp_idx}
    except Exception:
        return None


def audit_ccpd(root: Path) -> dict:
    green = root / "CCPD2020" / "CCPD2020" / "ccpd_green"
    splits = {}
    total = 0
    decoded = None
    for sp in ("train", "val", "test"):
        c, samples = walk_count(green / sp, IMG_EXTS)
        splits[sp] = c
        total += c
        if decoded is None and samples:
            decoded = decode_ccpd(samples[0])
            decoded_sample = samples[0]
    status = "ready" if (total > 0 and decoded) else ("partial" if total > 0 else "not_ready")
    return {"status": status, "edition": "CCPD2020 (green/new-energy plates)",
            "split_counts": splits, "total_images": total,
            "sample_filename": locals().get("decoded_sample"),
            "filename_decode": decoded,
            "note": "GT encoded in filename (field 5 = LP char indices). Not a renamed mirror."}


def audit_lisa(root: Path) -> dict:
    # find frameAnnotationsBOX.csv files
    box_csvs = []
    for dp, _d, files in os.walk(root):
        for f in files:
            if f == "frameAnnotationsBOX.csv":
                box_csvs.append(os.path.relpath(os.path.join(dp, f), root))
    frames, _ = walk_count(root, IMG_EXTS)
    status = "ready" if (box_csvs and frames > 0) else ("partial" if frames > 0 else "not_ready")
    return {"status": status, "frames": frames, "frameAnnotationsBOX_csv_count": len(box_csvs),
            "csv_samples": box_csvs[:4],
            "note": "frame-level signal-state GT in frameAnnotationsBOX.csv (7 states)"}


def audit_indian_lp(root: Path) -> dict:
    sirishan = root / "sirishan"
    dcl = root / "data cluster labs"
    sir_imgs, _ = walk_count(sirishan, IMG_EXTS)
    dcl_imgs, _ = walk_count(dcl, IMG_EXTS)
    # sidecar labels (yolo .txt / json / xml) anywhere
    txt, _ = walk_count(root, {".txt"})
    js, _ = walk_count(root, {".json"})
    xml, _ = walk_count(root, {".xml"})
    total = sir_imgs + dcl_imgs
    status = "ready" if total > 0 else "not_ready"
    note = (f"sirishan={sir_imgs} img, data-cluster-labs={dcl_imgs} img. "
            f"Documented sirishan total ~16,192 img/21,683 plates — real on-disk count reported above "
            f"(extraction may have skipped files). Sidecar labels: txt={txt}, json={js}, xml={xml}.")
    return {"status": status, "sirishan_images": sir_imgs, "data_cluster_labs_images": dcl_imgs,
            "total_images": total, "label_files": {"txt": txt, "json": js, "xml": xml}, "note": note}


def audit_detrac(root: Path) -> dict:
    imgs, _ = walk_count(root / "DETRAC-Images", IMG_EXTS)
    annos, _ = walk_count(root / "DETRAC-Annos", {".xml"})
    status = "ready" if (imgs > 0 and annos > 0) else ("partial" if imgs > 0 else "not_ready")
    return {"status": status, "images": imgs, "annotation_xml": annos,
            "note": "per-sequence MVI_* image folders + XML annos (optional detection backbone)"}


def audit_aicc(root: Path) -> dict:
    # imagery?
    imgs, _ = walk_count(root, IMG_EXTS, cap=1)
    vids, _ = walk_count(root, VID_EXTS, cap=1)
    # annotation CSVs + class label range
    labels = set()
    csv_found = []
    for dp, _d, files in os.walk(root):
        for f in files:
            if f in ("trainset_ai.csv", "trainset_head.csv"):
                p = os.path.join(dp, f)
                csv_found.append(os.path.relpath(p, root))
                if f == "trainset_ai.csv":
                    try:
                        with open(p, encoding="utf-8") as fh:
                            header = fh.readline().strip().split(",")
                            li = header.index("label") if "label" in header else -1
                            for line in fh:
                                cols = line.strip().split(",")
                                if li >= 0 and li < len(cols):
                                    try:
                                        labels.add(int(float(cols[li])))
                                    except Exception:
                                        pass
                    except Exception:
                        pass
    max_lbl = max(labels) if labels else None
    edition = ("unknown" if max_lbl is None else
               ("≤7-class style (2023 edition?)" if max_lbl <= 7 else "9-class (2024 edition?)"))
    # The on-disk content is the winning-solution code repo, not the Track-5 imagery.
    status = "not_ready" if (imgs == 0 and vids == 0) else "partial"
    return {"status": status, "images": imgs, "videos": vids,
            "annotation_csvs": csv_found, "label_values": sorted(labels), "edition_guess": edition,
            "note": ("⚠️ On disk = AI-City-Challenge-2023 winning-solution CODE repo + annotation CSVs "
                     "ONLY. No Track-5 videos/frames present, so helmet+triple is NOT trainable/"
                     "evaluable from this. Mark helmet module not_testable until raw AICC data is obtained.")}


def _read_yaml_names(yaml_path: Path) -> dict:
    """Tiny YAML 'names:' parser (avoids a pyyaml dep)."""
    names = {}
    if not yaml_path.exists():
        return names
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            s = line.strip()
            if not s or ":" not in s or not (s[0].isdigit()):
                if s and not s[0].isspace() and not s[0].isdigit():
                    break
                if not s:
                    continue
            if ":" in s and s[0].isdigit():
                k, v = s.split(":", 1)
                names[int(k)] = v.strip()
    return names


def audit_obb(root: Path, expect_names: dict) -> dict:
    data_yaml = root / "data.yaml"
    names = _read_yaml_names(data_yaml)
    splits = {}
    label_token_counts = set()
    for sp in ("train", "valid", "test"):
        imgs, _ = walk_count(root / sp / "images", IMG_EXTS)
        lbls, lbl_samples = walk_count(root / sp / "labels", {".txt"})
        splits[sp] = {"images": imgs, "labels": lbls}
        # inspect one label file to confirm OBB (9 tokens: class + 8 coords)
        lbl_dir = root / sp / "labels"
        if lbl_dir.exists():
            for lf in lbl_dir.glob("*.txt"):
                try:
                    first = lf.read_text(encoding="utf-8").strip().splitlines()
                    if first:
                        label_token_counts.add(len(first[0].split()))
                except Exception:
                    pass
                break
    is_obb = 9 in label_token_counts
    status = "ready" if any(s["images"] > 0 for s in splits.values()) else "not_ready"
    return {"status": status, "names": names, "splits": splits,
            "label_token_counts": sorted(label_token_counts),
            "format": "YOLOv8-OBB (class + 4 corner xy)" if is_obb else f"tokens={sorted(label_token_counts)}",
            "note": "OBB polygons -> convert to axis-aligned via src/utils/obb_convert.py"}


def audit_islab(root: Path) -> dict:
    vids, vsamples = walk_count(root, VID_EXTS)
    # any GT files?
    gt_exts = {".xml", ".json", ".txt", ".csv", ".srt"}
    gt, gsamples = walk_count(root, gt_exts)
    status = "partial" if (vids > 0 and gt == 0) else ("ready" if (vids > 0 and gt > 0) else "not_ready")
    return {"status": status, "videos": vids, "video_samples": vsamples,
            "ground_truth_files": gt, "gt_samples": gsamples,
            "note": ("CCTV-style .mp4 videos present but NO machine-readable GT on disk -> event-level "
                     "precision/recall NOT computable without manual annotation. Usable for qualitative "
                     "demo only this run; build eval_illegal_parking.py event-level once GT exists.")}


def audit_running_redlight(root: Path) -> dict:
    labels_dir = root / "combined_data_v2" / "processed_labels"
    jsons, jsamples = walk_count(labels_dir, {".json"})
    # frame-sequence folders ("*.avi_save") + image frames
    frames, _ = walk_count(root, IMG_EXTS)
    cross_true = cross_false = 0
    schema_ok = None
    if labels_dir.exists():
        for jf in list(labels_dir.glob("*.json"))[:200]:
            try:
                obj = json.loads(jf.read_text(encoding="utf-8"))
                meta = obj.get("meta", {})
                if "cross" in meta:
                    schema_ok = True
                    if meta["cross"]:
                        cross_true += 1
                    else:
                        cross_false += 1
            except Exception:
                pass
    status = "ready" if (jsons > 0 and schema_ok) else ("partial" if jsons > 0 else "not_ready")
    return {"status": status, "label_jsons": jsons, "frame_images": frames,
            "clip_label_field": "meta.cross (bool) = ran-red-light",
            "cross_sample_counts(first200)": {"true": cross_true, "false": cross_false},
            "note": ("Arrived as frame-sequence folders (*.avi_save/) + per-clip JSON labels. Clip-level "
                     "binary classification. Cross-check vs rule engine; do NOT auto-merge (J3).")}


AUDITS = {
    "IDD": lambda r: audit_idd(r),
    "BDD100K": lambda r: audit_bdd(r),
    "CCPD": lambda r: audit_ccpd(r),
    "LISA": lambda r: audit_lisa(r),
    "IndianLP": lambda r: audit_indian_lp(r),
    "UA-DETRAC": lambda r: audit_detrac(r),
    "AICC": lambda r: audit_aicc(r),
    "Seatbelt": lambda r: audit_obb(r, {0: "mobile", 1: "seatbelt", 2: "windshield"}),
    "ISLab-PVD": lambda r: audit_islab(r),
    "RunningRedlight": lambda r: audit_running_redlight(r),
    "WrongWay": lambda r: audit_obb(r, {0: "right-side", 1: "wrong-side"}),
}

# Phase 0c — violation coverage map. status one of: quantitative / quantitative_eventlevel /
# quantitative_cliplevel / qualitative / not_ready
COVERAGE = [
    ("Detection backbone", "IDD / UA-DETRAC (BDD100K NOT READY)", "quantitative"),
    ("ANPR", "CCPD2020 / Indian-LP", "quantitative"),
    ("Helmet", "AICC Track 5", "blocked: imagery absent (not_testable)"),
    ("Triple riding", "AICC Track 5 (same GT)", "blocked: imagery absent (not_testable)"),
    ("Seatbelt", "Seatbelt+Mobile (OBB)", "quantitative"),
    ("Illegal parking", "ISLab-PVD", "blocked: videos present, GT absent -> qualitative only"),
    ("Red-light (signal state)", "LISA", "quantitative (frame-level)"),
    ("Red-light (full event)", "RunningRedlight", "quantitative (clip-level cross-check)"),
    ("Wrong-side driving", "Wrong-Way (OBB)", "quantitative (frame-level)"),
    ("Stop-line", "— none —", "qualitative spot-check only (no dataset anywhere)"),
]


def build_report(results: dict, tier_note: str) -> str:
    L = ["# Dataset Audit Report (Phase 0b / 0c)\n",
         f"_Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}_\n",
         f"> Hardware context: {tier_note}\n",
         "## 0b — Per-dataset integrity\n",
         "| # | Dataset | Status | Key counts | Notes |",
         "|---|---|---|---|---|"]
    order = ["IDD", "BDD100K", "CCPD", "LISA", "IndianLP", "UA-DETRAC", "AICC",
             "Seatbelt", "ISLab-PVD", "RunningRedlight", "WrongWay"]
    badge = {"ready": "✅ ready", "partial": "🟨 partial", "not_ready": "❌ not_ready"}
    for i, key in enumerate(order, 1):
        r = results[key]
        st = badge.get(r.get("status"), r.get("status", "?"))
        counts = []
        for ck in ("images", "total_images", "frames", "frame_images", "videos",
                   "annotations_xml", "annotation_xml", "label_jsons"):
            if ck in r:
                counts.append(f"{ck}={r[ck]}")
        if "split_counts" in r:
            counts.append("splits=" + str(r["split_counts"]))
        if "splits" in r:
            counts.append("splits=" + json.dumps({k: v["images"] for k, v in r["splits"].items()}))
        note = str(r.get("note", "")).replace("\n", " ")
        L.append(f"| {i} | {key} | {st} | {'; '.join(counts) or '—'} | {note} |")
    L.append("\n### Selected details\n")
    L.append("```json")
    L.append(json.dumps(results, indent=2, ensure_ascii=False))
    L.append("```")
    L.append("\n## 0c — Violation coverage map\n")
    L.append("| Violation / capability | Dataset | Eval status |")
    L.append("|---|---|---|")
    for v, ds, stt in COVERAGE:
        L.append(f"| {v} | {ds} | {stt} |")
    L.append("\n**Quantitative & ready now:** Detection (IDD/UA-DETRAC), ANPR (CCPD/Indian-LP), "
             "Seatbelt, Red-light signal-state (LISA), Red-light full-event (RunningRedlight), "
             "Wrong-side (Wrong-Way).")
    L.append("\n**Blocked / not quantitative this run:**")
    L.append("- **Helmet & Triple riding** — AICC imagery absent (only code repo + annotation CSVs). "
             "`not_testable` until the real Track-5 data is downloaded.")
    L.append("- **Illegal parking** — ISLab-PVD has videos but no GT on disk → qualitative only until annotated.")
    L.append("- **BDD100K** — only `.md5` checksums → NOT READY, skipped, no placeholder.")
    L.append("- **Stop-line** — no dataset exists anywhere → qualitative spot-check only (never a metric).")
    L.append("")
    return "\n".join(L)


def main() -> int:
    run_id = new_run_id()
    log("[Phase 0b] Auditing datasets on disk...")
    results = {}
    for key, fn in AUDITS.items():
        root = ROOTS[key]
        try:
            r = fn(root) if root.exists() else {"status": "not_ready", "note": f"path missing: {root}"}
        except Exception as e:  # noqa: BLE001
            r = {"status": "error", "note": f"{type(e).__name__}: {e}"}
        r["path"] = str(root)
        r["path_exists"] = root.exists()
        results[key] = r
        log(f"   {key:16} -> {r.get('status')}")

    tier_note = "RTX 3050 Laptop, 4GB VRAM -> tier=cloud_required (see logs/environment_report.md)"
    report = build_report(results, tier_note)
    out = REPO_ROOT / "logs" / "dataset_audit_report.md"
    out.write_text(report, encoding="utf-8")

    write_run_log("phase0", "data_audit", run_id, {"results": results, "coverage": COVERAGE})
    ready = sum(1 for r in results.values() if r.get("status") == "ready")
    append_run_history({"run_id": run_id, "phase": "phase0", "module": "data_audit",
                        "dataset": "ALL(11)", "model": "-", "metric": "datasets_ready",
                        "value": ready, "target": 11, "pass_fail": "info",
                        "note": "see logs/dataset_audit_report.md"})
    log(f"[Phase 0b] {ready}/11 datasets fully ready. Report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
