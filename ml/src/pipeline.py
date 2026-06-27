"""Pipeline orchestrator — wires the modules per 00_master_design.md §2.

Flow:  ingest+QA -> preprocess(gated) -> detect -> [helmet/triple (SAM-3 helmet) ·
       seatbelt (yolo11n+mobilenet) · wrong-side (mobilenet) · signal (HSV) · geometry] ->
       confidence cascade (VLM verify on human_review) -> ANPR (SAM-3 plate + TrOCR-ft,
       violators only) -> evidence.

Single still vs sequence: temporal violations (red-light/stop-line/illegal-parking) need
motion; on a single image the geometry engine abstains (logged) and red-light's LSTM event
classifier is skipped (needs a tracked trajectory — see `process_clip` for video). Every module
degrades gracefully — a missing model is logged (`model_unavailable`) and the run continues.
SAM-3 (helmet-state crops, plate localization) and the VLM verifier are both OPTIONAL and
lazily/shared-loaded — disable with --no-sam3 / --no-vlm to run on the cheap models only.

CLI:
  python -m src.pipeline --mode demo --input <img-or-folder> --output outputs/
  python -m src.pipeline --write-sample-config configs/camera_config.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.modules.anpr import ANPRModule
from src.modules.confidence_cascade import ConfidenceCascade
from src.modules.detection import Detection, VehicleDetector
from src.modules.evidence import EvidenceGenerator
from src.modules.geometry_engine import GeometryEngine, SceneConfig
from src.modules.helmet_triple import HelmetTripleModule
from src.modules.preprocessing import Preprocessor
from src.modules.quality_gate import QualityGate
from src.modules.seatbelt import SeatbeltModule
from src.modules.signal_state import SignalStateClassifier
from src.modules.tracking import IoUTracker
from src.modules.vlm_verify import VLMVerifier
from src.modules.wrongside import WrongSideModule
from src.utils.logging import REPO_ROOT, log, new_run_id

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "autorickshaw"}
FOUR_WHEELERS = {"car", "truck", "bus"}  # seatbelt only applies to these (not two-wheelers)

SAMPLE_CAMERA_CONFIG = {
    "camera_id": "CAM_DEMO_01",
    "stop_line": [[200, 700], [1100, 700]],
    "no_parking_polygon": [[50, 600], [400, 600], [400, 900], [50, 900]],
    "lane_direction": [0.0, -1.0],
    "signal_roi": [[1150, 40], [1230, 40], [1230, 200], [1150, 200]],
    "parking_dwell_frames": 30,
}


@dataclass
class PipelineConfig:
    output_dir: str = "outputs"
    variant: str = "nano"
    threshold: float = 0.3
    camera_config: str | None = None
    tier: str = "cloud_required"
    use_sam3: bool = True
    use_vlm: bool = True
    use_roboflow: bool = False  # reverted — scene-blind per-class FPs; VLM is the arbiter now


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.quality = QualityGate()
        self.pre = Preprocessor(enable=True)
        self.detector = VehicleDetector.for_tier(cfg.tier, threshold=cfg.threshold)
        self.tracker = IoUTracker()

        self.sam3 = None  # legacy local subprocess worker — unused on the hosted path

        # Roboflow-hosted SAM-3 — open-vocab entity detection for the classes our trained
        # single-image models can't do well: helmet, triple-riding, stop-line, red-light. The
        # rules live in sam3_violations.py (ported from the reference notebook). One HTTP call per
        # image. Gated by use_sam3 + ROBOFLOW_API_KEY presence; degrades gracefully if absent.
        self.sam3v = None
        if cfg.use_sam3:
            from src.modules.roboflow_sam3 import RoboflowSAM3
            from src.modules.sam3_violations import SAM3Violations
            rf = RoboflowSAM3()
            if rf.available():
                self.sam3v = SAM3Violations(rf)

        # Wrong-side: SAM-3 can't detect a direction/heading attribute, but Roboflow's dedicated
        # wrong-way model nails it (eval 2/2 + 0/7 FP). Used in place of the weak local classifier.
        self.rf_wrongside = None
        if cfg.use_sam3:  # same gate/key as the rest of the hosted-inference path
            from src.modules.roboflow_detect import RoboflowDetector
            rfd = RoboflowDetector()
            self.rf_wrongside = rfd if rfd.available() else None

        self.helmet_triple = HelmetTripleModule(detector=self.detector, sam3=None)
        self.seatbelt = SeatbeltModule()
        self.wrongside = WrongSideModule()
        self.signal = SignalStateClassifier()
        scene = SceneConfig.from_dict(self._load_camera_cfg(cfg.camera_config))
        self.geometry = GeometryEngine(scene)
        self.cascade = ConfidenceCascade()
        self.vlm = VLMVerifier() if cfg.use_vlm else None
        self.anpr = ANPRModule(ocr_engine="trocr_ft" if self._has_trocr_ft() else "auto")
        self.evidence = EvidenceGenerator(
            cfg.output_dir, camera_id=scene.camera_id,
            model_versions={"detector": f"rfdetr-{cfg.variant}", "ocr": self.anpr._pref,
                            "tracker": self.tracker.backend,
                            "sam3": "roboflow-hosted" if self.sam3v else "disabled",
                            "vlm": self.vlm.model if self.vlm and self.vlm.available() else "disabled"},
            scene_config=scene.camera_id)

    @staticmethod
    def _has_trocr_ft() -> bool:
        return (REPO_ROOT / "checkpoints" / "anpr" / "trocr_ft" / "model.safetensors").exists()

    @staticmethod
    def _load_camera_cfg(path: str | None) -> dict:
        if path and Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
        return SAMPLE_CAMERA_CONFIG

    def process_image(self, img_path: str) -> dict:
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            return {"image": img_path, "error": "unreadable"}
        return self.process_array(img, frame_ref=img_path)

    def process_array(self, img, frame_ref: str = "") -> dict:
        """Core analysis on an in-memory BGR numpy array — used by process_image() and directly
        by the inference service (no temp-file round trip needed for API uploads)."""
        q = self.quality.assess(img)
        pre = self.pre.restore(img, q)
        work = pre.image if pre.image is not None else img

        seatbelt = self.seatbelt.analyze(work)

        violations: list[dict] = []
        ht: dict = {}
        all_dets: list = []
        detector_unavailable = False

        if self.sam3v is not None and self.sam3v.available():
            # Primary path: one SAM-3 call covers both detection and all violation rules.
            # SAM-3 is open-vocab so it detects Indian-specific classes (autorickshaw) that
            # COCO-trained RF-DETR structurally misses. RF-DETR not called on this path.
            sam3_result = self.sam3v.analyze(work)
            violations.extend(sam3_result["violations"])
            for sd in sam3_result["detections"]:
                all_dets.append(Detection(
                    xyxy=tuple(sd["xyxy"]),
                    confidence=sd["confidence"],
                    class_id=-1,
                    class_name=sd["class_name"],
                ))
        else:
            # Fallback path: all SAM-3 keys exhausted — run local RF-DETR + rule-based detection.
            det = self.detector.detect(work)
            all_dets = det.detections if not det.model_unavailable else []
            detector_unavailable = det.model_unavailable
            ht = self.helmet_triple.analyze(detections=all_dets, frame=work)
            self._collect_triple(ht, violations)
            self._collect_helmet(ht, violations)

        # single still -> temporal/geometry abstains (no track history)
        geo_note = "abstained: single still has no motion history (needs sequence; see process_clip)"

        signal_obs = self._observe_signal(work, all_dets)
        self._collect_seatbelt(seatbelt, all_dets, violations)
        # wrong-side: dedicated Roboflow heading model (2/2+0/7) when available, else weak local.
        if self.rf_wrongside is not None:
            self._collect_wrongside_roboflow(work, violations)
        else:
            self._collect_wrongside(work, all_dets, violations)

        decisions = self.cascade.decide_many(violations)
        for v, d in zip(violations, decisions):
            v["band"] = d.band
            v["calibrated_confidence"] = d.calibrated_confidence
            v["needs_vlm"] = d.needs_vlm

        if self.vlm is not None:
            self._run_vlm_verification(work, violations)

        # ROI-gated ANPR: only on violators, plate localized via SAM-3 within the vehicle's own box
        plate_info = self._run_anpr(work, violations)

        evidence_id = None
        evidence_image_path = None
        if violations:
            record = self.evidence.generate(
                work, violations, vehicle=plate_info, frame_ref=frame_ref, vlm_caption="")
            evidence_id = record["violation_id"]
            evidence_image_path = record.get("evidence_image")
        return {
            "image": frame_ref,
            "quality": q.to_dict(),
            "preprocess": pre.to_dict(),
            "detections": len(all_dets),
            "detection_list": [{"class_name": d.class_name, "confidence": round(d.confidence, 4),
                                "xyxy": [round(v, 1) for v in d.xyxy]} for d in all_dets],
            "detector_unavailable": detector_unavailable,
            "helmet_status": ht.get("helmet_status", "not_testable"),
            "triple_riding_count": ht.get("triple_riding_count", 0),
            "seatbelt": seatbelt.get("windshields", []),
            "signal_state": signal_obs,
            "geometry": geo_note,
            "violations": violations,
            "plate": plate_info,
            "evidence_record": evidence_id,
            "evidence_image_path": evidence_image_path,
        }

    # ---- per-violation-type collectors ---------------------------------

    @staticmethod
    def _collect_triple(ht: dict, violations: list[dict]) -> None:
        for g in ht.get("groups", []):
            if g.get("triple_riding"):
                mb = g["motorbike_bbox"]
                violations.append({
                    "type": "triple_riding", "confidence": g["triple_confidence"],
                    "bbox": [mb[0], mb[1], mb[2] - mb[0], mb[3] - mb[1]],
                    "vehicle_bbox": mb,
                    "evidence_chain": ["detect", "associate_riders", "count>=3"],
                    "rider_count": g["rider_count"], "basis": "proxy (no GT label)"})

    @staticmethod
    def _collect_helmet(ht: dict, violations: list[dict]) -> None:
        for g in ht.get("groups", []):
            for rh in g.get("rider_helmets", []):
                if rh.get("has_helmet") is False:
                    bb = rh["bbox"]
                    conf_raw = rh.get("confidence", 0.0)
                    conf = max(0.5, min(0.85, 1.0 - conf_raw)) if conf_raw > 0 else 0.6
                    violations.append({
                        "type": "no_helmet", "confidence": round(conf, 3),
                        "bbox": [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]],
                        "vehicle_bbox": g["motorbike_bbox"],
                        "evidence_chain": ["detect", "associate_rider", "sam3:helmet_absent"],
                        "basis": "SAM-3 open-vocab (no AICC fine-tune checkpoint)"})

    @staticmethod
    def _collect_seatbelt(seatbelt: dict, all_dets, violations: list[dict]) -> None:
        for w in seatbelt.get("windshields", []):
            if not w.get("no_seatbelt"):
                continue
            wb = w["bbox"]
            # Seatbelt only applies to 4-wheelers: require the windshield to sit inside a
            # car/truck/bus detection. A windshield over a motorbike/scooter (or no vehicle) is a
            # false positive and is skipped.
            vehicle_bbox = Pipeline._containing_vehicle(wb, all_dets, classes=FOUR_WHEELERS)
            if vehicle_bbox is None:
                continue
            violations.append({
                "type": "no_seatbelt", "confidence": w["confidence"],
                "bbox": [wb[0], wb[1], wb[2] - wb[0], wb[3] - wb[1]],
                "vehicle_bbox": list(vehicle_bbox),
                "evidence_chain": ["detect_windshield", "classify_belt"],
                "basis": "yolo11n-windshield + mobilenetv3l-belt (fine-tuned)"})

    def _collect_wrongside_roboflow(self, work, violations: list[dict]) -> None:
        r = self.rf_wrongside.detect(work, "wrong_side")
        if r.get("model_unavailable") or not r.get("fired"):
            return
        for b in r.get("boxes", []):
            x1, y1, x2, y2 = [int(round(v)) for v in b["xyxy"]]
            # Re-evaluated on a larger, more diverse sample: this model auto-confirmed clear false
            # positives on multi-vehicle street scenes (busy-road FPs at 0.89/0.92 conf) despite a
            # clean 2/2+0/7 result on the original tiny hand-picked sample. Cap below auto_confirm
            # so a human always reviews before any wrong-side challan goes out.
            violations.append({
                "type": "wrong_side", "confidence": min(b["confidence"], 0.79),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "vehicle_bbox": [x1, y1, x2, y2],
                "evidence_chain": ["roboflow:wrong-way-driving-detection", "class:wrong-side"],
                "basis": "roboflow wrong-way-driving-detection/2 (dedicated heading model; review-only)"})

    def _collect_wrongside(self, work, all_dets, violations: list[dict]) -> None:
        for d in all_dets:
            if d.class_name not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in d.xyxy]
            crop = work[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            r = self.wrongside.classify(crop)
            if r.get("model_unavailable") or not r.get("wrong_side"):
                continue
            # Single-frame heading from appearance is unreliable (eval: weak recall + false
            # positives) -> cap to human_review (<0.80) so it never auto-challans; the VLM verifies.
            violations.append({
                "type": "wrong_side", "confidence": min(r["confidence"], 0.79),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "vehicle_bbox": [x1, y1, x2, y2],
                "evidence_chain": ["detect_vehicle", "classify_heading"],
                "basis": "mobilenetv3s-ft (single-frame, appearance-based; review-only)"})

    @staticmethod
    def _containing_vehicle(box, all_dets, classes=VEHICLE_CLASSES):
        bx1, by1, bx2, by2 = box
        best, best_area = None, 0.0
        for d in all_dets:
            if d.class_name not in classes:
                continue
            vx1, vy1, vx2, vy2 = d.xyxy
            ix1, iy1 = max(bx1, vx1), max(by1, vy1)
            ix2, iy2 = min(bx2, vx2), min(by2, vy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter > best_area:
                best, best_area = d.xyxy, inter
        return best

    def _observe_signal(self, work, all_dets) -> dict:
        lights = [d for d in all_dets if d.class_name == "traffic light"]
        if not lights:
            return {"state": "unknown", "note": "no traffic light detected"}
        best = max(lights, key=lambda d: d.confidence)
        return self.signal.classify(work, best.xyxy)

    # ---- confidence-cascade VLM escalation -------------------------------

    def _run_vlm_verification(self, work, violations: list[dict]) -> None:
        """Agreement-gate: a violation may only AUTO-CONFIRM (auto-challan) when the specialized
        model AND the VLM agree it's real. On disagreement we never auto-challan — auto_confirm
        drops to human_review, and an already-uncertain human_review candidate is discarded.
        If the VLM is unavailable, the model's own band stands (graceful degradation)."""
        if self.vlm is None or not self.vlm.available():
            return
        # Gather the (violation, crop) pairs that need a VLM opinion, then fire all calls
        # CONCURRENTLY — a multi-violation image otherwise pays N x the per-call latency
        # sequentially (each NVIDIA NIM call is network-bound, so threads overlap well).
        pending = []
        for v in violations:
            if not v.get("needs_vlm"):
                continue
            # wrong-side comes from a dedicated heading model (2/2 + 0 FP on samples) that is more
            # reliable than the VLM, which mis-judges direction from a single frame (it denied a
            # real wrong-side). Trust the model -> skip VLM for this type so it isn't wrongly killed.
            if v.get("type") == "wrong_side":
                v["needs_vlm"] = False
                continue
            bb = v.get("vehicle_bbox") or v.get("bbox")
            crop = self._safe_crop(work, bb) if bb else None
            if crop is not None:
                pending.append((v, crop))
        if not pending:
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(pending))) as ex:
            results = list(ex.map(lambda pc: self.vlm.verify(pc[1], pc[0]["type"]), pending))
        for (v, _crop), r in zip(pending, results):
            self._apply_vlm_result(v, r)

    @staticmethod
    def _apply_vlm_result(v: dict, r: dict) -> None:
        """Agreement matrix: auto_confirm survives only if the VLM also confirms; on disagreement
        we never auto-challan (drop to human_review), and a low-confidence candidate the VLM denies
        is discarded. VLM-unavailable -> the model's own band stands."""
        confirmed = r.get("confirmed")
        band = v.get("band")
        v["vlm"] = {"model_unavailable": r["model_unavailable"], "confirmed": confirmed,
                    "vlm_confidence": r.get("vlm_confidence"), "caption": r.get("caption")}
        if r["model_unavailable"]:
            v["vlm"]["agreement"] = "vlm_unavailable: model band stands"
        elif confirmed is True:
            v["vlm"]["agreement"] = (
                "agree: model+VLM confirm -> auto_confirm" if band == "auto_confirm"
                else "agree: VLM confirms -> human_review (still needs sign-off)")
        elif confirmed is False:
            if band == "auto_confirm":
                v["band"] = "human_review"
                v["needs_vlm"] = False
                v["vlm"]["agreement"] = (
                    "DISAGREE: model=violation / VLM=no -> auto_confirm downgraded to human_review")
            else:
                v["band"] = "discard"
                v["vlm"]["agreement"] = "VLM denies low-confidence candidate -> discard"
        else:
            v["vlm"]["agreement"] = "vlm_inconclusive: model band stands"

    # ---- ROI-gated ANPR --------------------------------------------------

    @staticmethod
    def _bbox_area(v: dict) -> float:
        vb = v.get("vehicle_bbox")  # xyxy
        if vb:
            return max(0.0, vb[2] - vb[0]) * max(0.0, vb[3] - vb[1])
        b = v.get("bbox")  # xywh
        return max(0.0, b[2]) * max(0.0, b[3]) if b else 0.0

    def _run_anpr(self, work, violations: list[dict]) -> dict:
        kept = [v for v in violations if v.get("band") != "discard"]
        if not kept:
            return {}
        # Pick the violation with the largest vehicle box, not just the first in the list --
        # a frame can have several violators (e.g. a small unrelated motorcycle flagged for
        # "helmet" elsewhere in frame) and kept[0] could be a tiny crop with no legible plate
        # while a bigger violator (e.g. the actual triple-riding bike) sits right next to it.
        best = max(kept, key=self._bbox_area)
        bb = best.get("vehicle_bbox") or best.get("bbox")
        crop = self._safe_crop(work, bb) if bb else None
        if crop is None:
            return {"plate": {"text": "", "confidence": 0.0, "note": "no vehicle crop available"}}
        # self.sam3 is the legacy local-subprocess client and is always None on the hosted path
        # (see __init__) -- plate localization must go through the live Roboflow client that
        # sam3_violations.py already uses (self.sam3v.sam3), not the dead self.sam3.
        rf = self.sam3v.sam3 if self.sam3v is not None else None
        plate_box = ANPRModule.locate_plate_sam3(crop, rf) if rf is not None else None
        if plate_box is None:
            return {"plate": {"text": "", "confidence": 0.0,
                              "note": "SAM-3 unavailable or no plate localized"}}
        plate_crop = ANPRModule.crop_from_bbox(crop, plate_box["bbox"])
        if plate_crop.size == 0:
            return {"plate": {"text": "", "confidence": 0.0, "note": "empty plate crop"}}
        rec = self.anpr.recognize(plate_crop)
        rec["locate_confidence"] = plate_box["confidence"]
        return {"plate": rec}

    @staticmethod
    def _safe_crop(work, bb):
        x1, y1, x2, y2 = [int(round(v)) for v in bb]
        x1, y1 = max(0, x1), max(0, y1)
        crop = work[y1:y2, x1:x2]
        return crop if crop.size > 0 else None


def run_demo(cfg: PipelineConfig, input_path: str, limit: int | None = None) -> int:
    p = Path(input_path)
    imgs = [p] if p.is_file() else sorted([f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS])
    if not imgs:
        log(f"[pipeline] no images found at {input_path}")
        return 1
    if limit:
        imgs = imgs[:limit]
    pipe = Pipeline(cfg)
    run_id = new_run_id()
    out_summary = []
    n_viol = 0
    for i, img in enumerate(imgs, 1):
        r = pipe.process_image(str(img))
        out_summary.append(r)
        n_viol += len(r.get("violations", []))
        if i % 20 == 0 or i == len(imgs):
            log(f"   processed {i}/{len(imgs)} | violations so far: {n_viol}")
    sp = Path(cfg.output_dir) / f"demo_summary_{run_id}.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(out_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[pipeline] demo done: {len(imgs)} images, {n_viol} candidate violations.")
    log(f"[pipeline] evidence -> {cfg.output_dir}/  | summary -> {sp}")
    if pipe.sam3 is not None:
        pipe.sam3.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Gridlock 2.0 violation pipeline (orchestrator)")
    ap.add_argument("--mode", choices=["demo"], default="demo")
    ap.add_argument("--input", help="image file or folder")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--variant", default="nano")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--camera-config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max images to process")
    ap.add_argument("--no-sam3", action="store_true", help="disable SAM-3 (helmet/plate)")
    ap.add_argument("--no-vlm", action="store_true", help="disable VLM cascade verification")
    ap.add_argument("--write-sample-config", metavar="PATH",
                    help="write a sample per-camera geometry config and exit")
    args = ap.parse_args()

    if args.write_sample_config:
        outp = Path(args.write_sample_config)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(SAMPLE_CAMERA_CONFIG, indent=2), encoding="utf-8")
        log(f"[pipeline] wrote sample camera config -> {outp}")
        return 0

    if not args.input:
        ap.error("--input is required for --mode demo")
    cfg = PipelineConfig(output_dir=args.output, variant=args.variant,
                         threshold=args.threshold, camera_config=args.camera_config,
                         use_sam3=not args.no_sam3, use_vlm=not args.no_vlm)
    return run_demo(cfg, args.input, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
