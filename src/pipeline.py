"""Pipeline orchestrator — wires the modules per 00_master_design.md §2.

Flow:  ingest+QA → preprocess(gated) → detect → track → [helmet/triple · geometry] →
       confidence cascade → ANPR(violators) → evidence.

Single still vs sequence: temporal violations (wrong-side/stop-line/red-light/parking) need
motion, so on a single image the geometry engine abstains (logged), and the demo focuses on
the instance/counting violations + detection + signed evidence. Every module degrades
gracefully — a missing model is logged (`model_unavailable`) and the run continues.

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
from src.modules.detection import VehicleDetector
from src.modules.evidence import EvidenceGenerator
from src.modules.geometry_engine import GeometryEngine, SceneConfig
from src.modules.helmet_triple import HelmetTripleModule
from src.modules.preprocessing import Preprocessor
from src.modules.quality_gate import QualityGate
from src.modules.tracking import IoUTracker
from src.utils.logging import REPO_ROOT, log, new_run_id

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

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


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.quality = QualityGate()
        self.pre = Preprocessor(enable=True)
        self.detector = VehicleDetector.for_tier(cfg.tier, threshold=cfg.threshold)
        self.tracker = IoUTracker()
        self.helmet_triple = HelmetTripleModule(detector=self.detector)
        scene = SceneConfig.from_dict(self._load_camera_cfg(cfg.camera_config))
        self.geometry = GeometryEngine(scene)
        self.cascade = ConfidenceCascade()
        self.anpr = ANPRModule(ocr_engine="auto")
        self.evidence = EvidenceGenerator(
            cfg.output_dir, camera_id=scene.camera_id,
            model_versions={"detector": f"rfdetr-{cfg.variant}", "ocr": self.anpr._pref,
                            "tracker": self.tracker.backend},
            scene_config=scene.camera_id)

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
        q = self.quality.assess(img)
        pre = self.pre.restore(img, q)
        work = pre.image if pre.image is not None else img

        det = self.detector.detect(work)
        ht = self.helmet_triple.analyze(detections=det.detections if not det.model_unavailable else [])

        # single still -> temporal/geometry abstains (no track history)
        geo_note = "abstained: single still has no motion history (needs sequence)"

        # assemble candidate violations (instance/counting paradigm this run)
        violations = []
        for g in ht.get("groups", []):
            if g.get("triple_riding"):
                mb = g["motorbike_bbox"]
                violations.append({
                    "type": "triple_riding", "confidence": g["triple_confidence"],
                    "bbox": [mb[0], mb[1], mb[2] - mb[0], mb[3] - mb[1]],
                    "evidence_chain": ["detect", "associate_riders", "count>=3"],
                    "rider_count": g["rider_count"], "basis": "proxy (no GT label)"})
            # helmet intentionally NOT emitted as a positive — not_testable zero-shot
        decisions = self.cascade.decide_many(violations)
        for v, d in zip(violations, decisions):
            v["band"] = d.band
            v["calibrated_confidence"] = d.calibrated_confidence
            v["needs_vlm"] = d.needs_vlm

        # ANPR is ROI-gated to violators; zero-shot plate detection unavailable -> flag
        vehicle = {}
        if violations and not self.anpr.plate_detection_available():
            vehicle = {"plate": {"text": "", "confidence": 0.0,
                                 "note": "plate detection needs fine-tune (no COCO plate class)"}}

        # Only emit a signed evidence record when there is an actual violation to document.
        evidence_id = None
        if violations:
            record = self.evidence.generate(
                work, violations, vehicle=vehicle, frame_ref=img_path, vlm_caption="")
            evidence_id = record["violation_id"]
        return {
            "image": img_path,
            "quality": q.to_dict(),
            "preprocess": pre.to_dict(),
            "detections": len(det.detections),
            "detector_unavailable": det.model_unavailable,
            "helmet_status": ht.get("helmet_status", "not_testable"),
            "triple_riding_count": ht.get("triple_riding_count", 0),
            "geometry": geo_note,
            "violations": violations,
            "evidence_record": evidence_id,
        }


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
    log("[pipeline] NOTE: helmet=not_testable (zero-shot); temporal violations abstain on stills.")
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
                         threshold=args.threshold, camera_config=args.camera_config)
    return run_demo(cfg, args.input, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
