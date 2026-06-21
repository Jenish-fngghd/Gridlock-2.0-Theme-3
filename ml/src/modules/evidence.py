"""Module 7 — Evidence generator.

00_master_design.md §4 / 06_... row 22: each confirmed violation -> annotated image + signed
JSON record (timestamp, camera, bbox, evidence-chain, plate, VLM caption) with SHA-256 +
audit trail. Tamper-evident => court-admissible (the differentiator vs a demo, §11).

The "signature" here is a SHA-256 over the canonical record (integrity/tamper-evidence). A
real deployment would additionally sign that hash with a private key (HMAC/asymmetric); the
hook is left explicit in the audit block.

cv2 is used only for the annotated image; if missing, the JSON record is still emitted.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_of(record: dict) -> str:
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


class EvidenceGenerator:
    def __init__(self, output_dir: str | Path, camera_id: str = "CAM_UNKNOWN",
                 model_versions: dict | None = None, scene_config: str = ""):
        self.output_dir = Path(output_dir)
        (self.output_dir / "annotated").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "records").mkdir(parents=True, exist_ok=True)
        self.camera_id = camera_id
        self.model_versions = model_versions or {}
        self.scene_config = scene_config

    def generate(self, image, violations: list[dict], vehicle: dict | None = None,
                 frame_ref: str = "", vlm_caption: str = "", timestamp: str | None = None) -> dict:
        """violations: [{type, role?, confidence, bbox:[x,y,w,h], band?, evidence_chain?, verified_by?}].

        Consumers filter on `band`/flags; we record all and never crash.
        """
        vid = str(uuid.uuid4())
        ts = timestamp or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        annotated_path = self.output_dir / "annotated" / f"{vid}.jpg"
        annotated_ok = self._annotate(image, violations, vehicle, annotated_path)

        record = {
            "violation_id": vid,
            "timestamp": ts,
            "camera_id": self.camera_id,
            "frame_ref": frame_ref,
            "violations": violations,
            "vehicle": vehicle or {},
            "evidence_image": str(annotated_path) if annotated_ok else None,
            "vlm_caption": vlm_caption,
            "audit": {
                "model_versions": self.model_versions,
                "scene_config": self.scene_config,
                "review_status": "pending",
                "signature_alg": "sha256(canonical_record)",  # + key-signature hook in prod
            },
        }
        # hash over the record WITHOUT the sha field, then attach
        record["audit"]["sha256"] = sha256_of(record)
        out = self.output_dir / "records" / f"{vid}.json"
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    @staticmethod
    def _annotate(image, violations, vehicle, out_path: Path) -> bool:
        try:
            import cv2
            import numpy as np
            if isinstance(image, str):
                img = cv2.imread(image)
            elif hasattr(image, "mode"):
                img = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            else:
                img = np.asarray(image).copy()
            if img is None:
                return False
            for v in violations:
                bb = v.get("bbox")
                if not bb or len(bb) != 4:
                    continue
                x, y, w, h = [int(round(t)) for t in bb]
                color = (0, 0, 255)  # red
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                label = f"{v.get('type','?')} {v.get('confidence',0):.2f}"
                cv2.putText(img, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if vehicle and vehicle.get("plate", {}).get("text"):
                cv2.putText(img, vehicle["plate"]["text"], (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(str(out_path), img)
            return True
        except Exception:
            return False
