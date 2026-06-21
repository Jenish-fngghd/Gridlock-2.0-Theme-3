"""SAM-3 worker — runs INSIDE the isolated `.venv-sam3` interpreter (numpy<2 pinned by the
official facebookresearch/sam3 package; incompatible with the main pipeline's numpy>=2 env).

Loads the model ONCE, then serves newline-delimited JSON requests over stdin/stdout so the main
process never pays the ~3.2GB reload cost per call (cost control — SAM-3 is ~14x slower than
RF-DETR per the Phase-5 benchmark, so it's kept warm and called only for the rare crops that
need it: helmet-state and plate localization on confirmed/candidate violators).

Request:  {"id": "...", "image_b64": "...", "prompts": ["concept a", "concept b"], "threshold": 0.3}
Response: {"id": "...", "ok": true, "results": {"concept a": [{"box":[x1,y1,x2,y2],"score":0.9}, ...]}}
       or {"id": "...", "ok": false, "error": "..."}
A line `{"cmd": "ping"}` gets `{"ok": true, "ready": true}` without running inference.
A line `{"cmd": "shutdown"}` exits cleanly.
"""
from __future__ import annotations

import base64
import io
import json
import sys


def _log(msg: str) -> None:
    print(f"[sam3_worker] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    try:
        import torch
        from PIL import Image
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as e:  # noqa: BLE001
        _log(f"FATAL import error: {type(e).__name__}: {e}")
        print(json.dumps({"cmd": "boot", "ok": False, "error": str(e)}), flush=True)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = build_sam3_image_model(device=device)
        processor = Sam3Processor(model, confidence_threshold=0.0)  # filter ourselves per-request
        _log(f"model loaded on {device}")
        print(json.dumps({"cmd": "boot", "ok": True, "device": device}), flush=True)
    except Exception as e:  # noqa: BLE001
        _log(f"FATAL model load error: {type(e).__name__}: {e}")
        print(json.dumps({"cmd": "boot", "ok": False, "error": str(e)}), flush=True)
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"bad json: {e}"}), flush=True)
            continue

        if req.get("cmd") == "shutdown":
            _log("shutdown requested")
            return 0
        if req.get("cmd") == "ping":
            print(json.dumps({"ok": True, "ready": True}), flush=True)
            continue

        rid = req.get("id", "")
        try:
            img_bytes = base64.b64decode(req["image_b64"])
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            threshold = float(req.get("threshold", 0.3))
            prompts = req.get("prompts", [])

            inference_state = processor.set_image(image)
            results: dict[str, list[dict]] = {}
            for concept in prompts:
                state = processor.set_text_prompt(state=inference_state, prompt=concept)
                boxes = state["boxes"].to(torch.float32).cpu().numpy()
                scores = state["scores"].to(torch.float32).cpu().numpy()
                items = [{"box": [float(v) for v in box], "score": float(score)}
                         for box, score in zip(boxes, scores) if float(score) >= threshold]
                items.sort(key=lambda d: -d["score"])
                results[concept] = items
            print(json.dumps({"id": rid, "ok": True, "results": results}), flush=True)
        except Exception as e:  # noqa: BLE001
            _log(f"request {rid} failed: {type(e).__name__}: {e}")
            print(json.dumps({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
