"""Phase 0a — Hardware & environment detection.

Detects GPU/VRAM/CPU/RAM and which ML packages are importable, then derives a
*hardware tier* used to pick model variant sizes throughout the pipeline. Writes
logs/environment_report.md.

The Master Design's GPU-day budgets (§8) assume A100s. This script recomputes a
realistic verdict for the actual local machine instead of using those numbers
verbatim — including whether RF-DETR-L Stage-2, Co-DETR, and the VLM verifier are
runnable locally at all or must be stubbed / pushed to the cloud.

Run:  python src/hardware.py
"""
from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

from src.utils.logging import REPO_ROOT, log, new_run_id, write_run_log

# Packages the pipeline relies on, grouped by role (per 06_model_selection_justification.md).
PACKAGES = {
    "core": ["numpy", "cv2", "PIL", "pandas", "lxml", "yaml"],
    "torch": ["torch", "torchvision"],
    "detection": ["rfdetr", "ultralytics", "supervision"],
    "ocr": ["paddleocr", "paddle", "easyocr"],
    "vlm": ["transformers", "huggingface_hub"],
    "eval": ["pycocotools"],
}


def _probe_packages() -> dict:
    out = {}
    for group, mods in PACKAGES.items():
        out[group] = {}
        for m in mods:
            try:
                mod = importlib.import_module(m)
                out[group][m] = {"ok": True, "version": getattr(mod, "__version__", "?")}
            except Exception as e:  # noqa: BLE001 - we want to report any failure
                out[group][m] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}
    return out


def _gpu_via_torch() -> dict:
    info = {"torch_cuda_available": False, "devices": [], "error": None}
    try:
        import torch
        info["torch_version"] = torch.__version__
        try:
            avail = torch.cuda.is_available()
        except Exception as e:  # cudaGetDeviceCount can throw on broken/full GPUs
            info["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            avail = False
        info["torch_cuda_available"] = bool(avail)
        if avail:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "name": p.name,
                    "vram_gb": round(p.total_memory / 1024**3, 2),
                    "capability": f"{p.major}.{p.minor}",
                })
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return info


def _gpu_via_nvidia_smi() -> dict:
    out = {"available": False, "raw": None}
    if shutil.which("nvidia-smi") is None:
        return out
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0 and r.stdout.strip():
            out["available"] = True
            out["raw"] = r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def _cpu_ram() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "cpu_count": None, "ram_gb": None}
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        # Fallback: Windows WMIC-free estimate via ctypes
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["ram_gb"] = round(stat.ullTotalPhys / 1024**3, 1)
        except Exception:
            pass
    return info


def decide_tier(vram_gb: float | None, cuda_ok: bool) -> dict:
    """Map detected hardware to a tier that drives model-variant choices.

    Tiers (local-execution viability):
      cloud_required : no usable CUDA GPU OR <6GB VRAM -> heavy work must go to cloud
      low            : 6-11 GB  -> RF-DETR-N/S only, no Co-DETR/VLM locally
      mid            : 12-23 GB -> RF-DETR-L feasible, VLM tight (quantized only)
      high           : >=24 GB  -> close to the §8 A100 assumptions
    """
    if not cuda_ok or vram_gb is None or vram_gb < 6:
        tier = "cloud_required"
    elif vram_gb < 12:
        tier = "low"
    elif vram_gb < 24:
        tier = "mid"
    else:
        tier = "high"

    rec = {
        "cloud_required": {
            "detector_inference": "RF-DETR-N on CPU (slow) or push to cloud GPU",
            "detector_finetune": "NOT local — use cloud (e.g. Lightning H200)",
            "stage2_codetr": "stub (not runnable locally)",
            "vlm_verifier": "stub (not runnable locally)",
            "batch_size": 1,
        },
        "low": {
            "detector_inference": "RF-DETR-N/S",
            "detector_finetune": "RF-DETR-N with grad-accum, small batch",
            "stage2_codetr": "skip/stub",
            "vlm_verifier": "stub or tiny quantized only",
            "batch_size": 2,
        },
        "mid": {
            "detector_inference": "RF-DETR-S/L",
            "detector_finetune": "RF-DETR-S/L feasible",
            "stage2_codetr": "RF-DETR-L (Co-DETR still heavy)",
            "vlm_verifier": "quantized 7B only, selective",
            "batch_size": 8,
        },
        "high": {
            "detector_inference": "RF-DETR-L",
            "detector_finetune": "per §8 estimates",
            "stage2_codetr": "Co-DETR feasible",
            "vlm_verifier": "7B selective",
            "batch_size": 16,
        },
    }[tier]
    return {"tier": tier, "recommendations": rec}


def collect() -> dict:
    return {
        "cpu_ram": _cpu_ram(),
        "gpu_torch": _gpu_via_torch(),
        "gpu_nvidia_smi": _gpu_via_nvidia_smi(),
        "packages": _probe_packages(),
    }


def _md_report(env: dict, tier: dict) -> str:
    cr = env["cpu_ram"]
    gt = env["gpu_torch"]
    smi = env["gpu_nvidia_smi"]
    vram = gt["devices"][0]["vram_gb"] if gt["devices"] else "n/a"
    lines = []
    lines.append("# Environment Report (Phase 0a)\n")
    lines.append(f"_Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("## Hardware tier\n")
    lines.append(f"**TIER = `{tier['tier']}`**\n")
    lines.append("| Setting | Recommendation |")
    lines.append("|---|---|")
    for k, v in tier["recommendations"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## CPU / RAM\n")
    lines.append(f"- Platform: `{cr['platform']}`")
    lines.append(f"- Processor: `{cr['processor']}`")
    lines.append(f"- Python: `{cr['python']}`  · CPU count: `{cr['cpu_count']}`  · RAM: `{cr['ram_gb']} GB`")
    lines.append("")
    lines.append("## GPU\n")
    lines.append(f"- torch: `{gt.get('torch_version','?')}` · CUDA available: **{gt['torch_cuda_available']}**")
    if gt.get("error"):
        lines.append(f"- ⚠️ torch CUDA error: `{gt['error']}`")
    if gt["devices"]:
        for d in gt["devices"]:
            lines.append(f"- Device: `{d['name']}` · VRAM `{d['vram_gb']} GB` · capability `{d['capability']}`")
    else:
        lines.append("- No CUDA device visible to torch.")
    if smi.get("available"):
        lines.append(f"- nvidia-smi: `{smi['raw']}`")
    else:
        lines.append("- nvidia-smi: not available / no output")
    lines.append(f"\n  (Detected VRAM for tiering: **{vram} GB**)\n")
    lines.append("## Package availability\n")
    lines.append("| Group | Package | Status | Version / Error |")
    lines.append("|---|---|---|---|")
    for group, mods in env["packages"].items():
        for m, info in mods.items():
            status = "✅" if info["ok"] else "❌"
            detail = info.get("version") if info["ok"] else info.get("error", "")
            lines.append(f"| {group} | {m} | {status} | {detail} |")
    lines.append("")
    lines.append("## Realistic local budget verdict\n")
    if tier["tier"] == "cloud_required":
        lines.append(
            "- **Local GPU training/inference of the heavy stack is NOT viable.** Either no usable "
            "CUDA device or <6 GB VRAM. The §8 A100-GPU-day budgets do not apply locally.\n"
            "- **Plan:** run zero-shot baselines on CPU where packages permit (Phase 2/3), and push all "
            "fine-tuning (Phase 5) and any RF-DETR-L / Co-DETR / VLM work to a cloud GPU (the Lightning "
            "H200 already mentioned). Locally, those components are **stubbed** with `model_unavailable`.")
    else:
        lines.append("- Local execution feasible at the tier above; recompute per-module fine-tune time "
                     "empirically on the first run rather than trusting the A100 estimate.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    run_id = new_run_id()
    log("[Phase 0a] Detecting hardware & environment...")
    env = collect()
    gt = env["gpu_torch"]
    vram = gt["devices"][0]["vram_gb"] if gt["devices"] else None
    tier = decide_tier(vram, gt["torch_cuda_available"])

    report_path = REPO_ROOT / "logs" / "environment_report.md"
    report_path.write_text(_md_report(env, tier), encoding="utf-8")

    write_run_log("phase0", "hardware", run_id, {"environment": env, "tier": tier})

    log(f"[Phase 0a] TIER = {tier['tier']}")
    log(f"[Phase 0a] CUDA available: {gt['torch_cuda_available']}"
        + (f" (error: {gt['error']})" if gt.get("error") else ""))
    log(f"[Phase 0a] Report -> {report_path}")
    # Compact package summary to console
    miss = [f"{g}/{m}" for g, mods in env["packages"].items() for m, i in mods.items() if not i["ok"]]
    if miss:
        log(f"[Phase 0a] MISSING packages: {', '.join(miss)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
