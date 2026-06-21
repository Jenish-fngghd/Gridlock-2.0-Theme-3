# Environment Report (Phase 0a)

_Generated: 2026-06-19T12:53:25_

## Hardware tier

**TIER = `cloud_required`**

| Setting | Recommendation |
|---|---|
| detector_inference | RF-DETR-N on CPU (slow) or push to cloud GPU |
| detector_finetune | NOT local — use cloud (e.g. Lightning H200) |
| stage2_codetr | stub (not runnable locally) |
| vlm_verifier | stub (not runnable locally) |
| batch_size | 1 |

## CPU / RAM

- Platform: `Windows-11-10.0.26200-SP0`
- Processor: `AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`
- Python: `3.14.3`  · CPU count: `16`  · RAM: `15.3 GB`

## GPU

- torch: `2.12.0+cu126` · CUDA available: **True**
- Device: `NVIDIA GeForce RTX 3050 Laptop GPU` · VRAM `4.0 GB` · capability `8.6`
- nvidia-smi: `NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 572.60`

  (Detected VRAM for tiering: **4.0 GB**)

## Package availability

| Group | Package | Status | Version / Error |
|---|---|---|---|
| core | numpy | ✅ | 2.4.3 |
| core | cv2 | ❌ | ModuleNotFoundError: No module named 'cv2' |
| core | PIL | ✅ | 12.1.1 |
| core | pandas | ✅ | 3.0.1 |
| core | lxml | ❌ | ModuleNotFoundError: No module named 'lxml' |
| core | yaml | ✅ | 6.0.3 |
| torch | torch | ✅ | 2.12.0+cu126 |
| torch | torchvision | ❌ | ModuleNotFoundError: No module named 'torchvision' |
| detection | rfdetr | ❌ | ModuleNotFoundError: No module named 'rfdetr' |
| detection | ultralytics | ❌ | ModuleNotFoundError: No module named 'ultralytics' |
| detection | supervision | ❌ | ModuleNotFoundError: No module named 'supervision' |
| ocr | paddleocr | ❌ | ModuleNotFoundError: No module named 'paddleocr' |
| ocr | paddle | ❌ | ModuleNotFoundError: No module named 'paddle' |
| ocr | easyocr | ❌ | ModuleNotFoundError: No module named 'easyocr' |
| vlm | transformers | ❌ | ModuleNotFoundError: No module named 'transformers' |
| vlm | huggingface_hub | ✅ | 1.7.2 |
| eval | pycocotools | ❌ | ModuleNotFoundError: No module named 'pycocotools' |

## Realistic local budget verdict

- **Local GPU training/inference of the heavy stack is NOT viable.** Either no usable CUDA device or <6 GB VRAM. The §8 A100-GPU-day budgets do not apply locally.
- **Plan:** run zero-shot baselines on CPU where packages permit (Phase 2/3), and push all fine-tuning (Phase 5) and any RF-DETR-L / Co-DETR / VLM work to a cloud GPU (the Lightning H200 already mentioned). Locally, those components are **stubbed** with `model_unavailable`.
