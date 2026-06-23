"""Module 1 — Preprocessing (conditional, quality-gated).

§3.1 specifies Retinexformer (low-light) and OneRestore (composite degradation).
Both are loaded from ml/third_party/ at init time; if unavailable the class
falls back to classical CLAHE+gamma / unsharp, preserving the same interface.

Restoration runs ONLY when the quality gate flags a problem (avoids artifacts
+ wasted compute, per the design's "skip if good" rule).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_THIRD_PARTY = Path(__file__).parent.parent.parent / "third_party"


@dataclass
class PreprocessResult:
    applied: list = field(default_factory=list)
    learned_restorer_available: bool = False
    note: str = ""
    image = None  # restored image (numpy BGR) or original

    def to_dict(self) -> dict:
        return {"applied": self.applied, "learned_restorer_available": self.learned_restorer_available,
                "note": self.note}


class Preprocessor:
    # max spatial dimension for Retinexformer (>340 segfaults on constrained pagefile)
    _RETINEX_MAX_DIM = 320

    def __init__(self, enable: bool = True):
        self.enable = enable
        self._retinex = None          # RetinexFormer model
        self._onerestore = None       # (embedder, restorer) tuple
        self._try_load_retinex()
        self._try_load_onerestore()

    @property
    def learned_available(self) -> bool:
        return self._retinex is not None or self._onerestore is not None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_retinex(self):
        try:
            import torch
            retinex_dir = str(_THIRD_PARTY / "Retinexformer")
            if retinex_dir not in sys.path:
                sys.path.insert(0, retinex_dir)
            from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
            weights = _THIRD_PARTY / "Retinexformer" / "pretrained_weights" / "LOL_v2_real.pth"
            if not weights.exists():
                return
            device = torch.device("cpu")
            model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
            ckpt = torch.load(str(weights), map_location=device)
            model.load_state_dict(ckpt.get("params", ckpt), strict=True)
            model.to(device).eval()
            self._retinex = model
        except Exception:  # noqa: BLE001
            pass

    def _try_load_onerestore(self):
        try:
            onerestore_dir = str(_THIRD_PARTY / "OneRestore")
            stubs_dir = str(_THIRD_PARTY / "_stubs")
            for d in (stubs_dir, onerestore_dir):
                if d not in sys.path:
                    sys.path.insert(0, d)
            from utils.utils import load_restore_ckpt, load_embedder_ckpt  # noqa: PLC0415
            import torch
            device = torch.device("cpu")
            embedder_path = _THIRD_PARTY / "OneRestore" / "ckpts" / "embedder_model.tar"
            restorer_path = _THIRD_PARTY / "OneRestore" / "ckpts" / "onerestore_real.tar"
            if not embedder_path.exists() or not restorer_path.exists():
                return
            embedder = load_embedder_ckpt(device, freeze_model=True, ckpt_name=str(embedder_path))
            restorer = load_restore_ckpt(device, freeze_model=True, ckpt_name=str(restorer_path))
            self._onerestore = (embedder, restorer)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def restore(self, image, quality) -> PreprocessResult:
        """image: numpy/PIL/path. quality: QualityReport. Returns PreprocessResult (.image set)."""
        res = PreprocessResult(learned_restorer_available=self.learned_available)
        try:
            import cv2
            import numpy as np
            img = self._to_bgr(image, cv2, np)
            res.image = img
            if not self.enable or quality is None or not getattr(quality, "needs_restoration", False):
                res.note = "skipped (quality ok or disabled)"
                return res
            out = img
            if getattr(quality, "is_low_light", False):
                if self._retinex is not None:
                    out = self._retinex_enhance(out, np)
                    res.applied.append("Retinexformer (LOL_v2_real)")
                else:
                    out = self._clahe_gamma(out, cv2, np)
                    res.applied.append("clahe+gamma (fallback)")
            if getattr(quality, "is_blurry", False):
                if self._onerestore is not None:
                    out = self._onerestore_enhance(out, np)
                    res.applied.append("OneRestore (onerestore_real)")
                else:
                    out = self._unsharp(out, cv2)
                    res.applied.append("unsharp (fallback)")
            res.image = out
            res.note = "learned" if self.learned_available else "classical fallback"
            return res
        except Exception as e:  # noqa: BLE001
            res.note = f"preprocess error: {type(e).__name__}: {e}"
            return res

    # ------------------------------------------------------------------
    # Learned restorers
    # ------------------------------------------------------------------

    def _retinex_enhance(self, bgr, np):
        import torch
        from PIL import Image
        rgb_pil = Image.fromarray(bgr[:, :, ::-1])
        orig_w, orig_h = rgb_pil.size
        scale = self._RETINEX_MAX_DIM / max(orig_w, orig_h)
        if scale < 1.0:
            small = rgb_pil.resize((max(1, round(orig_w * scale)), max(1, round(orig_h * scale))),
                                   Image.BICUBIC)
        else:
            small = rgb_pil
        sw, sh = small.size
        pw = (4 - sw % 4) % 4
        ph = (4 - sh % 4) % 4
        if pw or ph:
            small = small.resize((sw + pw, sh + ph), Image.BICUBIC)
        inp = torch.from_numpy(np.array(small).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = self._retinex(inp)
        out_arr = (out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out_pil = Image.fromarray(out_arr).resize((orig_w, orig_h), Image.BICUBIC)
        return np.array(out_pil)[:, :, ::-1].copy()  # RGB→BGR

    def _onerestore_enhance(self, bgr, np):
        import torch
        from PIL import Image
        from torchvision import transforms
        transform_resize = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
        embedder, restorer = self._onerestore
        rgb_pil = Image.fromarray(bgr[:, :, ::-1])
        orig_w, orig_h = rgb_pil.size
        scale = 640 / max(orig_w, orig_h)
        small = rgb_pil.resize((max(1, round(orig_w * scale)), max(1, round(orig_h * scale))),
                               Image.BICUBIC) if scale < 1.0 else rgb_pil
        with torch.no_grad():
            lq_re = torch.Tensor((np.array(small) / 255).transpose(2, 0, 1)).unsqueeze(0)
            lq_em = transform_resize(small).unsqueeze(0)
            text_embedding, _, _ = embedder(lq_em, "image_encoder")
            out = restorer(lq_re, text_embedding)
        out_arr = (out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out_pil = Image.fromarray(out_arr).resize((orig_w, orig_h), Image.BICUBIC)
        return np.array(out_pil)[:, :, ::-1].copy()  # RGB→BGR

    # ------------------------------------------------------------------
    # Classical fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bgr(image, cv2, np):
        if isinstance(image, str):
            return cv2.imread(image)
        if hasattr(image, "mode"):
            return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return np.asarray(image).copy()

    @staticmethod
    def _clahe_gamma(img, cv2, np, gamma: float = 1.4):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img2 = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        inv = 1.0 / gamma
        table = (np.array([((i / 255.0) ** inv) * 255 for i in range(256)])).astype("uint8")
        return cv2.LUT(img2, table)

    @staticmethod
    def _unsharp(img, cv2):
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, blur, -0.5, 0)
