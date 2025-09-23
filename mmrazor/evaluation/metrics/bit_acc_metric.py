from __future__ import annotations

from typing import Dict, List, Optional

import os
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmrazor.registry import METRICS


@METRICS.register_module()
class BitAccMetric(BaseMetric):
    """Compute student/teacher bit accuracy using the aligned decoder pipeline.

    This metric assumes the algorithm model exposes `student` and `teacher`
    submodules, and the student has an attribute `sdecoder`.
    The decoding procedure mirrors the logic in BitAccEvalHook to ensure
    consistency with the demo's `decoder2` (rotation search, ROI crop, BCE pick).
    """

    def __init__(self,
                 image_size: tuple[int, int] = (400, 400),
                 mdecoder_ckpt: Optional[str] = None) -> None:
        super().__init__()
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.mdecoder_ckpt = mdecoder_ckpt
        self._mdecoder = None

    def _ensure_mdecoder(self, device: torch.device) -> None:
        if self._mdecoder is not None:
            return
        import sys
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        wmdemo_dir = os.path.join(project_root, 'WMdemo')
        if wmdemo_dir not in sys.path:
            sys.path.insert(0, wmdemo_dir)
        from nets.hrnet import HRnet  # type: ignore
        mdecoder = HRnet(num_classes=1, backbone='model_data/hrnetv2_w48_weights_voc.pth', pretrained=False)
        ckpt_path = self.mdecoder_ckpt if self.mdecoder_ckpt is not None else os.path.join(wmdemo_dir, 'models', 'Mdecoder_210000.pth')
        state = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        mdecoder.load_state_dict(state, strict=False)
        self._mdecoder = mdecoder.to(device).train()

    def process(self, data_batch: Dict, data_samples: List) -> None:  # type: ignore[override]
        model = self._runner.model  # type: ignore[attr-defined]
        device = next(model.parameters()).device
        self._ensure_mdecoder(device)

        student = model.student
        sdecoder = getattr(student, 'sdecoder', None)
        if sdecoder is None:
            return
        sdecoder = sdecoder.to(device)
        sdecoder.train()

        H, W = self.image_size

        import sys
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        wmdemo_dir = os.path.join(project_root, 'WMdemo')
        if wmdemo_dir not in sys.path:
            sys.path.insert(0, wmdemo_dir)
        import utils as wmd_utils  # type: ignore
        from EMD_maotai import decoder2  # type: ignore
        from PIL import Image, ImageOps
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        try:
            if hasattr(Image, 'Resampling'):
                RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
            elif hasattr(Image, 'LANCZOS'):
                RESAMPLE_LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]
            else:
                RESAMPLE_LANCZOS = Image.ANTIALIAS  # type: ignore[attr-defined]
        except Exception:
            RESAMPLE_LANCZOS = None  # type: ignore[assignment]

        correct_s = 0
        correct_t = 0
        total_bits = 0

        for ds in data_samples:
            meta = getattr(ds, 'metainfo', {})
            img_path = None
            if isinstance(meta, dict) and meta:
                img_path = meta.get('img_path', None) or meta.get('ori_filename', None) or meta.get('filename', None)
            if (not img_path) and hasattr(ds, 'img_path'):
                img_path = getattr(ds, 'img_path')
            if not img_path:
                continue
            # GT bits
            gt = getattr(ds, 'gt_payload', None)
            if gt is None:
                meta2 = getattr(ds, 'metainfo', {})
                if isinstance(meta2, dict):
                    gt = meta2.get('gt_payload', None)
            if gt is None:
                continue
            gt_bits = torch.as_tensor(gt, dtype=torch.float32, device=device)

            # Student decode with student's mask (decoder2-like steps)
            pil = Image.open(str(img_path)).convert('RGB')
            pil_fit = ImageOps.fit(pil, (W, H), RESAMPLE_LANCZOS) if RESAMPLE_LANCZOS is not None else ImageOps.fit(pil, (W, H))
            angles = [0, 3, -3]
            best_probs = None
            best_bce = None
            for ang in angles:
                try:
                    import torchvision.transforms.functional as TVF  # type: ignore
                    pil_rot = TVF.rotate(pil_fit, ang)
                except Exception:
                    pil_rot = pil_fit.rotate(ang)
                img_t = to_tensor(pil_rot).unsqueeze(0).to(device)
                # student's raw mask for ROI
                mask_logits = model.student._forward(img_t, [ds])
                if mask_logits.shape[-2:] != (H, W):
                    mask_logits = F.interpolate(mask_logits, size=(H, W), mode='bilinear', align_corners=False)
                mask_crop, img_crop = wmd_utils.crop_mask_and_image(mask_logits.detach().cpu(), img_t.detach().cpu())
                y_prob = sdecoder(img_crop.to(device), mask_crop.to(device)).squeeze(0)
                cur_bce = F.binary_cross_entropy(y_prob.unsqueeze(0), gt_bits.unsqueeze(0), reduction='mean').item()
                if (best_bce is None) or (cur_bce < best_bce):
                    best_bce = cur_bce
                    best_probs = y_prob
            if best_probs is None:
                best_probs = torch.zeros_like(gt_bits)
            pred_s = (best_probs >= 0.5).float()

            # Teacher via decoder2
            de_bits = decoder2(str(img_path), self._mdecoder, sdecoder)
            if isinstance(de_bits, torch.Tensor):
                if de_bits.dim() == 2 and de_bits.size(1) >= gt_bits.numel():
                    bces = [F.binary_cross_entropy(row.unsqueeze(0), gt_bits.unsqueeze(0), reduction='mean').item() for row in de_bits]
                    import numpy as _np
                    best_idx = int(_np.argmin(_np.asarray(bces))) if len(bces) > 0 else 0
                    pred_t = (de_bits[best_idx] >= 0.5).float().to(device)
                elif de_bits.dim() == 1 and de_bits.numel() >= gt_bits.numel():
                    pred_t = (de_bits[:gt_bits.numel()] >= 0.5).float().to(device)
                else:
                    pred_t = torch.zeros_like(gt_bits)
            else:
                pred_t = torch.zeros_like(gt_bits)

            correct_s += (pred_s == gt_bits).float().sum().item()
            correct_t += (pred_t == gt_bits).float().sum().item()
            total_bits += gt_bits.numel()

        if total_bits > 0:
            self.results.append(dict(student_bit_acc=correct_s / total_bits,
                                     teacher_bit_acc=correct_t / total_bits))

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:  # type: ignore[override]
        if len(results) == 0:
            return dict(student_bit_acc=0.0, teacher_bit_acc=0.0)
        s = sum(r.get('student_bit_acc', 0.0) for r in results) / len(results)
        t = sum(r.get('teacher_bit_acc', 0.0) for r in results) / len(results)
        return dict(student_bit_acc=float(s), teacher_bit_acc=float(t))


