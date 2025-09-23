from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np
try:  # optional, used for black border fill like demo
    from scipy import ndimage as scipy_ndimage  # type: ignore
except Exception:  # pragma: no cover
    scipy_ndimage = None

# Pillow LANCZOS constant to match demo's ImageOps.fit behavior
try:
    if hasattr(Image, 'Resampling'):
        RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    elif hasattr(Image, 'LANCZOS'):
        RESAMPLE_LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]
    else:
        RESAMPLE_LANCZOS = Image.ANTIALIAS  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    RESAMPLE_LANCZOS = None  # type: ignore[assignment]


@HOOKS.register_module()
class BitAccEvalHook(Hook):
    """Evaluate student/teacher bit accuracy with SDecoder on val_dataloader.

    This hook runs after each training epoch. It uses the student's SDecoder
    to decode both student and teacher outputs, compares against the GT bits
    loaded from a labels.json mapping (image file name -> 100-bit string), and
    logs student_bit_acc and teacher_bit_acc.

    Args:
        payload_json (str): Path to labels.json under val directory.
        image_size (tuple[int, int]): Expected (H, W) for SDecoder input.
        max_batches (int | None): If set, only evaluate up to this many
            batches to save time.
    """

    def __init__(self,
                 payload_json: str,
                 image_size: tuple[int, int] = (400, 400),
                 max_batches: Optional[int] = None,
                 mdecoder_ckpt: Optional[str] = None,
                 train_payload_json: Optional[str] = None,
                 student_use_teacher_mask: bool = False) -> None:
        super().__init__()
        self.payload_json = payload_json
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.max_batches = max_batches
        # Lazy-loaded mapping
        self._name_to_bits: Optional[Dict[str, torch.Tensor]] = None
        # Separate train mapping for TrainEval
        self.train_payload_json = train_payload_json
        self._name_to_bits_train: Optional[Dict[str, torch.Tensor]] = None
        # Mdecoder settings
        self.mdecoder_ckpt = mdecoder_ckpt
        self._mdecoder = None  # lazy init
        # If True, use teacher mask (from Mdecoder) as student's mask during eval
        self.student_use_teacher_mask = bool(student_use_teacher_mask)

    def _ensure_mapping(self) -> None:
        if self._name_to_bits is not None:
            return
        with open(self.payload_json, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        mapping: Dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            name = os.path.basename(k)
            if isinstance(v, str):
                bits = torch.tensor([1.0 if ch == '1' else 0.0 for ch in v.strip()], dtype=torch.float32)
            else:
                bits = torch.tensor(v, dtype=torch.float32)
            mapping[name] = bits
        self._name_to_bits = mapping

    def _ensure_mapping_train(self) -> None:
        if self._name_to_bits_train is not None or not self.train_payload_json:
            return
        with open(self.train_payload_json, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        mapping: Dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            name = os.path.basename(k)
            if isinstance(v, str):
                bits = torch.tensor([1.0 if ch == '1' else 0.0 for ch in v.strip()], dtype=torch.float32)
            else:
                bits = torch.tensor(v, dtype=torch.float32)
            mapping[name] = bits
        self._name_to_bits_train = mapping

    def _student_decode_with_student_mask(self, img_path: str, ds, device, model, sdecoder, gt_bits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode bits for one sample using student's own mask with decoder2-like preprocessing.

        Steps: PIL open -> ImageOps.fit(LANCZOS) -> angles [0,3,-3] rotation+EDT fill -> per-angle mask
        from student (or teacher Mdecoder if student_use_teacher_mask) -> ROI crop via WMdemo.utils -> SDecoder -> pick best by BCE.
        """
        H, W = self.image_size
        # Import WMdemo utils lazily
        import sys  # noqa: WPS433
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        wmdemo_dir = os.path.join(project_root, 'WMdemo')
        if wmdemo_dir not in sys.path:
            sys.path.insert(0, wmdemo_dir)
        import utils as wmd_utils  # type: ignore

        # Read and fit image with LANCZOS
        pil = Image.open(str(img_path)).convert('RGB')
        if RESAMPLE_LANCZOS is not None:
            base_pil = ImageOps.fit(pil, (W, H), RESAMPLE_LANCZOS)
        else:
            base_pil = ImageOps.fit(pil, (W, H))
        to_tensor = transforms.ToTensor()

        angles = [0, 3, -3]
        best_bce = None
        best_probs = None

        for ang in angles:
            filled_pil = self._rotate_and_fill_pil(base_pil, ang)
            rot_img = to_tensor(filled_pil).unsqueeze(0).to(device)

            # Mask per angle
            if self.student_use_teacher_mask:
                # Align with decoder2: use Mdecoder raw output (no sigmoid/no binarize) for ROI
                mask_for_crop = self._mdecoder(rot_img).float()
            else:
                # Use student's raw output (logits/prob) directly for ROI, matching decoder2 behavior
                mask_for_crop = model.student._forward(rot_img, [ds])
                if mask_for_crop.shape[-2:] != (H, W):
                    mask_for_crop = F.interpolate(mask_for_crop, size=(H, W), mode='bilinear', align_corners=False)

            # ROI crop on CPU then move back
            mask_cpu = mask_for_crop.detach().cpu()
            img_cpu = rot_img.detach().cpu()
            try:
                mask_crop, img_crop = wmd_utils.crop_mask_and_image(mask_cpu, img_cpu)
            except Exception:
                mask_crop, img_crop = mask_cpu, img_cpu
            mask_crop = mask_crop.to(device)
            img_crop = img_crop.to(device)

            y_prob = sdecoder(img_crop, mask_crop).squeeze(0)
            if gt_bits is None:
                # If GT not available, pick angle 0 (first) implicitly
                if best_probs is None:
                    best_probs = y_prob
                continue
            # Select best by BCE to GT
            cur_bce = F.binary_cross_entropy(y_prob.unsqueeze(0), gt_bits.unsqueeze(0), reduction='mean').item()
            if (best_bce is None) or (cur_bce < best_bce):
                best_bce = cur_bce
                best_probs = y_prob

        if best_probs is None:
            # Fallback zeros
            if gt_bits is not None:
                return torch.zeros_like(gt_bits)
            return torch.zeros(100, device=device)
        # Ensure correct length
        if gt_bits is not None and best_probs.numel() > gt_bits.numel():
            best_probs = best_probs[:gt_bits.numel()]
        return best_probs

    def _eval_on_batch(self, runner, data_batch, batch_idx: int) -> None:
        model = runner.model
        device = next(model.parameters()).device
        # Use train mapping for TrainEval if provided
        self._ensure_mapping_train()
        self._ensure_mapping()

        student = model.student
        sdecoder = getattr(student, 'sdecoder', None)
        if sdecoder is None:
            runner.logger.warning('BitAccEvalHook: student has no sdecoder, skip eval.')
            return
        sdecoder = sdecoder.to(device)
        sdecoder.train()

        # Ensure Mdecoder (HRNet) as in demo
        self._ensure_mdecoder(device, runner)

        H, W = self.image_size

        data_samples: List = data_batch.get('data_samples', [])
        # Strictly rebuild images from file using PIL -> ImageOps.fit -> ToTensor (kept for consistency)
        _ = self._load_fit_images_from_samples(data_samples, device, (H, W), runner)

        # Student: decode using student's mask with same preprocessing as decoder2

        # Debug lines
        try:
            meta0 = getattr(data_samples[0], 'metainfo', {}) if isinstance(data_samples, list) and len(data_samples) > 0 else {}
            img_path0 = None
            if isinstance(meta0, dict) or hasattr(meta0, 'get'):
                try:
                    img_path0 = meta0.get('img_path', None)  # type: ignore[index]
                except Exception:
                    img_path0 = None
            # image tensor no longer retained; skip shape log
            runner.logger.info(
                f"[BitAccEvalHook] debug(train): meta img_path={img_path0}, ori={meta0.get('ori_filename', None) if isinstance(meta0, dict) else None}, filename={meta0.get('filename', None) if isinstance(meta0, dict) else None}")
        except Exception:
            pass

        correct_s = 0
        correct_t = 0
        total = 0
        for i, ds in enumerate(data_samples):
            meta = getattr(ds, 'metainfo', {})
            filename = None
            if isinstance(meta, dict) or hasattr(meta, 'get'):
                try:
                    candidate = meta.get('img_path', None)  # type: ignore[index]
                    if candidate is None:
                        candidate = meta.get('ori_filename', None)  # type: ignore[index]
                    if candidate is None:
                        candidate = meta.get('filename', None)  # type: ignore[index]
                    if candidate is not None:
                        filename = str(candidate)
                except Exception:
                    filename = None
            if (not filename) and hasattr(ds, 'img_path'):
                try:
                    filename = str(getattr(ds, 'img_path'))
                except Exception:
                    filename = None
            if not filename:
                runner.logger.warning('BitAccEvalHook: skip sample without filename in metainfo')
                continue
            name = os.path.basename(str(filename))
            # Prefer train mapping for TrainEval
            bits = None
            if self._name_to_bits_train is not None:
                bits = self._name_to_bits_train.get(name)
            if bits is None and self._name_to_bits is not None:
                bits = self._name_to_bits.get(name)
            runner.logger.info(f"[BitAccEvalHook] debug(train): resolved name={name}, has_gt={bits is not None} (train_mapping={'Y' if self._name_to_bits_train is not None else 'N'})")
            if bits is None:
                continue
            gt_bits = bits.to(device)
            # Resolve image path
            img_path = None
            if isinstance(meta, dict) and meta:
                img_path = meta.get('img_path', None) or meta.get('ori_filename', None) or meta.get('filename', None)
            if (not img_path) and hasattr(ds, 'img_path'):
                img_path = getattr(ds, 'img_path')
            # Student via student's mask
            probs_s = self._student_decode_with_student_mask(str(img_path), ds, device, model, sdecoder, gt_bits)
            pred_s = (probs_s >= 0.5).float()
            # Teacher via decoder2 (unchanged)
            pred_t = None
            try:
                import sys
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                wmdemo_dir = os.path.join(project_root, 'WMdemo')
                if wmdemo_dir not in sys.path:
                    sys.path.insert(0, wmdemo_dir)
                from EMD_maotai import decoder2  # type: ignore
                de_bits = decoder2(str(img_path), self._mdecoder, sdecoder)
                if isinstance(de_bits, torch.Tensor):
                    if de_bits.dim() == 2 and de_bits.size(1) >= gt_bits.numel():
                        bces = [F.binary_cross_entropy(row.unsqueeze(0), gt_bits.unsqueeze(0), reduction='mean').item() for row in de_bits]
                        import numpy as _np
                        best_idx = int(_np.argmin(_np.asarray(bces))) if len(bces) > 0 else 0
                        pred_t = (de_bits[best_idx] >= 0.5).float().to(device)
                    elif de_bits.dim() == 1 and de_bits.numel() >= gt_bits.numel():
                        pred_t = (de_bits[:gt_bits.numel()] >= 0.5).float().to(device)
                if pred_t is None:
                    pred_t = torch.zeros_like(gt_bits)
            except Exception:
                pred_t = torch.zeros_like(gt_bits)
            correct_s += (pred_s == gt_bits).float().sum().item()
            correct_t += (pred_t == gt_bits).float().sum().item()
            total += gt_bits.numel()

        if total > 0:
            runner.logger.info(f"[TrainEval] student_bit_acc={correct_s/total:.4f} teacher_bit_acc={correct_t/total:.4f}")

    @torch.no_grad()
    def after_val_epoch(self, runner, metrics=None, **kwargs) -> None:  # type: ignore[override]
        # Full validation on val_dataloader using payload_json mapping
        if not hasattr(runner, 'val_dataloader') or runner.val_dataloader is None:
            return

        model = runner.model
        device = next(model.parameters()).device
        self._ensure_mapping()

        student = model.student
        sdecoder = getattr(student, 'sdecoder', None)
        if sdecoder is None:
            return
        sdecoder = sdecoder.to(device)
        sdecoder.train()

        # Ensure Mdecoder
        self._ensure_mdecoder(device, runner)

        H, W = self.image_size
        total_bits = 0
        correct_s = 0
        correct_t = 0

        # Hard-coded debug save directory as requested
        debug_root = r'G:\project_WaterMark\WMdemo\workdir\test\train\newMask'
        debug_stu_dir = os.path.join(debug_root, 'stu')
        debug_tea_dir = os.path.join(debug_root, 'tea')
        try:
            os.makedirs(debug_stu_dir, exist_ok=True)
            os.makedirs(debug_tea_dir, exist_ok=True)
        except Exception:
            pass

        # Collect per-sample 100-bit outputs to save as JSON at epoch end
        stu_bits_map: Dict[str, str] = {}
        tea_bits_map: Dict[str, str] = {}

        # Determine if this is the last epoch; only then save mask images and JSON
        is_last_epoch = False
        try:
            current_epoch = int(getattr(runner, 'epoch', -1))
            max_epochs = int(getattr(getattr(runner, 'train_loop', None), 'max_epochs', -1))
            is_last_epoch = (current_epoch + 1) == max_epochs
        except Exception:
            is_last_epoch = False

        for batch_idx, data_batch in enumerate(runner.val_dataloader):
            data_samples: List = data_batch.get('data_samples', [])

            # Student path: use student's pipeline (student mask or teacher mask per flag) with decoder2-like preprocessing
            y_s_rows: List[torch.Tensor] = []
            for i, ds in enumerate(data_samples):
                meta = getattr(ds, 'metainfo', {})
                img_path = None
                if isinstance(meta, dict) and meta:
                    img_path = meta.get('img_path', None) or meta.get('ori_filename', None) or meta.get('filename', None)
                if (not img_path) and hasattr(ds, 'img_path'):
                    img_path = getattr(ds, 'img_path')
                if not img_path:
                    y_s_rows.append(torch.zeros(100, device=device))
                    continue
                name = os.path.basename(str(img_path))
                gt_bits = None if self._name_to_bits is None else self._name_to_bits.get(name, None)
                gt_tensor = gt_bits.to(device) if isinstance(gt_bits, torch.Tensor) else None
                probs_s = self._student_decode_with_student_mask(str(img_path), ds, device, model, sdecoder, gt_tensor)
                y_s_rows.append(probs_s)
            y_s = torch.stack(y_s_rows, dim=0) if len(y_s_rows) > 0 else torch.zeros(len(data_samples), 100, device=device)

            # Teacher bits will be obtained via WMdemo EMD_maotai.decoder2 per-sample below

            # Prepare teacher predictions per sample will be selected by best-rotation later

            for i, ds in enumerate(data_samples):
                meta = getattr(ds, 'metainfo', {})
                filename = None
                if isinstance(meta, dict) or hasattr(meta, 'get'):
                    try:
                        candidate = meta.get('img_path', None)  # type: ignore[index]
                        if candidate is None:
                            candidate = meta.get('ori_filename', None)  # type: ignore[index]
                        if candidate is None:
                            candidate = meta.get('filename', None)  # type: ignore[index]
                        if candidate is not None:
                            filename = str(candidate)
                    except Exception:
                        filename = None
                if (not filename) and hasattr(ds, 'img_path'):
                    try:
                        filename = str(getattr(ds, 'img_path'))
                    except Exception:
                        filename = None
                if not filename:
                    continue
                name = os.path.basename(str(filename))
                bits = None if self._name_to_bits is None else self._name_to_bits.get(name)
                if bits is None:
                    continue
                gt_bits = bits.to(device)
                # student: single-angle pred
                pred_s = (y_s[i] >= 0.5).float()

                # teacher: use WMdemo EMD_maotai.decoder2(img_path, Mdecoder, Sdecoder)
                pred_t = None
                try:
                    import sys
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                    wmdemo_dir = os.path.join(project_root, 'WMdemo')
                    if wmdemo_dir not in sys.path:
                        sys.path.insert(0, wmdemo_dir)
                    from EMD_maotai import decoder2  # type: ignore
                    # Resolve image path
                    img_path = None
                    if isinstance(meta, dict) and meta:
                        img_path = meta.get('img_path', None) or meta.get('ori_filename', None) or meta.get('filename', None)
                    if (not img_path) and hasattr(ds, 'img_path'):
                        img_path = getattr(ds, 'img_path')
                    # Run decoder2
                    de_bits = decoder2(str(img_path), self._mdecoder, sdecoder)
                    # de_bits may be shape [K,100] or [100]
                    if isinstance(de_bits, torch.Tensor):
                        if de_bits.dim() == 2 and de_bits.size(1) >= gt_bits.numel():
                            # choose best row by BCE to GT
                            bces = [F.binary_cross_entropy(row.unsqueeze(0), gt_bits.unsqueeze(0), reduction='mean').item() for row in de_bits]
                            best_idx = int(np.argmin(np.asarray(bces))) if len(bces) > 0 else 0
                            pred_t = (de_bits[best_idx] >= 0.5).float().to(device)
                        elif de_bits.dim() == 1 and de_bits.numel() >= gt_bits.numel():
                            pred_t = (de_bits[:gt_bits.numel()] >= 0.5).float().to(device)
                    if pred_t is None:
                        # Fallback to zeros if unexpected
                        pred_t = torch.zeros_like(gt_bits)
                except Exception:
                    pred_t = torch.zeros_like(gt_bits)
                correct_s += (pred_s == gt_bits).float().sum().item()
                correct_t += (pred_t == gt_bits).float().sum().item()
                total_bits += gt_bits.numel()

                # Save 100-bit outputs to maps (as 0/1 string)
                try:
                    bits_s_str = ''.join(['1' if int(x) == 1 else '0' for x in pred_s.detach().cpu().to(torch.int32).tolist()])
                    bits_t_str = ''.join(['1' if int(x) == 1 else '0' for x in pred_t.detach().cpu().to(torch.int32).tolist()])
                    stu_bits_map[name] = bits_s_str
                    tea_bits_map[name] = bits_t_str
                except Exception:
                    pass

                # Option B: do not save teacher masks; only JSON is produced at last epoch

        if total_bits > 0:
            student_acc = correct_s / total_bits
            teacher_acc = correct_t / total_bits
            runner.logger.info(f"[ValEval] student_bit_acc={student_acc:.4f} teacher_bit_acc={teacher_acc:.4f}")
            # 不再写 MessageHub，不再修改 metrics；由 evaluator 统一产出指标

        if is_last_epoch and not getattr(self, '_json_written', False):
            try:
                with open(os.path.join(debug_root, 'stu.json'), 'w', encoding='utf-8') as f:
                    json.dump(stu_bits_map, f, ensure_ascii=False)
                with open(os.path.join(debug_root, 'tea.json'), 'w', encoding='utf-8') as f:
                    json.dump(tea_bits_map, f, ensure_ascii=False)
                self._json_written = True
            except Exception:
                pass

    def _ensure_mdecoder(self, device, runner=None):
        if self._mdecoder is not None:
            return
        try:
            import sys
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            wmdemo_dir = os.path.join(project_root, 'WMdemo')
            if wmdemo_dir not in sys.path:
                sys.path.insert(0, wmdemo_dir)
            from nets.hrnet import HRnet  # type: ignore
            mdecoder = HRnet(num_classes=1, backbone='model_data/hrnetv2_w48_weights_voc.pth', pretrained=False)
            # Load weights similar to demo
            ckpt_path = self.mdecoder_ckpt if self.mdecoder_ckpt is not None else os.path.join(wmdemo_dir, 'models', 'Mdecoder_210000.pth')
            state = torch.load(ckpt_path, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            mdecoder.load_state_dict(state, strict=False)
            self._mdecoder = mdecoder.to(device).train()
        except Exception as e:
            if runner is not None:
                runner.logger.error(f"[BitAccEvalHook] Failed to init Mdecoder: {repr(e)}")
            raise

    def _load_fit_images_from_samples(self, data_samples: List, device, size_hw: tuple[int, int], runner=None) -> torch.Tensor:
        H, W = size_hw
        imgs: List[torch.Tensor] = []
        to_tensor = transforms.ToTensor()
        for ds in data_samples:
            path = None
            meta = getattr(ds, 'metainfo', {})
            if isinstance(meta, dict):
                path = meta.get('img_path', None) or meta.get('ori_filename', None) or meta.get('filename', None)
            if path is None and hasattr(ds, 'img_path'):
                try:
                    path = getattr(ds, 'img_path')
                except Exception:
                    path = None
            if path is None:
                if runner is not None:
                    runner.logger.warning('[BitAccEvalHook] missing img_path; using zeros image')
                imgs.append(torch.zeros(3, H, W, device=device))
                continue
            try:
                pil = Image.open(str(path)).convert('RGB')
                if RESAMPLE_LANCZOS is not None:
                    pil_fit = ImageOps.fit(pil, (W, H), RESAMPLE_LANCZOS)
                else:
                    pil_fit = ImageOps.fit(pil, (W, H))
                tensor = to_tensor(pil_fit).to(device)
                imgs.append(tensor)
            except Exception as e:
                if runner is not None:
                    runner.logger.warning(f'[BitAccEvalHook] failed to read {path}: {repr(e)}; using zeros')
                imgs.append(torch.zeros(3, H, W, device=device))
        if len(imgs) == 0:
            return torch.zeros(1, 3, H, W, device=device)
        batch = torch.stack(imgs, dim=0)
        return batch

    def _rotate_and_fill_pil(self, pil_img: Image.Image, angle: float) -> Image.Image:
        # Rotate PIL image using torchvision F.rotate to match demo behavior
        try:
            import torchvision.transforms.functional as TVF  # type: ignore
            rotated = TVF.rotate(pil_img, angle)
        except Exception:
            rotated = pil_img.rotate(angle)
        if scipy_ndimage is None:
            return rotated
        try:
            arr = np.array(rotated)
            if arr.ndim == 2:  # gray -> expand to 3 channels
                arr = np.stack([arr, arr, arr], axis=-1)
            is_black = np.all(arr == [0, 0, 0], axis=-1)
            if not np.any(is_black) or np.all(is_black):
                return rotated
            _, (r_idx, c_idx) = scipy_ndimage.distance_transform_edt(is_black, return_indices=True)
            filled = arr.copy()
            filled[is_black] = arr[r_idx[is_black], c_idx[is_black]]
            return Image.fromarray(filled)
        except Exception:
            return rotated

    @torch.no_grad()
    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:  # type: ignore[override]
        # Only run on last train batch of each epoch
        try:
            epoch_len = len(runner.train_dataloader)
        except Exception:
            epoch_len = None
        if epoch_len is None or data_batch is None:
            return
        if (batch_idx + 1) == epoch_len:
            try:
                runner.logger.info(f"[BitAccEvalHook] debug(train): entering eval on last batch idx={batch_idx}, epoch_len={epoch_len}")
                self._eval_on_batch(runner, data_batch, batch_idx)
            except Exception as e:
                # Do not crash training; log and continue
                try:
                    y_types = {}
                    if isinstance(data_batch, dict) and 'inputs' in data_batch:
                        y_types['inputs_type'] = type(data_batch['inputs']).__name__
                    runner.logger.error(f"[BitAccEvalHook] eval failed: {repr(e)} | batch_idx={batch_idx} info={y_types}")
                except Exception:
                    pass
                return


