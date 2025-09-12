from __future__ import annotations

import os
from typing import List, Optional

import mmcv
import numpy as np
from mmengine.hooks import Hook

from mmrazor.registry import HOOKS


@HOOKS.register_module()
class SavePredMaskHook(Hook):
    """Save predicted outputs as PNG during test/val.

    Priority: if probability map exists (pred_score), save grayscale 0-255.
    Otherwise, save binary mask 0/255 from pred_sem_seg.

    Args:
        out_dir (str): Directory to save masks.
        file_suffix (str): Image suffix to use when saving.
    foreground_value (int): Value to write for class=1 when saving masks (default 255).
    """

    def __init__(
        self,
        out_dir: str = 'work_dirs/pred_masks',
        file_suffix: str = 'png',
        foreground_value: int = 255,
    ) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.file_suffix = file_suffix
        self.fg_val = int(foreground_value)
        os.makedirs(self.out_dir, exist_ok=True)
        self._counter = 0

    def _save_outputs(self, outputs: Optional[List] = None) -> None:
        if not outputs:
            return
        for sample in outputs:
            # Expect mmseg.structures.SegDataSample
            # Try prob first
            if hasattr(sample, 'pred_score') and getattr(sample, 'pred_score') is not None:
                data = sample.pred_score.data
                if hasattr(data, 'detach'):
                    data = data.detach().cpu().numpy()
                arr = np.array(data)
                if arr.ndim == 3:
                    arr = arr.squeeze(0)
                arr = np.clip(arr, 0.0, 1.0)
                out_img = (arr * 255.0).astype(np.uint8)
                save_as_mask = False
            elif hasattr(sample, 'pred_sem_seg'):
                data = sample.pred_sem_seg.data
                if hasattr(data, 'detach'):
                    data = data.detach().cpu().numpy()
                arr = np.array(data)
                if arr.ndim == 3:
                    arr = arr.squeeze(0)
                out_img = np.where(arr > 0, self.fg_val, 0).astype(np.uint8)
                save_as_mask = True
            else:
                continue

            # Resolve output filename
            meta = getattr(sample, 'metainfo', {}) or {}
            img_path = meta.get('img_path') or meta.get('ori_filename')
            if img_path:
                stem = os.path.splitext(os.path.basename(img_path))[0]
            else:
                stem = f'img_{self._counter:06d}'
                self._counter += 1
            suffix = 'prob' if not save_as_mask else 'pred'
            out_path = os.path.join(self.out_dir, f'{stem}_{suffix}.{self.file_suffix}')
            mmcv.imwrite(out_img, out_path)

    # test loop callback
    def after_test_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:  # type: ignore[override]
        self._save_outputs(outputs)

    # val loop callback (optional)
    def after_val_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:  # type: ignore[override]
        self._save_outputs(outputs)
