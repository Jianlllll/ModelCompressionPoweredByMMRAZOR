from __future__ import annotations

from typing import Dict, Literal

import numpy as np

from mmrazor.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BinarizeSegLabel:
    """Binarize segmentation labels in results['gt_seg_map'].

    Place after LoadAnnotations and before PackSegInputs.

    Args:
        mode: 'nonzero' maps non-zero to 1; 'threshold' maps values > threshold to 1.
        threshold: Used when mode='threshold'.
        to_dtype: Output dtype, e.g., 'uint8'.
        invert: If True, invert mapping (e.g., 0->1 for nonzero mode).
    """

    def __init__(
        self,
        mode: Literal['nonzero', 'threshold'] = 'nonzero',
        threshold: int = 127,
        to_dtype: str = 'uint8',
        invert: bool = False,
    ) -> None:
        assert mode in ('nonzero', 'threshold'), (
            "mode must be 'nonzero' or 'threshold'"
        )
        self.mode = mode
        self.threshold = int(threshold)
        self.to_dtype = np.dtype(to_dtype)
        self.invert = bool(invert)

    def transform(self, results: Dict) -> Dict:
        seg_key = 'gt_seg_map'
        if seg_key not in results or results.get(seg_key) is None:
            return results

        seg_np = np.array(results[seg_key])
        if self.mode == 'nonzero':
            bin_seg = (seg_np == 0) if self.invert else (seg_np != 0)
        else:  # 'threshold'
            bin_seg = (seg_np <= self.threshold) if self.invert else (seg_np > self.threshold)

        results[seg_key] = bin_seg.astype(self.to_dtype)
        return results

    def __call__(self, results: Dict) -> Dict:
        return self.transform(results)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(mode={self.mode!r}, '
                f'threshold={self.threshold}, to_dtype={self.to_dtype}, '
                f'invert={self.invert})')
