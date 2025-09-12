from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageOps

from mmrazor.registry import TRANSFORMS


def _get_lanczos():
    if hasattr(Image, 'Resampling'):
        return Image.Resampling.LANCZOS
    if hasattr(Image, 'LANCZOS'):
        return Image.LANCZOS
    return Image.BICUBIC


@TRANSFORMS.register_module()
class CenterFitResize:
    """Use PIL ImageOps.fit with LANCZOS to center-fit to target size.

    Keeps behavior aligned with the provided standalone script.

    Notes:
    - mmseg pipeline keeps images in BGR numpy before data_preprocessor.
      Here we convert BGR->RGB for PIL processing, then convert back RGB->BGR
      to keep the rest of pipeline consistent.
    - Masks use nearest interpolation, kept as uint8.
    """

    def __init__(self, size: Tuple[int, int] = (400, 400)) -> None:
        self.target_w, self.target_h = int(size[0]), int(size[1])
        assert self.target_w > 0 and self.target_h > 0
        self._resample = _get_lanczos()

    def _process_image(self, img_bgr: np.ndarray) -> np.ndarray:
        # BGR -> RGB
        rgb = img_bgr[..., ::-1]
        pil = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
        fitted = ImageOps.fit(pil, (self.target_w, self.target_h), method=self._resample)
        out_rgb = np.array(fitted)
        # RGB -> BGR
        out_bgr = out_rgb[..., ::-1]
        return out_bgr

    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        # Ensure single channel
        if mask.ndim == 3:
            mask = mask[..., 0]
        pil = Image.fromarray(mask.astype(np.uint8), mode='L')
        fitted = ImageOps.fit(pil, (self.target_w, self.target_h), method=Image.NEAREST)
        out = np.array(fitted).astype(np.uint8)
        return out

    def transform(self, results: Dict) -> Dict:
        if 'img' in results and results['img'] is not None:
            results['img'] = self._process_image(results['img'])
            results['img_shape'] = results['img'].shape
        if 'gt_seg_map' in results and results['gt_seg_map'] is not None:
            results['gt_seg_map'] = self._process_mask(np.array(results['gt_seg_map']))
        return results

    def __call__(self, results: Dict) -> Dict:
        return self.transform(results)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size=({self.target_w},{self.target_h}), method=LANCZOS)'
