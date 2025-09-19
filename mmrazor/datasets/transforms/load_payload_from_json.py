from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from mmrazor.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadPayloadFromJSON:
    """Load 100-bit watermark payload per image from a JSON file.

    JSON format (keys are image file names in the same folder):
        {
          "00001.png": "0101...",  # length 100 string of '0'/'1'
          ...
        }

    Args:
        json_path (str): Absolute or relative path to labels.json.
        filename_key (str): Key in results meta to get file name.
            Defaults to 'ori_filename'. Fallback to basename of
            results['img_path'] if not found.
    """

    def __init__(self, json_path: str, filename_key: str = 'ori_filename') -> None:
        assert isinstance(json_path, str) and json_path, 'json_path must be provided'
        self.json_path = json_path
        self.filename_key = filename_key
        with open(self.json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        # Normalize to name -> np.ndarray[float32] of shape (100,)
        self.name_to_bits: Dict[str, np.ndarray] = {}
        for k, v in raw.items():
            name = os.path.basename(k)
            if isinstance(v, str):
                bits = np.fromiter((1.0 if ch == '1' else 0.0 for ch in v.strip()), dtype=np.float32)
            else:
                bits = np.asarray(v, dtype=np.float32)
            self.name_to_bits[name] = bits

    def __call__(self, results: Dict) -> Dict:
        # Determine image file name
        file_name = None
        if self.filename_key in results:
            file_name = os.path.basename(str(results[self.filename_key]))
        elif 'img_path' in results and results['img_path'] is not None:
            file_name = os.path.basename(str(results['img_path']))
        elif 'img_info' in results and isinstance(results['img_info'], dict):
            file_name = os.path.basename(str(results['img_info'].get('filename', '')))

        if not file_name:
            return results

        bits = self.name_to_bits.get(file_name)
        if bits is None:
            return results

        results['gt_payload'] = bits  # shape (100,), float32 in {0.0,1.0}
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(json_path={self.json_path!r})'


