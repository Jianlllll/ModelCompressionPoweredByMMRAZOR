import argparse
import os
import sys
from typing import Dict

import torch


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith('module.') for k in sd.keys()):
        return sd
    return {k[len('module.'):]: v for k, v in sd.items()}


def _maybe_add_backbone_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith('backbone.') for k in sd.keys()):
        return sd
    return {f'backbone.{k}': v for k, v in sd.items()}


def main():
    parser = argparse.ArgumentParser(description='Convert legacy ckpt to standard state_dict format')
    parser.add_argument('input', help='Path to legacy checkpoint (e.g., Mdecoder_210000.pth)')
    parser.add_argument('output', help='Path to save converted checkpoint')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'Input file not found: {args.input}', file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(args.input, map_location='cpu')

    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
            sd = ckpt['model_state_dict']
        elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            sd = ckpt['state_dict']
        else:
            # assume this is already a raw state_dict-like
            sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    else:
        print('Unsupported checkpoint format.', file=sys.stderr)
        sys.exit(2)

    sd = _strip_module_prefix(sd)
    sd = _maybe_add_backbone_prefix(sd)

    torch.save({'state_dict': sd}, args.output)
    print(f'Converted: {args.input} -> {args.output}\nKeys: {len(sd)}')


if __name__ == '__main__':
    main()
