import argparse
import os
import sys
from typing import Dict

import torch


def pick_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
	"""Pick the inner state_dict from a checkpoint dict, or return tensors dict.

	Supports formats:
	- { 'model_state_dict': {...} }
	- { 'state_dict': {...} }
	- raw tensor dict { str: Tensor }
	"""
	if isinstance(ckpt, dict) and isinstance(ckpt.get('model_state_dict'), dict):
		return ckpt['model_state_dict']  # type: ignore[return-value]
	if isinstance(ckpt, dict) and isinstance(ckpt.get('state_dict'), dict):
		return ckpt['state_dict']  # type: ignore[return-value]
	# fallback: filter tensors
	return {k: v for k, v in ckpt.items() if torch.is_tensor(v)}  # type: ignore[arg-type]


def normalize_key(k: str) -> str:
	"""Normalize one key to the target prefix space 'backbone.'.

	Rules:
	- strip leading 'module.' and 'Mdecoder.'
	- backbone.model.X -> backbone.X
	- model.X -> backbone.X
	- otherwise ensure startswith 'backbone.'
	"""
	nk = k
	if nk.startswith('module.'):
		nk = nk[7:]
	if nk.startswith('Mdecoder.'):
		nk = nk[9:]
	if nk.startswith('backbone.model.'):
		nk = 'backbone.' + nk[len('backbone.model.'):]
	elif nk.startswith('model.'):
		nk = 'backbone.' + nk[len('model.'):]
	elif not nk.startswith('backbone.'):
		nk = 'backbone.' + nk
	return nk


def main():
	parser = argparse.ArgumentParser(description='Normalize HRNet checkpoint keys to backbone.* prefix')
	parser.add_argument('--input', '-i', default='mmrazor/models/Mdecoder_210000.pth', help='input ckpt path')
	parser.add_argument('--output', '-o', default='mmrazor/models/Mdecoder_210000.backbone.pth', help='output ckpt path')
	args = parser.parse_args()

	if not os.path.isfile(args.input):
		print(f'Input not found: {args.input}', file=sys.stderr)
		sys.exit(1)

	ck = torch.load(args.input, map_location='cpu')
	sd = pick_state_dict(ck)
	if not isinstance(sd, dict) or not sd:
		print('No tensor state_dict found in checkpoint', file=sys.stderr)
		sys.exit(2)

	out: Dict[str, torch.Tensor] = {}
	for k, v in sd.items():
		out[normalize_key(k)] = v

	# Save in a standard wrapper
	torch.save({'state_dict': out}, args.output)

	# Print brief summary
	some = list(out.keys())[:8]
	print(f'Wrote {len(out)} keys to {args.output}')
	print('Examples:', some)


if __name__ == '__main__':
	main()
