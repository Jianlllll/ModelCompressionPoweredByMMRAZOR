
'''
python extract_student_from_ckpt.py -i best_mIoU_epoch_52.pth -o work_dirs\distill_52\best_mIoU_epoch_52.student_backbone.pth

'''
import argparse
from pathlib import Path
from typing import Dict, Any

import torch


def pick_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Return the inner state_dict from a loaded checkpoint.

    Supports mmengine-style checkpoints with top-level keys like
    { 'meta', 'state_dict', 'message_hub' } or raw tensor dicts.
    """
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]  # type: ignore[return-value]
        # Fallback: collect tensor-like entries
        return {k: v for k, v in ckpt.items() if hasattr(v, "shape")}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract student weights from a combined KD checkpoint and map\n"
            "'architecture.backbone.*' -> 'backbone.*'; drop all 'teacher.*'."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="input checkpoint path")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="output checkpoint path for student-only weights",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    ck = torch.load(str(in_path), map_location="cpu")
    sd = pick_state_dict(ck)
    if not sd:
        raise SystemExit("No tensor state_dict found in checkpoint.")

    out: Dict[str, torch.Tensor] = {}
    pre = "architecture.backbone."
    for k, v in sd.items():
        # Skip any teacher-side weights entirely
        if k.startswith("teacher."):
            continue
        # Map student stored as 'architecture.backbone.*' to 'backbone.*'
        if k.startswith(pre):
            nk = "backbone." + k[len(pre) :]
            out[nk] = v

    if not out:
        raise SystemExit(
            "No 'architecture.backbone.*' keys found. Inspect your checkpoint structure."
        )

    torch.save({"state_dict": out}, str(out_path))
    sample = list(out.items())[:10]
    print(f"Saved {len(out)} student keys -> {out_path}")
    for i, (k, w) in enumerate(sample):
        print(i, k, tuple(w.shape))


if __name__ == "__main__":
    main()
