import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

pathh = Path(r"G:\project_WaterMark\mmrazor\work_dirs\distill_52\best_mIoU_epoch_52.backbone.pth") 
# G:\project_WaterMark\WMdemo\models\Mdecoder_210000.pth


def pick_state_dict(ck: Dict) -> Dict[str, torch.Tensor]:
    if isinstance(ck, dict):
        for k in ("state_dict", "model_state_dict"):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]  # type: ignore[return-value]
        return {k: v for k, v in ck.items() if hasattr(v, "shape")}
    return {}


def detect_root_prefix(keys: Iterable[str]) -> Optional[str]:
    """Detect the base prefix for HRNet backbone keys.

    Try common patterns: '', 'backbone.', 'architecture.backbone.', 'teacher.backbone.'.
    """
    candidates = [
        "",
        "backbone.",
        "architecture.backbone.",
        "teacher.backbone.",
        "student.backbone.",
    ]
    keys_list = list(keys)
    for pre in candidates:
        if any(k.startswith(pre + "stage2.") for k in keys_list) or any(
            k.startswith(pre + "layer1.") for k in keys_list
        ):
            return pre
    return None


def list_indices(keys: List[str], base: str, slot: str) -> List[int]:
    # find integer indices directly after base+slot+
    prefix = base + slot
    idxs = set()
    for k in keys:
        if k.startswith(prefix):
            rest = k[len(prefix) :]
            parts = rest.split(".")
            if parts and parts[0].isdigit():
                idxs.add(int(parts[0]))
    return sorted(list(idxs))


def blocks_per_branch(keys: List[str], base: str) -> Dict[int, int]:
    """Count number of blocks in each branch of a module.

    Supports both 'blocks' and 'layers' naming, and nested patterns like 'blocks.0.0.*'.
    """
    bidxs = list_indices(keys, base, "branches.")
    out: Dict[int, int] = {}
    for b in bidxs:
        blk_idxs: set[int] = set()
        for family in ("blocks", "layers"):
            prefix = f"{base}branches.{b}.{family}."
            for k in keys:
                if k.startswith(prefix):
                    rest = k[len(prefix):]
                    first = rest.split(".", 1)[0]
                    if first.isdigit():
                        blk_idxs.add(int(first))
        if blk_idxs:
            out[b] = len(sorted(blk_idxs))
    return out


def infer_branch_channels(keys: List[str], sd: Dict[str, torch.Tensor], base: str, b: int) -> Optional[int]:
    """Infer out channels of a branch by probing common param names.

    Scans keys to build candidates and picks the first existing tensor.
    """
    families = ("blocks", "layers")
    cand_suffix = (
        "bn3.weight", "bn2.weight", "bn1.weight",
        "norm3.weight", "norm2.weight", "norm1.weight",
        "conv3.weight", "conv2.weight", "conv1.weight",
        "downsample.1.weight",
    )
    for fam in families:
        # try first block index 0 by default, but if nested like 0.0 also accept
        for k in keys:
            # fast-path filter by prefix
            prefix = f"{base}branches.{b}.{fam}."
            if not k.startswith(prefix):
                continue
            # we want keys like ...{fam}.<blk>(.<subblk>)?.<name>
            rest = k[len(prefix):].split(".")
            if not rest or not rest[0].isdigit():
                continue
            # assemble candidate heads for block 0 (or any found block)
            blk0 = rest[0]
            head1 = f"{base}branches.{b}.{fam}.{blk0}."
            head2 = f"{base}branches.{b}.{fam}.{blk0}.0."
            for head in (head1, head2):
                for suf in cand_suffix:
                    cand = head + suf
                    if cand in sd:
                        try:
                            return int(sd[cand].shape[0])
                        except Exception:
                            return None
        # if loop exhausts, continue to next family
    return None


def summarize_stage(keys: List[str], sd: Dict[str, torch.Tensor], root: str, stage: int) -> None:
    base = f"{root}stage{stage}."
    modules = list_indices(keys, base, "")
    print(f"stage{stage}: num_modules={len(modules)} modules={modules}")
    for m in modules:
        mbase = f"{base}{m}."
        bs = list_indices(keys, mbase, "branches.")
        bp = blocks_per_branch(keys, mbase)
        # infer branch widths using sd
        widths = {b: infer_branch_channels(keys, sd=sd, base=mbase, b=b) for b in bs}
        bpb_str = ", ".join(
            f"{b}: blocks={bp.get(b, '?')}, C={widths.get(b, '?')}" for b in bs
        )
        print(f"  module {m}: num_branches={len(bs)} branches={bs} {bpb_str}")


def summarize_transitions(keys: List[str], root: str) -> None:
    for t in (1, 2, 3):
        base = f"{root}transition{t}."
        bs = list_indices(keys, base, "")
        if bs:
            print(f"transition{t}: has_to_branches={bs} (count={len(bs)})")


def summarize_last_layer(sd: Dict[str, torch.Tensor], root: str) -> None:
    ll = [k for k in sd if k.startswith(root + "last_layer.")]
    if not ll:
        print("last_layer params found: 0")
        return
    ll.sort()
    print(f"last_layer params found: {len(ll)} (examples: {[ll[0], ll[1] if len(ll)>1 else ll[0]]})")
    # Print a few shapes
    shown = 0
    for k in ll:
        if shown >= 5:
            break
        print("  ", k, tuple(sd[k].shape))
        shown += 1


def main():
    # 支持命令行：python tools/tmp_KeysNames.py <ckpt_path>
    if len(sys.argv) > 1 and sys.argv[1].strip():
        in_path = Path(sys.argv[1])
    else:
        # 兼容旧用法：如需改路径请直接修改这里
        in_path = pathh

    ck = torch.load(str(in_path), map_location="cpu")
    sd = pick_state_dict(ck)
    print("total keys in state_dict:", len(sd))
    if not sd:
        return

    keys = list(sd.keys())
    rp = detect_root_prefix(keys)
    if rp is None:
        print("detected root prefix: <none>")
    else:
        print(f"detected root prefix: '{rp}'")
    root = rp or ""

    # Stem / layer1 quick check
    for name in ("conv1.weight", "conv2.weight", "layer1.0.conv1.weight"):
        k = root + name
        if k in sd:
            print("stem:", name, tuple(sd[k].shape))

    summarize_transitions(keys, root)
    for st in (2, 3, 4):
        summarize_stage(keys, sd, root, st)

    summarize_last_layer(sd, root)


if __name__ == "__main__":
    main()