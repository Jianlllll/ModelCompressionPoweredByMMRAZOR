import re
import sys
import torch
from collections import defaultdict

PREFIX = r"(?:(?:backbone\.)?(?:model\.)?)?"
PAT_STAGE = re.compile(rf"^{PREFIX}stage(?P<stage>[0-9]+)\.(?P<module>[0-9]+)\.")
PAT_BRANCH_BLOCK = re.compile(rf"^{PREFIX}stage(?P<stage>[0-9]+)\.(?P<module>[0-9]+)\.branches\.(?P<branch>[0-9]+)\.(?P<block>[0-9]+)\.")
PAT_TRANSITION = re.compile(rf"^{PREFIX}transition(?P<stage>[0-9]+)\.(?P<to_branch>[0-9]+)\.")
PAT_LAST = re.compile(rf"^{PREFIX}last_layer\.(?P<idx>[0-9]+)\.")


def main(pth_path: str):
    ckpt = torch.load(pth_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            sd = ckpt['state_dict']
        elif 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
            sd = ckpt['model_state_dict']
        else:
            sd = ckpt
    else:
        sd = ckpt

    # strip module. prefix (DataParallel)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('module.'):
            new_sd[k[len('module.'):]] = v
        else:
            new_sd[k] = v
    sd = new_sd

    stages = defaultdict(lambda: defaultdict(set))  # stage -> module -> set(keys)
    branches = defaultdict(lambda: defaultdict(set))  # stage -> module -> set(branch idx)
    blocks = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))  # stage -> module -> branch -> set(block idx)
    transitions = defaultdict(set)  # stage -> set(to_branch)

    for k in sd.keys():
        m = PAT_STAGE.match(k)
        if m:
            s = int(m.group('stage'))
            mod = int(m.group('module'))
            stages[s][mod].add(k)
        m2 = PAT_BRANCH_BLOCK.match(k)
        if m2:
            s = int(m2.group('stage'))
            mod = int(m2.group('module'))
            br = int(m2.group('branch'))
            bl = int(m2.group('block'))
            branches[s][mod].add(br)
            blocks[s][mod][br].add(bl)
        m3 = PAT_TRANSITION.match(k)
        if m3:
            s = int(m3.group('stage'))
            to_b = int(m3.group('to_branch'))
            transitions[s].add(to_b)

    print('== HRNet checkpoint structure summary ==')
    for s in sorted(stages.keys()):
        mods = sorted(stages[s].keys())
        print(f'stage{s}: num_modules={len(mods)} modules={mods}')
        for mod in mods:
            brs = sorted(branches[s][mod])
            num_blocks_each = {b: (max(blocks[s][mod][b]) + 1) if blocks[s][mod][b] else 0 for b in brs}
            print(f'  module {mod}: num_branches={len(brs)} branches={brs} blocks_per_branch={num_blocks_each}')
    for s in sorted(transitions.keys()):
        tb = sorted(transitions[s])
        print(f'transition{s}: has_to_branches={tb} (count={len(tb)})')

    # last_layer channels check
    last_weights = [k for k in sd.keys() if PAT_LAST.match(k)]
    if last_weights:
        print(f'last_layer params found: {len(last_weights)} (examples: {last_weights[:5]})')
    else:
        print('last_layer not found in ckpt')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python tools/inspect_hrnet_ckpt.py <path_to_ckpt.pth>')
        sys.exit(1)
    main(sys.argv[1])
