import os
import sys
from typing import Tuple

import torch


def add_paths() -> None:
    # Ensure we can import configs.student_model
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    cfg_dir = os.path.join(proj_root, 'configs')
    if cfg_dir not in sys.path:
        sys.path.insert(0, cfg_dir)


def build_student() -> torch.nn.Module:
    add_paths()
    from configs.student_model import StudentModel_HRNet_W18  # type: ignore
    model = StudentModel_HRNet_W18(num_classes=1)
    model.eval()
    return model


def build_teacher() -> torch.nn.Module:
    add_paths()
    from configs.teacher_model import TeacherModel_HRNet_W48  # type: ignore
    model = TeacherModel_HRNet_W48(num_classes=1)
    model.eval()
    return model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_profile(model: torch.nn.Module, input_size: Tuple[int, int]) -> Tuple[float, float]:
    h, w = input_size
    x = torch.randn(1, 3, h, w)
    try:
        from thop import profile  # type: ignore
        macs, params = profile(model, inputs=(x,), verbose=False)
        # Return as (MACs, params)
        return float(macs), float(params)
    except Exception as e:
        print(f"[warn] thop not available or failed ({repr(e)}); FLOPs unavailable.")
        return -1.0, float(count_params(model))


def format_count(n: float) -> str:
    units = [(1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')]
    for base, suf in units:
        if abs(n) >= base:
            return f"{n / base:.3f}{suf}"
    return f"{n:.0f}"


def main() -> None:
    print("=== Student HRNet-W18 ===")
    s_model = build_student()
    s_params = count_params(s_model)
    print(f"Params (count): {s_params} ({format_count(s_params)})")
    for size in [(400, 400), (416, 416)]:
        macs, params_from_thop = try_profile(s_model, size)
        if macs > 0:
            print(f"Input {size}: MACs={format_count(macs)}, ~FLOPs={format_count(macs*2)}; Params(thop)={format_count(params_from_thop)}")
        else:
            print(f"Input {size}: MACs/FLOPs unavailable (thop missing).")

    print("\n=== Teacher HRNet-W48 ===")
    t_model = build_teacher()
    t_params = count_params(t_model)
    print(f"Params (count): {t_params} ({format_count(t_params)})")
    for size in [(400, 400), (416, 416)]:
        macs, params_from_thop = try_profile(t_model, size)
        if macs > 0:
            print(f"Input {size}: MACs={format_count(macs)}, ~FLOPs={format_count(macs*2)}; Params(thop)={format_count(params_from_thop)}")
        else:
            print(f"Input {size}: MACs/FLOPs unavailable (thop missing).")


if __name__ == '__main__':
    main()


