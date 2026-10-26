"""内部采样器（VideoReader.frames 的 step、fps 参数用的）的选帧结果：EMDownSampler 与 VPF 原代码逐位比较，
StrideSampler 与列表切片比较。不需要 GPU。

VPF 的 vpf/sampler/em.py 只依赖 bitarray 的几个操作，这里用一个按同样语义写的列表实现代替（不安装 bitarray），
只加载 em.py 和 base.py 两个文件，不导入 VPF 的其他部分。结果写进 report.json，输出目录必须不存在。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import vidlo
from vidlo._sampler import EMDownSampler, Sampler, StrideSampler


class ListBitArray:
    """em.py 用到的 bitarray 操作：bitarray(n)、按下标和切片读写、count(value, start, stop)。"""

    def __init__(self, size: int) -> None:
        self._bits = [False] * size

    def __getitem__(self, key: Any) -> Any:
        return self._bits[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, slice):
            value = list(value)
            if len(value) != len(range(*key.indices(len(self._bits)))):
                raise ValueError("slice assignment must keep the length")
        self._bits[key] = value

    def count(self, value: int, start: int, stop: int) -> int:
        return sum(1 for bit in self._bits[start:stop] if bit == bool(value))


def load_vpf_em(vpf_src: Path) -> Any:
    shim = types.ModuleType("bitarray")
    shim.bitarray = ListBitArray  # type: ignore[attr-defined]
    sys.modules["bitarray"] = shim
    for name in ("vpf", "vpf.sampler"):
        package = types.ModuleType(name)
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = package
    modules = {}
    for name in ("base", "em"):
        spec = importlib.util.spec_from_file_location(f"vpf.sampler.{name}", vpf_src / "vpf" / "sampler" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["em"].EMDownSampler


# 帧率组合：b1 的 60→30、常见非整数比例、不降帧、极低目标帧率、小于 2 的源帧率（VPF 对窗口长度取下限 1）
EM_CASES = [(60, 30), (60, 24), (60, 20), (60, 15), (60, 60), (60, 1), (60, 59), (50, 24), (30, 15), (25, 10),
            (59.94, 29.97), (29.97, 23.976), (120, 25), (2, 1), (1.5, 1), (1, 1), (100, 33.3)]
EM_LENGTH = 20000
STRIDE_CASES = [(1, 0), (2, 0), (2, 1), (3, 2), (7, 3), (60, 59)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vpf-src", required=True, type=Path, help="VPF 仓库的 src 目录")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    VpfEM = load_vpf_em(args.vpf_src)

    checks: dict[str, Any] = {}
    for s_fps, t_fps in EM_CASES:
        vpf = VpfEM(s_fps, t_fps)
        expected = [next(vpf) for _ in range(EM_LENGTH)]
        decisions = EMDownSampler(s_fps, t_fps).decisions()
        got = [next(decisions) for _ in range(EM_LENGTH)]
        first_difference = next((i for i, (a, b) in enumerate(zip(expected, got)) if a != b), None)
        checks[f"em_{s_fps}_to_{t_fps}"] = {"passed": got == expected, "kept": sum(got), "first_kept": [i for i, k in enumerate(got) if k][:8],
                                            "first_difference": first_difference}
    # select() 与逐个位置的决定一致；同一个采样器第二次使用时从头开始
    sampler = EMDownSampler(60, 24)
    frames = list(range(5000, 6000)) + [3, 3, 2]
    once, twice = sampler.select(frames), sampler.select(frames)
    vpf = VpfEM(60, 24)
    checks["em_select_matches_vpf"] = {"passed": once == twice == [f for f in frames if next(vpf)]}

    for step, phase in STRIDE_CASES:
        frames = [9, 3, 3, 0, 7, 100, 5, 5, 5, 2, 8] * 13
        got = StrideSampler(step, phase).select(frames)
        checks[f"stride_{step}_{phase}"] = {"passed": got == frames[phase::step]}
    checks["stride_rejects_bad_phase"] = {"passed": raises(ValueError, lambda: StrideSampler(2, 2))}
    checks["stride_rejects_bool"] = {"passed": raises(ValueError, lambda: StrideSampler(True))}
    checks["em_rejects_upsampling"] = {"passed": raises(ValueError, lambda: EMDownSampler(30, 60))}
    checks["em_rejects_non_finite_or_bool"] = {"passed": all(raises(ValueError, lambda v=v: EMDownSampler(*v))
                                                         for v in ((float("inf"), 30), (float("nan"), 30), (60, True), ("60", 30)))}

    class Finite(Sampler):
        def decisions(self):  # type: ignore[no-untyped-def]
            return iter([True, False])

    checks["finite_custom_sampler_rejected"] = {"passed": raises(vidlo.Error, lambda: Finite().select([1, 2, 3]))}

    report = {"vidlo_version": vidlo.__version__, "em_length": EM_LENGTH, "checks": checks,
              "passed": all(c["passed"] for c in checks.values())}
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    for name, check in checks.items():
        print(("PASS " if check["passed"] else "FAIL ") + name + ("" if check["passed"] else "  " + json.dumps(check)[:300]))
    print("ALL PASSED" if report["passed"] else "SOME FAILED")
    return 0 if report["passed"] else 1


def raises(kind: type[BaseException], action: Any) -> bool:
    try:
        action()
    except kind:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
