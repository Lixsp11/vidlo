"""采样器：决定一组帧里保留哪些。VideoReader.frames() 的 step、fps 参数在内部换成这里的采样器；帧率由调用方给出，
索引里不存帧率。

采样器只按位置决定去留：第 0、1、2… 个位置各保留与否，和帧号本身无关。顺序读时位置就是帧号；
按帧号读时位置是帧号列表里的下标。采样器对象本身不保存读取进度，每次读取都重新从第 0 个位置开始，
所以同一个采样器可以同时用于多段录像。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Sequence
from itertools import cycle

from ._native import Error


class Sampler:
    """所有采样器的基类。自定义采样器实现 decisions()。"""

    def decisions(self) -> Iterator[bool]:
        """从第 0 个位置起依次给出每个位置是否保留。顺序读时要能一直给下去，直到录像读完。"""
        raise NotImplementedError

    def select(self, frames: Sequence[int]) -> list[int]:
        """从帧号列表里按位置挑出保留的帧号，保持原来的顺序和重复项。"""
        decisions = self.decisions()
        selected = []
        for position, frame in enumerate(frames):
            keep = next(decisions, None)
            if keep is None:
                raise Error(f"{self!r} gave no decision for position {position} of {len(frames)}")
            if keep:
                selected.append(frame)
        return selected


class StrideSampler(Sampler):
    """每 step 个位置留一个，留第 phase 个：对帧号列表等于 frames[phase::step]。"""

    def __init__(self, step: int, phase: int = 0) -> None:
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError(f"step must be a positive integer, got {step!r}")
        if not isinstance(phase, int) or isinstance(phase, bool) or not 0 <= phase < step:
            raise ValueError(f"phase must be an integer in [0, step), got {phase!r}")
        self.step = step
        self.phase = phase

    def decisions(self) -> Iterator[bool]:
        return cycle([position == self.phase for position in range(self.step)])

    def __repr__(self) -> str:
        return f"StrideSampler(step={self.step}, phase={self.phase})"


class EMDownSampler(Sampler):
    """VPF 的降帧率采样器（vpf/sampler/em.py），支持非整数比例（如 60→24），选帧结果与 VPF 逐位相同。

    在最近 0.5、1、3 秒三个窗口里比较“留下这一帧”和“不留”哪个让实际帧率更接近目标帧率。
    60→30 时留第 1、3、5…帧，与整数步长 StrideSampler(2) 留第 0、2、4…帧不同。
    """

    def __init__(self, source_fps: float, target_fps: float) -> None:
        for name, value in (("source_fps", source_fps), ("fps", target_fps)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number, got {value!r}")
        if target_fps > source_fps:
            raise ValueError(f"target fps {target_fps} is higher than source fps {source_fps}")
        self.source_fps = source_fps
        self.target_fps = target_fps

    def decisions(self) -> Iterator[bool]:
        s_fps, t_fps = self.source_fps, self.target_fps
        # VPF 的 RecurrentBitQueue 只保留最近 maxsize 个结果，等于 deque(maxlen=maxsize)；另记其中留下的个数
        windows = [deque(maxlen=max(1, round(s_fps * seconds) - 1)) for seconds in (0.5, 1.0, 3.0)]
        kept = [0, 0, 0]
        while True:
            # 与 VPF 相同的浮点运算和求和顺序，保证结果逐位相同
            keep_margin, skip_margin = 0, 0
            for i, window in enumerate(windows):
                margin = s_fps * kept[i] - t_fps * (len(window) + 1)
                keep_margin += abs(margin)
                skip_margin += abs(margin + s_fps)
            keep = keep_margin > skip_margin
            for i, window in enumerate(windows):
                if len(window) == window.maxlen:
                    kept[i] -= window[0]
                window.append(keep)
                kept[i] += keep
            yield keep

    def __repr__(self) -> str:
        return f"EMDownSampler(source_fps={self.source_fps}, target_fps={self.target_fps})"
