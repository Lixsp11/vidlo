"""按帧号读的请求规划（design.md 第 2.3 节）：把调用方的帧号列表变成一次连续的字节读取和解码顺序上连续的一段包。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Plan:
    numbers: list[int]  # 调用方每个位置要的帧号，负下标已换算，保持原顺序和重复
    targets: list[int]  # 要解码交出的帧号，不重复，升序
    first: int  # 解码顺序上第一个要送的包：各目标 decode_start 的最小值
    last: int  # 最后一个要送的包：各目标 decode_order 的最大值
    range_start: int  # 一次读取的字节区间 [range_start, range_end)
    range_end: int
    packets: np.ndarray  # [last-first+1, 4] int64，按解码顺序每行 (packet_start, packet_end, 帧号, 是否 GOP 入口)


class VideoIndex:
    """检查过的视频索引，按 numpy 数组保存一份，调用方之后改动原来的 dict 也不受影响。"""

    def __init__(self, index: dict[str, Any], frame_at: list[int]) -> None:
        self.count = len(index["decode_order"])
        self.decode_order = np.array(index["decode_order"], dtype=np.int64)
        self.decode_start = np.array(index["decode_start"], dtype=np.int64)
        self.packet_start = np.array(index["packet_start"], dtype=np.int64)
        self.packet_end = np.array(index["packet_end"], dtype=np.int64)
        self.frame_at = np.array(frame_at, dtype=np.int64)  # 解码顺序第 k 个包属于哪一帧
        self.key_frame = self.decode_start == self.decode_order  # GOP 入口的那一帧


def frame_numbers(frames: Any, count: int) -> list[int]:
    """检查帧号列表并把负下标换算成正的（-1 是最后一帧）。越界报 IndexError；不是整数（含 bool）报 ValueError。"""
    try:
        if isinstance(frames, (str, bytes)):
            raise TypeError
        items = iter(frames)  # 零维 numpy 数组有 __iter__，调用时才报 TypeError
    except TypeError:
        raise ValueError(f"frames must be a sequence of integers, got {frames!r}") from None
    numbers = []
    for f in items:
        if isinstance(f, (bool, np.bool_)) or not isinstance(f, (int, np.integer)):
            raise ValueError(f"frame numbers must be integers, got {f!r}")
        n = int(f)
        if not -count <= n < count:
            raise IndexError(f"frame {n} is out of range for a video with {count} frames")
        numbers.append(n % count)
    return numbers


def plan(index: VideoIndex, numbers: list[int]) -> Plan:
    """numbers 须非空、已由 frame_numbers 检查。"""
    targets = sorted(set(numbers))
    first = int(index.decode_start[targets].min())
    last = int(index.decode_order[targets].max())
    frames = index.frame_at[first:last + 1]
    packets = np.stack([index.packet_start[frames], index.packet_end[frames], frames,
                        index.key_frame[frames].astype(np.int64)], axis=1)
    return Plan(numbers=numbers, targets=targets, first=first, last=last, range_start=int(packets[0, 0]),
                range_end=int(packets[-1, 1]), packets=np.ascontiguousarray(packets))
