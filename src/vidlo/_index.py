"""读取并检查两种索引（格式见 docs/index-format.md）。本库只读索引，不生成、不保存。

检查不通过时报 Error；字段齐全但用到了还没实现的格式（如别的编码）时报 UnsupportedError。
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import PurePosixPath
from typing import Any

from ._native import Error, UnsupportedError

VIDEO_INDEX_VERSION = 1
VIDEO_INDEX_FIELDS = ("version", "file_size", "codec", "codec_config", "decode_order", "decode_start", "packet_start", "packet_end")
FRAME_ARRAYS = ("decode_order", "decode_start", "packet_start", "packet_end")
SAMPLE_INDEX_FIELDS = ("video", "frames")
SUPPORTED_CODECS = ("hevc",)


def _load_json(source: str | os.PathLike[str] | dict[str, Any], kind: str) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    with open(source, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise Error(f"{kind} must be a JSON object: {os.fspath(source)}")
    return data


def _check_fields(data: dict[str, Any], expected: tuple[str, ...], kind: str) -> None:
    missing = [k for k in expected if k not in data]
    unknown = [k for k in data if k not in expected]
    if missing or unknown:
        raise Error(f"{kind}: missing fields {missing}, unknown fields {unknown}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_video_index(index: dict[str, Any]) -> None:
    """按第 3.1 节“读取时的检查”逐项核对，不通过就报错。"""
    _check_fields(index, VIDEO_INDEX_FIELDS, "video index")
    if not _is_int(index["version"]):
        raise Error(f"video index: version must be an integer, got {index['version']!r}")
    if index["version"] != VIDEO_INDEX_VERSION:
        raise Error(f"video index: unknown version {index['version']} (this vidlo reads version {VIDEO_INDEX_VERSION})")
    file_size = index["file_size"]
    if not _is_int(file_size) or file_size <= 0:
        raise Error(f"video index: file_size must be a positive integer, got {file_size!r}")
    if not isinstance(index["codec"], str):
        raise Error("video index: codec must be a string")
    if index["codec"] not in SUPPORTED_CODECS:
        raise UnsupportedError(f"video index: codec {index['codec']!r} is not supported yet (only {SUPPORTED_CODECS})")
    if not isinstance(index["codec_config"], str):
        raise Error("video index: codec_config must be a base64 string")
    try:
        config = base64.b64decode(index["codec_config"], validate=True)
    except ValueError as e:  # binascii.Error 是 ValueError 的子类；非 ASCII 字符直接报 ValueError
        raise Error(f"video index: codec_config is not valid base64: {e}") from None
    if not config:
        raise Error("video index: codec_config is empty")

    arrays = {name: index[name] for name in FRAME_ARRAYS}
    for name, values in arrays.items():
        if not isinstance(values, list) or not all(_is_int(v) for v in values):
            raise Error(f"video index: {name} must be a list of integers")
    count = len(arrays["decode_order"])
    if count == 0:
        raise Error("video index: no frames")
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise Error(f"video index: arrays have different lengths {lengths}")

    decode_order = arrays["decode_order"]
    if sorted(decode_order) != list(range(count)):
        raise Error("video index: decode_order must contain each of 0..frames-1 exactly once")
    for frame, (order, start) in enumerate(zip(decode_order, arrays["decode_start"])):
        if not 0 <= start <= order:
            raise Error(f"video index: frame {frame} has decode_start {start} outside [0, decode_order={order}]")

    frame_at = decode_positions(index)
    # decode_start 指向的包必须是一个 GOP 的入口：那个包所属的帧，decode_start 就是它自己的解码位置
    decode_start = arrays["decode_start"]
    for frame, start in enumerate(decode_start):
        entry = frame_at[start]
        if decode_start[entry] != start:
            raise Error(f"video index: frame {frame} has decode_start {start}, but the packet there (frame {entry}) does not start a GOP")

    # 按解码顺序，各包的字节范围递增、互不重叠、不超过文件大小
    previous_end = 0
    for order in range(count):
        frame = frame_at[order]
        start, end = arrays["packet_start"][frame], arrays["packet_end"][frame]
        if not previous_end <= start < end <= file_size:
            raise Error(f"video index: packet of frame {frame} (decode position {order}) has byte range [{start}, {end}) "
                        f"that overlaps the previous packet, is empty, or exceeds file_size {file_size}")
        previous_end = end


def decode_positions(index: dict[str, Any]) -> list[int]:
    """解码顺序第 k 个包属于哪一帧（decode_order 的反向映射）。decode_order 须已检查过是 0..帧数-1 的排列。"""
    frame_at = [0] * len(index["decode_order"])
    for frame, order in enumerate(index["decode_order"]):
        frame_at[order] = frame
    return frame_at


def load_video_index(source: str | os.PathLike[str] | dict[str, Any]) -> dict[str, Any]:
    """读取视频索引（JSON 文件路径或已读入的 dict）并检查。"""
    index = _load_json(source, "video index")
    validate_video_index(index)
    return index


def validate_sample_index(sample: dict[str, Any]) -> None:
    """检查样本索引（第 3.2 节）。帧号是否越界要对照视频索引，在读取时检查。"""
    _check_fields(sample, SAMPLE_INDEX_FIELDS, "sample index")
    video = sample["video"]
    if not isinstance(video, str) or not video:
        raise Error("sample index: video must be a non-empty string")
    # 每段都要有名字：拼到对象存储前缀后面时，空段、. 段会变成另一个对象键
    segments = video.split("/")
    if PurePosixPath(video).is_absolute() or "\\" in video or any(s in ("", ".", "..") for s in segments) or "://" in video:
        raise Error(f"sample index: video must be a relative path under the data root, got {video!r}")
    frames = sample["frames"]
    if not isinstance(frames, list) or not frames or not all(_is_int(v) and v >= 0 for v in frames):
        raise Error("sample index: frames must be a non-empty list of non-negative integers")


def load_sample_index(source: str | os.PathLike[str] | dict[str, Any]) -> dict[str, Any]:
    """读取样本索引（JSON 文件路径或已读入的 dict）并检查。"""
    sample = _load_json(source, "sample index")
    validate_sample_index(sample)
    return sample
