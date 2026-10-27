"""公共读取接口，用法见 docs/api.md。

已实现：本地、R2（s3://）、OSS（oss://）的 MKV 顺序读，输出裁剪、缩放后的 RGB 或原生 NV12，可按步长或帧率选帧；
按视频索引从本地文件、R2、OSS 按帧号读（get_frames_at、get_frame_at、get_frames_in_range、下标与切片、read_sample）。
架构里其余的参数和函数都保留，用到时报 UnsupportedError。
"""

from __future__ import annotations

import base64
import os
import weakref
from dataclasses import dataclass
from itertools import islice, repeat
from typing import Any, Iterator

import numpy as np

from . import _hevc, _index, _native, _planner, _storage
from ._native import Error, UnsupportedError
from ._sampler import EMDownSampler, Sampler, StrideSampler

_COLORSPACES = ("bt601", "bt709", "bt2020")
_INTERPOLATIONS = ("nearest_exact", "linear", "cubic", "area")
_FORMATS = ("rgb24", "nv12")
_REMOTE_SCHEMES = ("s3://", "oss://")
# 码流的 matrix_coefficients（H.265 表 E.5）→ CV-CUDA 能用的矩阵
_SOURCE_COLORSPACES = {1: "bt709", 5: "bt601", 6: "bt601", 9: "bt2020"}
_SELECTION_CHUNK = 4096  # 每次交给 C++ 的选帧决定个数
# 按帧号读时一次请求的上限：要解码的帧数（解码顺序上从最早的 GOP 入口到最后一个所需的包）与要读的字节数。帧号很分散的
# 请求不自动拆开，超了在读取前报错。默认值来自实测（H200，1280x720 的 b1 录像）：单进程约 0.8 ms 解一帧、16 个进程共用
# 一块卡时约 2.1 ms，2,000 帧约 1.6～4.3 s；128 MiB 在 OSS 内网测到的最慢单流（约 9 MB/s）下约 15 s，不超过 30 s 的读取时限
MAX_DECODE_FRAMES = 2000
MAX_READ_BYTES = 128 << 20
# 按帧号读时不读文件头，没有容器的颜色信息：按“容器没写”处理，只看 SPS
_NO_CONTAINER_COLOR = {"range": 0, "matrix": 2, "chroma_location": 0}


@dataclass(frozen=True)
class Frame:
    """一帧画面和它在录像里的位置。format="rgb24" 时 image 形状为 [高, 宽, 3]；
    format="nv12" 时是 NV12，形状为 [高×3/2, 宽]。decode_order 到 packet_end 四项是写视频索引用的。"""

    image: np.ndarray
    number: int  # 显示顺序中的帧号，从 0 开始，没选中的帧也计数
    pts: int | None  # 录像里的原始 PTS，乘 metadata.time_base 是秒；按帧号读时不读文件头、索引也不存时间戳，为 None
    key_frame: bool
    decode_status: int  # cuvidGetDecodeStatus 的结果，交出的帧都是成功
    decode_order: int  # 这帧的包在解码顺序中排第几
    decode_start: int  # 要正确解出这帧须从解码顺序的第几个包开始送
    packet_start: int  # 这帧的压缩数据在文件里的起止字节，不含 MKV 块头
    packet_end: int
    colorspace: str | None = None  # RGB 时实际用的矩阵
    interpolation: str | None = None  # 实际用的插值；没缩放或 NV12 时为 None


_FRAME_FIELDS = ("number", "pts", "key_frame", "decode_status", "decode_order", "decode_start", "packet_start", "packet_end",
                 "colorspace", "interpolation")


@dataclass(frozen=True)
class FrameBatch:
    """按帧号读一次交出的帧，按请求的顺序排列，重复的帧号各占一个位置（数据各自独立）。images 在 RGB 时为
    [帧数, 高, 宽, 3]，NV12 时为 [帧数, 高×3/2, 宽]；其余属性与 Frame 同名，每项是按帧排列的列表。"""

    images: np.ndarray
    number: list[int]
    pts: list[int | None]
    key_frame: list[bool]
    decode_status: list[int]
    decode_order: list[int]
    decode_start: list[int]
    packet_start: list[int]
    packet_end: list[int]
    colorspace: list[str | None]
    interpolation: list[str | None]


@dataclass(frozen=True)
class VideoMetadata:
    """录像信息。画面尺寸和颜色标记从码流的 SPS 读出。顺序读时打开就读文件头，全部得到；container_ 开头的三项是 MKV 容器
    另记的颜色信息（FFmpeg 的枚举值），打开时与 SPS 核对。按帧号读时不读文件头，SPS 来自视频索引的 codec_config，
    container、time_base 和 container_ 开头的三项为 None。"""

    container: str | None  # libavformat 的格式名，如 "matroska,webm"
    codec: str
    codec_config: bytes  # 文件头里的解码参数（MKV 的 hvcC），写视频索引要用
    time_base: tuple[int, int] | None  # (分子, 分母)
    file_size: int
    width: int  # 显示区域的尺寸，不随 crop、width、height 参数改变
    height: int
    full_range: bool
    matrix_coefficients: int  # H.265 表 E.5 的编号：1 为 BT.709，2 为未标注
    chroma_location: int  # ChromaLocType：0 为 type 0
    container_range: int | None  # AVColorRange：0 未指定，1 limited，2 full
    container_matrix: int | None  # AVColorSpace，编号同 matrix_coefficients
    container_chroma_location: int | None  # AVChromaLocation：0 未指定，1 起依次是 type 0 到 5
    etag: str | None  # 远程对象的 ETag，读取期间用它确认对象没变；本地文件为 None


def _color_conflicts(sps: _hevc.Sps, container: dict[str, int]) -> list[str]:
    """SPS 与容器（MKV Colour 元素）都明确写了、但写得不一样的颜色标记。只看影响 RGB 转换的范围、矩阵和色度位置；
    一方没写时不算冲突，也不拿容器的值代替 SPS 按标准推断的值。容器的值是 FFmpeg 的枚举：范围 1 limited、2 full；
    矩阵编号同 matrix_coefficients，2 为未指定；色度位置 0 为未指定，1 起依次是 type 0 到 5。"""
    conflicts = []
    if container["range"] in (1, 2) and sps.video_signal_type_present and (container["range"] == 2) != bool(sps.video_full_range_flag):
        conflicts.append(f"color range (container {'full' if container['range'] == 2 else 'limited'}, "
                         f"SPS {'full' if sps.video_full_range_flag else 'limited'})")
    if container["matrix"] != 2 and sps.matrix_coefficients != 2 and container["matrix"] != sps.matrix_coefficients:
        conflicts.append(f"matrix coefficients (container {container['matrix']}, SPS {sps.matrix_coefficients})")
    if (container["chroma_location"] != 0 and sps.chroma_loc_info_present
            and container["chroma_location"] - 1 != sps.chroma_sample_loc_type_top_field):
        conflicts.append(f"chroma location (container type {container['chroma_location'] - 1}, "
                         f"SPS type {sps.chroma_sample_loc_type_top_field})")
    return conflicts


def _interpolation(requested: str | None, source: tuple[int, int], target: tuple[int, int]) -> str:
    """实际用的插值：没指定时，有一个方向缩小就用面积插值（缩小时没有摩尔纹），否则用三次插值。nearest_exact 取半像素中心：
    输出第 i 个像素取源的 floor((i + 0.5) × 源长 / 目标长)，同 PIL、cv2.INTER_NEAREST_EXACT，不同于 cv2.INTER_NEAREST。

    CV-CUDA 0.19 的面积插值没有放大分支：放大方向上，输出像素对应的源区间整个落在一个源像素内部时，这个像素被区间两端
    各算一次，权重和大于 1，结果偏大到接近 255；1.5、2 倍时每个区间都碰到像素边界，所以与 OpenCV 一致，其他倍数差到
    一两百级（CV-CUDA 0.19 的 cvcuda/cuda_tools/InterpolationWrap.hpp 里 AREA 的区间计算）。
    所以面积插值遇到放大的方向时报错，包括一个方向缩小、另一个方向放大；不自动换成别的插值，以免悄悄改掉调用方要的算法。
    """
    shrinks = target[0] < source[0] or target[1] < source[1]
    enlarges = target[0] > source[0] or target[1] > source[1]
    method = requested or ("area" if shrinks else "cubic")
    if method == "area" and enlarges:
        raise UnsupportedError(f"area interpolation for {source[0]}x{source[1]} -> {target[0]}x{target[1]} enlarges in one direction, "
                               "which is not supported yet (CV-CUDA 0.19 AREA is only reliable when shrinking); "
                               "pass interpolation='linear' or 'cubic'")
    return method


def _check_ints(name: str, value: Any, count: int, positive_from: int) -> tuple[int, ...]:
    if (not isinstance(value, (tuple, list)) or len(value) != count
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)):
        raise ValueError(f"{name} must be {count} integers, got {value!r}")
    if any(v < 0 for v in value) or any(v <= 0 for v in value[positive_from:]):
        raise ValueError(f"{name} has negative or zero values: {value!r}")
    return tuple(value)


def _check_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _sampler(step: int | None, fps: float | None, source_fps: float | None) -> Sampler | None:
    """frames() 的选帧参数换成内部的采样器：step 为整数步长，fps 与 source_fps 为 VPF 的按帧率降采样；都不给时保留全部帧。"""
    if step is not None:
        if fps is not None or source_fps is not None:
            raise ValueError("step cannot be combined with fps/source_fps")
        return StrideSampler(_check_positive_int("step", step))
    if fps is None and source_fps is None:
        return None
    if fps is None or source_fps is None:
        raise ValueError("fps and source_fps must be given together")
    return EMDownSampler(source_fps, fps)


class VideoReader:
    """打开一段录像读帧。参数在打开前检查，打开时只读文件头；只在一个线程里使用。

    source 为本地路径、s3://桶/路径（R2 等 S3 兼容存储）或 oss://桶/路径。crop 为 (x, y, 宽, 高)，坐标相对于显示区域，
    四个数都必须是偶数（每 2×2 像素共用一份色度）；width、height 为输出尺寸，没给的一边保持裁剪后的尺寸。colorspace 是
    YUV 换成 RGB 用的矩阵，None 按录像标记。format="nv12" 是解码器的原始输出，不裁剪、不缩放、不换颜色。index 为视频
    索引：给了就只按帧号读，不读文件头。storage_options 是交给存储 SDK 的连接参数，见 _storage.py。max_decode_frames、
    max_read_bytes 是按帧号读时一次请求要解码的帧数与要读的字节数的上限，默认 MAX_DECODE_FRAMES、MAX_READ_BYTES。
    """

    def __init__(
        self,
        source: str | os.PathLike[str],
        *,
        device: int = 0,
        crop: tuple[int, int, int, int] | None = None,
        width: int | None = None,
        height: int | None = None,
        interpolation: str | None = None,
        colorspace: str | None = None,
        format: str = "rgb24",
        index: Any = None,
        storage_options: dict[str, Any] | None = None,
        max_decode_frames: int | None = None,
        max_read_bytes: int | None = None,
    ) -> None:
        if not isinstance(device, int) or isinstance(device, bool) or device < 0:
            raise ValueError(f"device must be a non-negative integer, got {device!r}")
        if crop is not None:
            crop = _check_ints("crop", crop, 4, 2)
            if any(v % 2 for v in crop):
                raise ValueError(f"crop values must be even (4:2:0 chroma is shared by 2x2 pixels), got {crop!r}")
        if width is not None:
            _check_positive_int("width", width)
        if height is not None:
            _check_positive_int("height", height)
        if interpolation is not None and interpolation not in _INTERPOLATIONS:
            raise ValueError(f"interpolation must be None or one of {_INTERPOLATIONS}, got {interpolation!r}")
        if colorspace is not None and colorspace not in _COLORSPACES:
            raise ValueError(f"colorspace must be None or one of {_COLORSPACES}, got {colorspace!r}")
        if format not in _FORMATS:
            raise ValueError(f"format must be one of {_FORMATS}, got {format!r}")
        if format == "nv12" and any(v is not None for v in (crop, width, height, interpolation, colorspace)):
            raise ValueError("format='nv12' is the decoder's raw output; crop, width, height, interpolation and colorspace do not apply")
        if index is None and (max_decode_frames is not None or max_read_bytes is not None):
            raise ValueError("max_decode_frames and max_read_bytes only apply when reading by frame number (index=...)")
        self._max_decode_frames = MAX_DECODE_FRAMES if max_decode_frames is None else _check_positive_int("max_decode_frames", max_decode_frames)
        self._max_read_bytes = MAX_READ_BYTES if max_read_bytes is None else _check_positive_int("max_read_bytes", max_read_bytes)
        self._device = device
        self._crop = crop
        self._width = width
        self._height = height
        self._interpolation = interpolation
        self._colorspace = colorspace
        self._format = format
        self._started = False
        self._closed = False
        self._failed = False  # 按帧号读出过错：之后的请求一律报错，包括空请求
        self._index: _planner.VideoIndex | None = None
        self._download = None
        self._range: _storage.RangeReader | None = None  # 按帧号读对象存储时取区间
        if index is None:
            self._open_sequential(source, storage_options)
        else:
            self._open_indexed(source, index, storage_options)

    def _open_sequential(self, source: str | os.PathLike[str], storage_options: dict[str, Any] | None) -> None:
        """顺序读：读文件头，建解码会话。"""
        download = None
        if isinstance(source, str) and "://" in source:
            if not source.startswith(_REMOTE_SCHEMES):
                raise ValueError(f"unknown source scheme in {source!r}; use a local path, s3:// or oss://")
            download = _storage.open_download(source, storage_options)
            name, data = source, download.input
        else:
            if storage_options is not None:
                raise ValueError("storage_options only apply to s3:// and oss:// sources")
            name = os.fspath(source)
            if not os.path.isfile(name):
                raise FileNotFoundError(name)
            data = _native.FileInput(name)
        native = None
        try:
            native = _native.SequentialDecoder(data, name)
            info = native.info()
            sps = _hevc.stream_sps(info["codec_config"])
            self._spec = self._output_spec(sps, info["container_color"])
            native.start(self._device, self._spec)
        except BaseException:
            if download is not None:
                download.close()
            if native is not None:
                native.close()
            raise
        left, top, right, bottom = sps.display_area
        color = info["container_color"]
        self._native = native
        self._metadata = VideoMetadata(
            container=info["container"], codec=info["codec"], codec_config=info["codec_config"], time_base=tuple(info["time_base"]),
            file_size=info["file_size"], width=right - left, height=bottom - top, full_range=bool(sps.video_full_range_flag),
            matrix_coefficients=sps.matrix_coefficients, chroma_location=sps.chroma_sample_loc_type_top_field,
            container_range=color["range"], container_matrix=color["matrix"], container_chroma_location=color["chroma_location"],
            etag=None if download is None else download.etag)
        self._download = download
        if download is not None:
            # 没有 close 就丢掉时也要让下载线程停下，否则它会一直等着往满了的队列里推数据
            weakref.finalize(self, download.cancel)

    def _open_indexed(self, source: str | os.PathLike[str], index: Any, storage_options: dict[str, Any] | None) -> None:
        """按帧号读：只读视频索引，不读文件头；SPS 取自索引的 codec_config。第一次 get_frames_at 时才初始化 CUDA。
        对象存储在打开时不发请求，对象在不在、大小与索引是否相符，第一次读取时核对。"""
        data = _index.load_video_index(index)
        file_input = None
        if isinstance(source, str) and "://" in source:
            if not source.startswith(_REMOTE_SCHEMES):
                raise ValueError(f"unknown source scheme in {source!r}; use a local path, s3:// or oss://")
            self._range = _storage.open_range(source, storage_options, data["file_size"])
        else:
            if storage_options is not None:
                raise ValueError("storage_options only apply to s3:// and oss:// sources")
            name = os.fspath(source)
            if not os.path.isfile(name):
                raise FileNotFoundError(name)
            file_input = _native.FileInput(name)
            if file_input.size != data["file_size"]:
                raise Error(f"{name} has {file_input.size} bytes but the video index says {data['file_size']}: "
                            "the index is for another file")
        codec_config = base64.b64decode(data["codec_config"])
        sps = _hevc.stream_sps(codec_config)
        self._spec = self._output_spec(sps, _NO_CONTAINER_COLOR)
        self._native = _native.IndexedDecoder(file_input, codec_config, self._device, self._spec)
        self._index = _planner.VideoIndex(data, _index.decode_positions(data))
        left, top, right, bottom = sps.display_area
        if self._spec.rgb:
            self._image_shape = (self._spec.height or self._spec.crop_height or bottom - top,
                                 self._spec.width or self._spec.crop_width or right - left, 3)
        else:
            self._image_shape = ((bottom - top) * 3 // 2, right - left)
        self._metadata = VideoMetadata(
            container=None, codec=data["codec"], codec_config=codec_config, time_base=None, file_size=data["file_size"],
            width=right - left, height=bottom - top, full_range=bool(sps.video_full_range_flag),
            matrix_coefficients=sps.matrix_coefficients, chroma_location=sps.chroma_sample_loc_type_top_field,
            container_range=None, container_matrix=None, container_chroma_location=None, etag=None)

    def _output_spec(self, sps: _hevc.Sps, container: dict[str, int]) -> _native.OutputSpec:
        """按 SPS 检查输出 RGB 的条件，给出交给 C++ 的输出规格。原生 NV12 输出不做颜色转换，不受这些限制。"""
        spec = _native.OutputSpec()
        if self._format == "nv12":
            return spec
        spec.rgb = True
        conflicts = _color_conflicts(sps, container)
        if conflicts:
            raise Error("the container and the bitstream disagree on " + "; ".join(conflicts))
        if not sps.video_full_range_flag:
            reason = "is limited range (16-235)" if sps.video_signal_type_present else "has no range flag, so it is limited range (16-235) by default"
            raise UnsupportedError(f"the video {reason}; converting limited range to RGB is not supported yet "
                                   "(CV-CUDA 0.19 only converts full range). Use format='nv12'")
        location = (sps.chroma_sample_loc_type_top_field, sps.chroma_sample_loc_type_bottom_field)
        if location != (0, 0):
            raise UnsupportedError(f"chroma sample location type {location} is not supported yet for RGB output (only type 0)")
        if self._colorspace is not None:
            spec.colorspace = self._colorspace
        elif sps.matrix_coefficients in _SOURCE_COLORSPACES:
            spec.colorspace = _SOURCE_COLORSPACES[sps.matrix_coefficients]
        else:
            what = "does not specify a matrix (matrix_coefficients 2)" if sps.matrix_coefficients == 2 else \
                f"uses matrix_coefficients {sps.matrix_coefficients}, which is not supported"
            raise UnsupportedError(f"the video {what}; pass colorspace={_COLORSPACES[0]!r}, {_COLORSPACES[1]!r} or {_COLORSPACES[2]!r}")
        left, top, right, bottom = sps.display_area
        crop_width, crop_height = right - left, bottom - top
        if self._crop is not None:
            x, y, w, h = self._crop
            if x + w > crop_width or y + h > crop_height:
                raise ValueError(f"crop {self._crop} does not fit the {crop_width}x{crop_height} picture")
            spec.crop_x, spec.crop_y, spec.crop_width, spec.crop_height = self._crop
            crop_width, crop_height = w, h
        target = (crop_width if self._width is None else self._width, crop_height if self._height is None else self._height)
        if target != (crop_width, crop_height):
            spec.width, spec.height = target
            spec.interpolation = _interpolation(self._interpolation, (crop_width, crop_height), target)
        return spec

    @property
    def metadata(self) -> VideoMetadata:
        """录像信息，关闭后仍可读。"""
        return self._metadata

    @property
    def decoder_config(self) -> dict[str, Any]:
        """实际交给 NVDEC 的参数和它报告的原始值。解出第一帧后才有格式、能力和建解码器的参数。"""
        self._check_open()
        return self._native.decoder_config()

    @property
    def stats(self) -> dict[str, Any]:
        """包数、解码次数、交出与选中的帧数、各解码状态的帧数、读取量；远程录像另有 HTTP 请求数（requests，含重试）
        和重试次数。按帧号读时是各次请求的累计，reads 为读取的次数。"""
        self._check_open()
        stats = self._native.stats()
        remote = self._download or self._range
        if remote is not None:
            stats.update(requests=remote.requests, retries=remote.retries)
        return stats

    def __iter__(self) -> Iterator[Frame]:
        """按显示顺序逐帧返回整段录像，同 frames()。"""
        return self.frames()

    def frames(self, *, step: int | None = None, fps: float | None = None, source_fps: float | None = None) -> Iterator[Frame]:
        """按显示顺序逐帧返回整段录像里选中的帧，默认全部。step=k 从第 0 帧起每 k 帧留一帧；fps 与 source_fps 一起给时
        按 VPF 的 EMDownSampler 从 source_fps 降到 fps（60→30 留第 1、3、5…帧），两种不能同时用。没选中的帧照样解码和检查，
        但不转换、不交出。每个 VideoReader 只能顺序读一遍。"""
        self._check_open()
        sampler = _sampler(step, fps, source_fps)
        if self._index is not None:
            raise Error("this VideoReader was opened with a video index; read frames with get_frames_at")
        if self._started:
            raise Error("a VideoReader can only be read sequentially once")
        self._started = True
        return self._iterate(repeat(True) if sampler is None else sampler.decisions())

    def _iterate(self, decisions: Iterator[bool]) -> Iterator[Frame]:
        given = 0
        while True:
            self._check_open()
            item = self._native.next_frame()
            if item is False:
                chunk = bytes(1 if keep else 0 for keep in islice(decisions, _SELECTION_CHUNK))
                if not chunk:
                    raise Error(f"the sampler gave no decision for frame {given}")
                self._native.add_selection(chunk)
                given += len(chunk)
                continue
            if item is None:
                return
            image, info = item
            yield Frame(image=image, **info)

    def __len__(self) -> int:
        """帧数，即视频索引的帧数。顺序读时帧数要读完才知道，报 TypeError（同没有长度的对象）。"""
        if self._index is None:
            raise TypeError("the frame count is only known when reading by frame number (open the VideoReader with index=...)")
        return self._index.count

    def __bool__(self) -> bool:
        return True  # 不经 __len__：顺序读的读取器没有帧数

    def __getitem__(self, key: int | slice) -> Frame | FrameBatch:
        """reader[i] 同 get_frame_at(i)；reader[a:b:k] 按切片的规则取帧号，可以倒序，返回 FrameBatch。"""
        if isinstance(key, slice):
            return self.get_frames_at(list(range(*key.indices(self._require_index().count))))
        return self.get_frame_at(key)

    def get_frame_at(self, number: int) -> Frame:
        """按帧号读一帧，负数从末尾数。"""
        batch = self.get_frames_at([number])
        return Frame(image=batch.images[0], **{name: getattr(batch, name)[0] for name in _FRAME_FIELDS})

    def get_frames_in_range(self, start: int, stop: int, step: int = 1) -> FrameBatch:
        """帧号从 start 起、到 stop 之前，每 step 帧取一帧（同 range）。负数从末尾数，超出的边界按切片的规则收进录像范围；
        step 必须为正，倒序请用切片或 get_frames_at。"""
        for name, value in (("start", start), ("stop", stop), ("step", step)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        return self.get_frames_at(list(range(*slice(start, stop, step).indices(self._require_index().count))))

    def _require_index(self) -> _planner.VideoIndex:
        self._check_open()
        if self._index is None:
            raise ValueError("reading by frame number needs a video index: open the VideoReader with index=...")
        return self._index

    def get_frames_at(self, frames: list[int]) -> FrameBatch:
        """按帧号列表读，保持列表的顺序和重复项（负下标从末尾数，-1 是最后一帧），返回 FrameBatch。要打开时给了视频索引。
        一次请求从所需帧中最早的 GOP 入口解码到解码顺序最靠后的所需帧，只读这一段字节；要么交出全部帧，要么报错。"""
        self._require_index()
        if self._failed:
            raise Error("an earlier error stopped reading this video")
        numbers = _planner.frame_numbers(frames, self._index.count)
        if not numbers:
            return self._batch([], np.empty((0, *self._image_shape), dtype=np.uint8), {})
        plan = _planner.plan(self._index, numbers)
        decode_frames, read_bytes = len(plan.packets), plan.range_end - plan.range_start
        if decode_frames > self._max_decode_frames or read_bytes > self._max_read_bytes:
            raise ValueError(f"frames {min(numbers)} to {max(numbers)} in one request need decoding {decode_frames} frames and reading "
                             f"{read_bytes} bytes; the limits are {self._max_decode_frames} frames (max_decode_frames) and "
                             f"{self._max_read_bytes} bytes (max_read_bytes). Request frames that are far apart separately, "
                             "or raise the limits")
        # 按请求一次分配最终的数组，C++ 把每个不同的帧直接写进它第一次出现的位置，重复的位置在这里复制：各占一份数据
        images = np.empty((len(numbers), *self._image_shape), dtype=np.uint8)
        first: dict[int, int] = {}
        for i, n in enumerate(numbers):
            first.setdefault(n, i)
        slots = np.array([first[n] for n in plan.targets], dtype=np.int64)
        try:
            data = None if self._range is None else self._range.fetch(plan.range_start, plan.range_end)
            decoded = self._native.read(plan.range_start, plan.range_end, plan.packets, np.array(plan.targets, dtype=np.int64),
                                        images, slots, data)
        except (Error, FileNotFoundError):
            self._failed = True
            raise
        for i, n in enumerate(numbers):
            if first[n] != i:
                images[i] = images[first[n]]
        return self._batch(numbers, images, dict(decoded))

    def _batch(self, numbers: list[int], images: np.ndarray, statuses: dict[int, int]) -> FrameBatch:
        index = self._index
        colorspace = self._spec.colorspace if self._spec.rgb else None
        interpolation = (self._spec.interpolation or None) if self._spec.rgb else None
        return FrameBatch(
            images=images, number=list(numbers), pts=[None] * len(numbers), key_frame=[bool(index.key_frame[n]) for n in numbers],
            decode_status=[statuses[n] for n in numbers], decode_order=[int(index.decode_order[n]) for n in numbers],
            decode_start=[int(index.decode_start[n]) for n in numbers], packet_start=[int(index.packet_start[n]) for n in numbers],
            packet_end=[int(index.packet_end[n]) for n in numbers], colorspace=[colorspace] * len(numbers),
            interpolation=[interpolation] * len(numbers))

    def submit(self, frames: list[int]) -> Any:
        """get_frames_at() 的异步版本，返回可取结果或取消的请求句柄。"""
        raise UnsupportedError("asynchronous requests are not implemented yet")

    def close(self) -> None:
        """停止下载，释放解码器和显存。已经交出的帧仍然有效。"""
        if not self._closed:
            self._closed = True
            if self._download is not None:
                self._download.close()
            self._native.close()

    def _check_open(self) -> None:
        if self._closed:
            raise Error("video is closed")

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_sample(
    sample_index: Any,
    video_index: Any,
    root: str | os.PathLike[str],
    *,
    step: int | None = None,
    fps: float | None = None,
    source_fps: float | None = None,
    device: int = 0,
    crop: tuple[int, int, int, int] | None = None,
    width: int | None = None,
    height: int | None = None,
    interpolation: str | None = None,
    colorspace: str | None = None,
    format: str = "rgb24",
    storage_options: dict[str, Any] | None = None,
    max_decode_frames: int | None = None,
    max_read_bytes: int | None = None,
) -> FrameBatch:
    """按样本索引读帧（格式见 docs/index-format.md）：在 root 下打开样本的录像，对样本的帧号列表按 step 或 fps 选帧后读取，
    返回值同 VideoReader.get_frames_at。先检查两份索引和样本全部帧号的上界，再选帧，选帧丢掉的越界帧号也报错。
    step、fps 作用在样本帧号列表的位置上（step=2 取第 0、2、4… 项），不看帧号本身的间隔。"""
    sample = _index.load_sample_index(sample_index)
    index = _index.load_video_index(video_index)
    count = len(index["decode_order"])
    over = [f for f in sample["frames"] if f >= count]
    if over:
        raise IndexError(f"sample frames {over[:5]} are out of range for a video with {count} frames")
    sampler = _sampler(step, fps, source_fps)
    frames = list(sample["frames"]) if sampler is None else sampler.select(sample["frames"])
    with VideoReader(_sample_source(root, sample["video"]), device=device, crop=crop, width=width, height=height,
                     interpolation=interpolation, colorspace=colorspace, format=format, index=index,
                     storage_options=storage_options, max_decode_frames=max_decode_frames, max_read_bytes=max_read_bytes) as reader:
        return reader.get_frames_at(frames)


def _sample_source(root: str | os.PathLike[str], video: str) -> str:
    """样本录像的位置：本地根目录按路径各段拼接；s3://、oss:// 前缀只在后面接上对象键，不做 URL 解析（样本路径已检查过
    是不含 ..、空段、. 段的相对 POSIX 路径）。"""
    base = os.fspath(root)
    if "://" in base:
        if not base.startswith(_REMOTE_SCHEMES):
            raise ValueError(f"unknown root scheme in {base!r}; use a local directory, s3:// or oss://")
        return base.rstrip("/") + "/" + video
    return os.path.join(base, *video.split("/"))


def set_decoder_cache_capacity(capacity: int) -> None:
    """按帧号读时，每个进程每块 GPU 缓存几个空闲的硬件解码器（默认 1）。新建、销毁硬件解码器要几十到两百多毫秒，
    缓存让下一次请求直接复用，换文件也一样，只要视频格式（编码、尺寸、位深等）相同。每个空闲解码器占一些显存；
    设为 0 关闭缓存并销毁现有的空闲解码器。"""
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError(f"capacity must be a non-negative integer, got {capacity!r}")
    _native.set_decoder_cache_capacity(capacity)
