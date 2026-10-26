"""vidlo：用 GPU 读视频帧。用法见 docs/api.md。

导入时不初始化 CUDA，第一次打开录像时才初始化，所以先导入再启动的子进程也能用。
"""

from importlib.metadata import version as _package_version

from ._index import load_sample_index, load_video_index, validate_sample_index, validate_video_index
from ._native import Error, TruncatedError, UnsupportedError
from ._reader import Frame, VideoMetadata, VideoReader, read_sample

__version__ = _package_version("vidlo")

__all__ = [
    "Error",
    "Frame",
    "TruncatedError",
    "UnsupportedError",
    "VideoMetadata",
    "VideoReader",
    "load_sample_index",
    "load_video_index",
    "read_sample",
    "validate_sample_index",
    "validate_video_index",
]
