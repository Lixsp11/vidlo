"""RGB 输出的颜色、裁剪、缩放与基准逐帧比较。

同一段录像同时开几路 vidlo 读取，逐帧对齐比较：
- 颜色：整帧 RGB 与 FFmpeg swscale 显式指定矩阵和范围的结果比较。colorspace="bt601" 对 swscale bt601；默认（colorspace=None）
  按录像标记的矩阵，对 swscale 同一矩阵。swscale 的输入是 FFmpeg CPU 解码的 YUV，已知与 vidlo 解码逐位相同。
- 标准公式：每隔若干帧，用 vidlo 的原生 NV12 按 full range 浮点公式（每 2×2 像素共用一份色度）算出 RGB，交叉核对。
- 裁剪：裁剪输出必须与整帧 RGB 切出的同一块逐位相同（转换按 2×2 块逐点进行，裁剪起点为偶数）。
- 缩放：缩放输出与 OpenCV cv2.resize 对同一份裁剪输出的结果比较，与颜色误差分开；缩小为面积插值，放大另取前若干帧比三次插值。
- 最近邻：interpolation="nearest_exact" 的缩小与放大（前若干帧）必须与按 CV-CUDA 取样公式从裁剪输出直接取像素的结果逐位相同。
每项记下逐通道差值的分布与最大值；验收为每个通道最多差 2，裁剪与最近邻要求逐位相同。输出目录必须不存在，结果写进 report.json。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import vidlo

# 默认取训练用的裁剪与尺寸：去掉左侧 32 列条码，1248×720 缩小到 832×480（2/3）；放大检查用 1.5 倍
CROP = (32, 0, 1248, 720)
SIZE = (832, 480)
UPSCALE = (1872, 1080)
MATRIX_NAMES = {1: "bt709", 5: "bt601", 6: "bt601", 9: "bt2020"}
# 浮点公式的系数：Kr、Kb（BT.601、BT.709、BT.2020 各自的定义）
KR_KB = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722), "bt2020": (0.2627, 0.0593)}


class DiffStats:
    """逐元素差值 |a-b| 的分布：0、1、2 与大于 2 的个数，各通道最大值。"""

    def __init__(self) -> None:
        self.frames = 0
        self.counts = [0, 0, 0, 0]
        self.channel_max = [0, 0, 0]
        self.worst_frame = None

    def add(self, number: int, ours: np.ndarray, theirs: np.ndarray) -> None:
        if ours.shape != theirs.shape:
            raise RuntimeError(f"frame {number}: shape {ours.shape} vs {theirs.shape}")
        d = cv2.absdiff(ours, theirs)
        flat = d.reshape(d.shape[0], -1)  # 单通道视图，供 OpenCV 计数（比逐个 numpy 比较快一个数量级）
        nonzero = cv2.countNonZero(flat)
        at_least_2 = cv2.countNonZero(cv2.compare(flat, 1, cv2.CMP_GT)) if nonzero else 0
        over = cv2.countNonZero(cv2.compare(flat, 2, cv2.CMP_GT)) if at_least_2 else 0
        self.counts[0] += d.size - nonzero
        self.counts[1] += nonzero - at_least_2
        self.counts[2] += at_least_2 - over
        self.counts[3] += over
        channel = [int(cv2.minMaxLoc(c)[1]) for c in cv2.split(d)] if nonzero else [0, 0, 0]
        if max(channel) > max(self.channel_max):
            self.worst_frame = number
        self.channel_max = [max(a, int(b)) for a, b in zip(self.channel_max, channel)]
        self.frames += 1

    def report(self) -> dict[str, Any]:
        total = sum(self.counts) or 1
        return {"frames": self.frames, "max": max(self.channel_max), "channel_max": self.channel_max,
                "counts": dict(zip(("0", "1", "2", ">2"), self.counts)),
                "shares": {k: round(v / total, 6) for k, v in zip(("0", "1", "2", ">2"), self.counts)},
                "worst_frame": self.worst_frame, "passed": self.frames > 0 and max(self.channel_max) <= 2}


def read_exact(stream, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def swscale(video: Path, matrix: str, log: Path) -> tuple[list[str], subprocess.Popen]:
    cmd = ["ffmpeg", "-v", "warning", "-nostdin", "-i", str(video), "-map", "0:v:0", "-fps_mode", "passthrough",
           "-vf", f"scale=in_color_matrix={matrix}:in_range=full:out_range=full:flags=bicubic", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    return cmd, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log.open("wb"), bufsize=0)


def nearest_reference(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """CV-CUDA NVCV_INTERP_NEAREST 的取样：输出第 i 个像素取源的 floor((i + 0.5) × 源长 / 目标长)，按 float32 计算，
    不超过源的最后一个像素（OpResize.cu 的 NearestResize）。与 cv2.INTER_NEAREST 的 floor(i × 源长 / 目标长) 不同。"""
    def index(source: int, target: int) -> np.ndarray:
        scale = np.float32(source) / np.float32(target)
        position = (np.arange(target, dtype=np.float32) + np.float32(0.5)) * scale
        return np.minimum(np.floor(position).astype(np.int64), source - 1)

    height, width = image.shape[:2]
    return image[index(height, size[1])][:, index(width, size[0])]


def formula_rgb(nv12: np.ndarray, matrix: str) -> np.ndarray:
    """full range 的标准公式：R = Y + 2(1-Kr)V'，B = Y + 2(1-Kb)U'，G 由亮度定义解出；U' = U-128，V' = V-128。"""
    kr, kb = KR_KB[matrix]
    kg = 1 - kr - kb
    h = nv12.shape[0] * 2 // 3
    y = nv12[:h].astype(np.float64)
    uv = nv12[h:].astype(np.float64) - 128
    u = np.repeat(np.repeat(uv[:, 0::2], 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(uv[:, 1::2], 2, axis=0), 2, axis=1)
    r = y + 2 * (1 - kr) * v
    b = y + 2 * (1 - kb) * u
    g = y - (2 * kb * (1 - kb) * u + 2 * kr * (1 - kr) * v) / kg
    return np.clip(np.rint(np.stack([r, g, b], axis=-1)), 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--formula-every", type=int, default=60, help="每隔多少帧做一次标准公式核对")
    parser.add_argument("--upscale-frames", type=int, default=120, help="放大检查的帧数")
    parser.add_argument("--crop", type=int, nargs=4, default=CROP, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--size", type=int, nargs=2, default=SIZE, metavar=("W", "H"), help="缩小后的尺寸")
    parser.add_argument("--upscale", type=int, nargs=2, default=UPSCALE, metavar=("W", "H"), help="放大后的尺寸")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    crop, size, upscale_size = tuple(args.crop), tuple(args.size), tuple(args.upscale)

    with vidlo.VideoReader(args.video, format="nv12") as probe:
        source_matrix = MATRIX_NAMES[probe.metadata.matrix_coefficients]
    commands = {}
    commands["bt601"], ref601 = swscale(args.video, "bt601", args.output / "ffmpeg-bt601.stderr.log")
    commands["source"], ref_source = swscale(args.video, source_matrix, args.output / "ffmpeg-source.stderr.log")

    stats = {name: DiffStats() for name in ("color_bt601", "color_source", "formula_bt601", "formula_source", "crop_vs_full",
                                            "resize_area_vs_opencv", "resize_cubic_vs_opencv", "nearest_shrink", "nearest_enlarge")}
    started = time.perf_counter()
    # 裁剪、缩放几路都用默认的按录像标记转换，与 "source" 那一路的整帧比较
    settings: dict[str, dict[str, Any]] = {
        "bt601": {"colorspace": "bt601"},
        "source": {},
        "crop": {"crop": crop},
        "resize": {"crop": crop, "width": size[0], "height": size[1]},
        "nearest": {"crop": crop, "width": size[0], "height": size[1], "interpolation": "nearest_exact"},
        "yuv": {"format": "nv12"},
    }
    videos = {name: vidlo.VideoReader(args.video, **s) for name, s in settings.items()}
    upscale = vidlo.VideoReader(args.video, crop=crop, width=upscale_size[0], height=upscale_size[1])
    upscale_nearest = vidlo.VideoReader(args.video, crop=crop, width=upscale_size[0], height=upscale_size[1],
                                        interpolation="nearest_exact")
    frame_bytes = None
    compared = 0
    metadata: dict[str, Any] = {}
    error = None
    try:
        iterators = {name: iter(video) for name, video in videos.items()}
        upscale_frames = iter(upscale)
        upscale_nearest_frames = iter(upscale_nearest)
        for frames in zip(*iterators.values()):
            f = dict(zip(iterators, frames))
            numbers = {name: frame.number for name, frame in f.items()}
            if set(numbers.values()) != {compared}:
                raise RuntimeError(f"frame numbers differ at position {compared}: {numbers}")
            full = f["bt601"].image
            frame_bytes = full.nbytes
            theirs601 = np.frombuffer(read_exact(ref601.stdout, frame_bytes), np.uint8)
            theirs_source = np.frombuffer(read_exact(ref_source.stdout, frame_bytes), np.uint8)
            if theirs601.size != frame_bytes or theirs_source.size != frame_bytes:
                raise RuntimeError(f"ffmpeg ended early at frame {compared}")
            stats["color_bt601"].add(compared, full, theirs601.reshape(full.shape))
            stats["color_source"].add(compared, f["source"].image, theirs_source.reshape(full.shape))
            x, y, w, h = crop
            stats["crop_vs_full"].add(compared, f["crop"].image, f["source"].image[y:y + h, x:x + w])
            area = cv2.resize(f["crop"].image, size, interpolation=cv2.INTER_AREA)
            stats["resize_area_vs_opencv"].add(compared, f["resize"].image, area)
            stats["nearest_shrink"].add(compared, f["nearest"].image, nearest_reference(f["crop"].image, size))
            if compared % args.formula_every == 0:
                stats["formula_bt601"].add(compared, full, formula_rgb(f["yuv"].image, "bt601"))
                stats["formula_source"].add(compared, f["source"].image, formula_rgb(f["yuv"].image, source_matrix))
            if compared < args.upscale_frames:
                up = next(upscale_frames)
                cubic = cv2.resize(f["crop"].image, upscale_size, interpolation=cv2.INTER_CUBIC)
                stats["resize_cubic_vs_opencv"].add(compared, up.image, cubic)
                up_nearest = next(upscale_nearest_frames)
                stats["nearest_enlarge"].add(compared, up_nearest.image, nearest_reference(f["crop"].image, upscale_size))
                metadata["upscale"] = {k: v for k, v in vars(up).items() if k != "image"}
                metadata["upscale_nearest"] = {k: v for k, v in vars(up_nearest).items() if k != "image"}
            if compared == 0:
                metadata.update({name: {k: v for k, v in vars(frame).items() if k != "image"} for name, frame in f.items()})
            compared += 1
    except (vidlo.Error, RuntimeError) as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        info = asdict(videos["bt601"].metadata)
        video_stats = videos["bt601"].stats
        for video in [*videos.values(), upscale, upscale_nearest]:
            video.close()
    elapsed = time.perf_counter() - started

    extra = {}
    for name, process in (("bt601", ref601), ("source", ref_source)):
        count = 0
        if frame_bytes:
            while len(read_exact(process.stdout, frame_bytes)) == frame_bytes:
                count += 1
        process.stdout.close()
        extra[name] = {"exit": process.wait(), "extra_frames": count}

    results = {name: s.report() for name, s in stats.items()}
    for name in ("crop_vs_full", "nearest_shrink", "nearest_enlarge"):
        results[name]["passed"] = results[name]["frames"] > 0 and results[name]["max"] == 0
    info.pop("codec_config")
    report = {
        "video": str(args.video), "vidlo_version": vidlo.__version__, "opencv_version": cv2.__version__,
        "ffmpeg_version": subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
        "ffmpeg_commands": commands, "ffmpeg": extra, "source_matrix": source_matrix, "crop": crop, "size": size, "upscale": upscale_size,
        "frames": compared, "error": error, "elapsed_s": round(elapsed, 3), "results": results, "first_frame_info": metadata,
        "info": info, "stats": video_stats,
    }
    report["passed"] = (error is None and compared > 0 and compared == video_stats["displayed"]
                        and all(e["exit"] == 0 and e["extra_frames"] == 0 for e in extra.values())
                        and all(r["passed"] for r in results.values()))
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"video": str(args.video), "passed": report["passed"], "frames": compared, "error": error,
                      "elapsed_s": report["elapsed_s"]}, ensure_ascii=False))
    for name, r in results.items():
        print(f"  {name:24s} max={r['max']} channel_max={r['channel_max']} shares={r['shares']} frames={r['frames']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
