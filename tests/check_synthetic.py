"""用合成视频和构造的输入做的检查：原生 YUV 逐位比较、RGB 颜色与几何比较、SPS 逐字段对照、按步长与帧率选帧、未实现情况的报错、
索引检查、导入时不初始化 CUDA、只导入后 fork 的子进程能读、初始化过 CUDA 后 fork 的子进程报错。

用 FFmpeg（libx264、libx265）在输出目录里生成合成视频，再逐项检查，结果写进 report.json。输出目录必须不存在。
x265 默认 open-gop=1（关键帧是 CRA，后面带要参考上一个 GOP 的帧），第一期不支持，所以能正常读的合成视频都显式关掉它；
另有一段打开它的视频，用来确认会报错。各视频实际含有哪些 NAL 类型由 trace_headers 记录在报告里。
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2

import vidlo
from vidlo import _hevc
from vidlo._sampler import EMDownSampler, StrideSampler

HERE = Path(__file__).resolve().parent

# 名称 → (ffmpeg 编码参数, 封装扩展名)。testsrc2 有运动和颜色变化，能产生 B 帧
SYNTHETIC: dict[str, tuple[list[str], str]] = {
    # b1 同类：HEVC、full range、B 帧；GOP 25 帧、共 62 帧，最后一个 GOP 不满；参数集只在 extradata 里
    "hevc_full_bframes": (["-t", "2.5", "-r", "25", "-c:v", "libx265", "-pix_fmt", "yuvj420p", "-color_range", "pc",
                           "-x265-params", "keyint=25:min-keyint=25:bframes=3:scenecut=0:open-gop=0:log-level=error"], "mkv"),
    "hevc_limited_bframes": (["-t", "2", "-r", "30", "-c:v", "libx265", "-pix_fmt", "yuv420p",
                              "-x265-params", "keyint=30:min-keyint=30:bframes=4:b-pyramid=1:scenecut=0:open-gop=0:log-level=error"],
                             "mkv"),
    "hevc_single_frame": (["-frames:v", "1", "-c:v", "libx265", "-pix_fmt", "yuv420p", "-x265-params", "open-gop=0:log-level=error"],
                          "mkv"),
    # 能输出 RGB 的：full range，标明 BT.709 或 BT.601（smpte170m，matrix_coefficients 6）
    "hevc_full_bt709": (["-t", "2", "-r", "30", "-c:v", "libx265", "-pix_fmt", "yuvj420p", "-color_range", "pc", "-colorspace", "bt709",
                         "-color_primaries", "bt709", "-color_trc", "bt709",
                         "-x265-params", "keyint=30:min-keyint=30:bframes=3:scenecut=0:open-gop=0:log-level=error"], "mkv"),
    "hevc_full_bt601": (["-t", "1", "-r", "30", "-c:v", "libx265", "-pix_fmt", "yuvj420p", "-color_range", "pc", "-colorspace", "smpte170m",
                         "-color_primaries", "smpte170m", "-color_trc", "smpte170m",
                         "-x265-params", "keyint=15:min-keyint=15:bframes=2:scenecut=0:open-gop=0:log-level=error"], "mkv"),
    # 色度位置 type 2：原生 YUV 能读，RGB 报错
    "hevc_full_chromaloc2": (["-t", "1", "-c:v", "libx265", "-pix_fmt", "yuvj420p", "-color_range", "pc",
                              "-x265-params", "chromaloc=2:open-gop=0:log-level=error"], "mkv"),
    # SPS 解析的覆盖：缩放列表、多个时间子层；参数集同时写在码流里（与 codec_config 相同）
    "hevc_scaling_list": (["-t", "1", "-c:v", "libx265", "-pix_fmt", "yuv420p", "-x265-params", "scaling-list=default:open-gop=0:log-level=error"],
                          "mkv"),
    "hevc_temporal_layers": (["-t", "1", "-c:v", "libx265", "-pix_fmt", "yuv420p",
                              "-x265-params", "temporal-layers=3:bframes=3:b-pyramid=1:open-gop=0:log-level=error"], "mkv"),
    "hevc_inband_ps": (["-t", "2", "-r", "30", "-c:v", "libx265", "-pix_fmt", "yuvj420p", "-color_range", "pc",
                        "-x265-params", "keyint=30:min-keyint=30:repeat-headers=1:open-gop=0:log-level=error"], "mkv"),
    # 未实现的情况
    "h264": (["-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p"], "mkv"),
    "hevc_mp4": (["-t", "1", "-c:v", "libx265", "-pix_fmt", "yuv420p", "-x265-params", "log-level=error"], "mp4"),
    "hevc_10bit": (["-t", "1", "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error"], "mkv"),
    "hevc_444": (["-t", "1", "-c:v", "libx265", "-pix_fmt", "yuv444p", "-x265-params", "log-level=error"], "mkv"),
    "hevc_open_gop": (["-t", "2", "-r", "30", "-c:v", "libx265", "-pix_fmt", "yuv420p",
                       "-x265-params", "keyint=30:min-keyint=30:bframes=4:scenecut=0:open-gop=1:log-level=error"], "mkv"),
}
POSITIVE = ("hevc_full_bframes", "hevc_limited_bframes", "hevc_single_frame", "hevc_full_bt709", "hevc_full_bt601",
            "hevc_full_chromaloc2", "hevc_scaling_list", "hevc_temporal_layers", "hevc_inband_ps")
RGB_POSITIVE = ("hevc_full_bt709", "hevc_full_bt601")
FRAME_ARRAYS = ("decode_order", "decode_start", "packet_start", "packet_end")


def make_video(directory: Path, name: str) -> Path:
    args, ext = SYNTHETIC[name]
    path = directory / f"{name}.{ext}"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", *args, str(path)]
    subprocess.run(cmd, check=True)
    return path


NAL_NAMES = {0: "TRAIL_N", 1: "TRAIL_R", 2: "TSA_N", 3: "TSA_R", 4: "STSA_N", 5: "STSA_R", 6: "RADL_N", 7: "RADL_R",
             8: "RASL_N", 9: "RASL_R", 16: "BLA_W_LP", 17: "BLA_W_RADL", 18: "BLA_N_LP", 19: "IDR_W_RADL", 20: "IDR_N_LP",
             21: "CRA_NUT", 32: "VPS", 33: "SPS", 34: "PPS", 35: "AUD", 39: "SEI_PREFIX", 40: "SEI_SUFFIX"}


def nal_types(path: Path) -> dict[str, int]:
    """trace_headers 自己输出的每个 NAL 的 nal_unit_type 及个数（含 extradata 里的参数集）。"""
    result = subprocess.run(["ffmpeg", "-v", "info", "-nostdin", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                             "-bsf:v", "trace_headers", "-f", "null", "-"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"trace_headers failed on {path}: {result.stderr[-1000:]}")
    types = [int(m) for m in re.findall(r"^\[trace_headers[^\]]*\]\s+\d+\s+nal_unit_type\s+[01]+ = (\d+)$",
                                        result.stderr, re.M)]
    if not any(t <= 31 for t in types):
        raise RuntimeError(f"trace_headers listed no slice NAL units in {path}")
    return dict(Counter(NAL_NAMES.get(t, str(t)) for t in types))


def make_truncated(source: Path, target: Path) -> int:
    """把录像截断在最后一个视频包（文件里最后一个块）的负载中间，返回这个包的 pts。"""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,pos,size", "-of", "json",
                          str(source)], check=True, capture_output=True, text=True).stdout
    last = json.loads(out)["packets"][-1]
    cut = int(last["pos"]) + int(last["size"]) // 2
    target.write_bytes(source.read_bytes()[:cut])
    return int(last["pts"])


def frame_hashes(path: Path) -> tuple[dict[int, str], str | None, dict[str, Any]]:
    """顺序读一遍，返回 pts → 画面 SHA256、截断错误的信息（没有则为 None）和统计。"""
    hashes: dict[int, str] = {}
    truncated = None
    with vidlo.VideoReader(path, format="nv12") as video:
        try:
            for frame in video:
                hashes[frame.pts] = hashlib.sha256(frame.image.tobytes()).hexdigest()
        except vidlo.TruncatedError as e:
            truncated = str(e)
        stats = video.stats
    return hashes, truncated, stats


def check_truncated(original: Path, truncated: Path, missing_pts: int) -> dict[str, Any]:
    """截断的录像：交出除最后那个不完整的包以外的全部帧，画面与完整录像相同，最后报 TruncatedError。"""
    expected, _, _ = frame_hashes(original)
    expected.pop(missing_pts)
    try:
        got, message, stats = frame_hashes(truncated)
    except BaseException as e:  # noqa: BLE001
        return {"passed": False, "raised": type(e).__name__, "message": str(e)}
    return {"passed": message is not None and got == expected and stats["truncated_at"] >= 0,
            "message": message, "frames": len(got), "expected_frames": len(expected), "same_pictures": got == expected,
            "truncated_at": stats["truncated_at"]}


def make_unknown_size_segment(source: Path, target: Path) -> None:
    """把 Segment 的长度改成“未知”（8 字节全 1），截断就只能在 Cluster、块这一层发现。"""
    data = bytearray(source.read_bytes())
    at = data.find(bytes.fromhex("18538067"))
    if at < 0 or data[at + 4] != 0x01:
        raise RuntimeError(f"unexpected Segment header in {source}")
    data[at + 4:at + 12] = bytes.fromhex("01ffffffffffffff")
    target.write_bytes(bytes(data))


def make_inband_mismatch(source: Path, target: Path) -> None:
    """码流里那份 SPS 的 general_level_idc 改成另一个值，使它与 codec_config 里的 SPS 不同（其余字节不变）。

    文件里第一次出现的 SPS 是 Tracks 里的 hvcC，第二次是第一个关键帧里的。level 在 RBSP 的第 14 字节，
    前面的保留位全为 0，NAL 里有防竞争字节，所以按 RBSP 下标换算到 NAL 里的位置；新旧值都不含 0，不影响防竞争字节。
    """
    with vidlo.VideoReader(source, format="nv12") as video:
        config = video.metadata.codec_config
    sps = next(nal for nal_type, nal in _hevc.parse_hvcc(config).nal_units if nal_type == _hevc.NAL_SPS)
    data = bytearray(source.read_bytes())
    first = data.find(sps)
    second = data.find(sps, first + 1)
    if first < 0 or second < 0:
        raise RuntimeError(f"{source} does not carry the SPS inside the stream")
    rbsp_index, zeros = 0, 0
    for i, byte in enumerate(sps):
        if zeros >= 2 and byte == 3:
            zeros = 0
            continue
        if rbsp_index == 14:
            break
        rbsp_index += 1
        zeros = zeros + 1 if byte == 0 else 0
    level = data[second + i]
    if level == 0 or level == 3:
        raise RuntimeError(f"unexpected level byte {level}")
    data[second + i] = 90 if level != 90 else 93
    target.write_bytes(bytes(data))


def decode_all(path: Path, **settings: Any) -> int:
    """顺序读完返回帧数；没给设置时读原生 NV12。"""
    with vidlo.VideoReader(path, **(settings or {"format": "nv12"})) as video:
        return sum(1 for _ in video)


def sampled(path: Path, settings: dict[str, Any], **selection: Any) -> list[tuple[int, str]]:
    with vidlo.VideoReader(path, **settings) as video:
        return [(f.number, hashlib.sha256(f.image.tobytes()).hexdigest()) for f in video.frames(**selection)]


def check_resize(path: Path, size: tuple[int, int], method: str | None, expected: str, cv_interp: int) -> dict[str, Any]:
    """缩放输出与 cv2.resize 对同一份裁剪输出的结果比较，并核对 frame.interpolation 记下的插值方式。"""
    crop = (16, 8, 288, 224)
    worst, methods, frames = 0, set(), 0
    with vidlo.VideoReader(path, crop=crop) as plain, \
            vidlo.VideoReader(path, crop=crop, width=size[0], height=size[1], interpolation=method) as scaled:
        for a, b in zip(plain, scaled, strict=True):  # 一路提前结束就抛 ValueError
            if a.number != b.number:
                raise RuntimeError(f"frame numbers differ: {a.number} vs {b.number}")
            methods.add(b.interpolation)
            reference = cv2.resize(a.image, size, interpolation=cv_interp)
            worst = max(worst, int(cv2.absdiff(b.image, reference).max()))
            frames += 1
    return {"passed": frames > 0 and methods == {expected} and worst <= 2, "frames": frames, "methods": sorted(methods), "max": worst}


def check_selection(path: Path, settings: dict[str, Any], sampler: Any, **selection: Any) -> dict[str, Any]:
    """顺序读时按 step 或 fps 选帧：交出的帧号等于对应内部采样器 select(全部帧号) 的结果，画面与全部读出时同一帧相同。"""
    full = sampled(path, settings)
    got = sampled(path, settings, **selection)
    expected = [full[n] for n in sampler.select(list(range(len(full))))]
    return {"passed": got == expected and len(got) > 0, "frames": len(full), "kept": [n for n, _ in got][:20], "kept_count": len(got)}


def read_index(path: Path) -> dict[str, Any]:
    """顺序读一遍，按 docs/index-format.md 拼出视频索引。"""
    with vidlo.VideoReader(path, format="nv12") as video:
        info = video.metadata
        frames = list(video)
    return {"version": 1, "file_size": info.file_size, "codec": info.codec,
            "codec_config": base64.b64encode(info.codec_config).decode(),
            **{name: [getattr(f, name) for f in frames] for name in FRAME_ARRAYS}}


def broken(index: dict[str, Any], change: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    result = copy.deepcopy(index)
    change(result)
    return result


def expect_raises(kind: type[BaseException], action: Callable[[], Any]) -> dict[str, Any]:
    try:
        action()
    except kind as e:
        return {"passed": True, "raised": type(e).__name__, "message": str(e)}
    except BaseException as e:  # noqa: BLE001 - 记录意外的异常类型
        return {"passed": False, "raised": type(e).__name__, "message": str(e)}
    return {"passed": False, "raised": None, "message": "no exception"}


def expect_no_error(action: Callable[[], Any]) -> dict[str, Any]:
    try:
        action()
    except BaseException as e:  # noqa: BLE001
        return {"passed": False, "raised": type(e).__name__, "message": str(e)}
    return {"passed": True}


def check_missing_credentials() -> dict[str, Any]:
    """凭证只从环境变量读：没有时在发请求之前就报错。在子进程里去掉这几个变量再试。"""
    code = ("import vidlo\n"
            "for source, options in (('s3://b/k.mkv', {'endpoint_url': 'https://127.0.0.1:9'}), ('oss://b/k.mkv', {'region': 'us-southeast-1'})):\n"
            "    try:\n"
            "        vidlo.VideoReader(source, format='nv12', storage_options=options)\n"
            "        print('no error')\n"
            "    except vidlo.Error as e:\n"
            "        print('Error', 'environment variables' in str(e))\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    return {"passed": out.stdout.split() == ["Error", "True", "Error", "True"], "stdout": out.stdout, "stderr": out.stderr[-1000:]}


def check_cuda_not_initialized_at_import() -> dict[str, Any]:
    # 没调用过 cuInit 时，驱动 API 返回 CUDA_ERROR_NOT_INITIALIZED（3）
    code = ("import ctypes, vidlo; n = ctypes.c_int(); "
            "print(ctypes.CDLL('libcuda.so.1').cuDeviceGetCount(ctypes.byref(n)))")
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.strip()
    return {"passed": out == "3", "cuDeviceGetCount_after_import": out}


# 在全新的进程里检查：父进程只导入 vidlo、不打开录像，fork 出的子进程打开并读完。
# 不能在本进程里做，因为前面的检查已经打开过录像、初始化了 CUDA。
FORK_CODE = """
import os, sys, traceback
import vidlo
path = sys.argv[1]
r, w = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(r)
    try:
        with vidlo.VideoReader(path, format="nv12") as video:
            os.write(w, str(sum(1 for _ in video)).encode())
        os._exit(0)
    except BaseException:
        os.write(w, traceback.format_exc().encode())
        os._exit(1)
os.close(w)
message = b""
while chunk := os.read(r, 65536):
    message += chunk
_, status = os.waitpid(pid, 0)
print(status, message.decode())
"""


def check_fork(path: Path, expected: int) -> dict[str, Any]:
    out = subprocess.run([sys.executable, "-c", FORK_CODE, str(path)], check=True, capture_output=True, text=True).stdout.strip()
    status, _, message = out.partition(" ")
    return {"passed": status == "0" and message == str(expected), "child_exit": status, "child_output": message[-1500:],
            "expected_frames": expected}


# 父进程打开录像、解出一帧（已初始化 CUDA）后 fork：子进程里新开录像、使用继承来的读取器（取帧、stats、decoder_config）
# 都报 fork 的错；之后关闭（close）或不关闭直接回收（gc）继承来的读取器，再走完解释器的退出流程，都要正常退出；
# 父进程照常读完剩下的帧。参数：录像、VideoReader 的设置（JSON）、close 或 gc
FORK_AFTER_CUDA_CODE = """
import gc, json, os, sys
import vidlo
path, settings, mode = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
reader = vidlo.VideoReader(path, **settings)
frames = iter(reader)
next(frames)
r, w = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(r)
    results = []
    for name, action in (("open", lambda: vidlo.VideoReader(path, **settings)), ("next", lambda: next(frames)),
                         ("stats", lambda: reader.stats), ("decoder_config", lambda: reader.decoder_config)):
        try:
            action()
            results.append(name + "=no-error")
        except vidlo.Error as e:
            results.append(name + ("=fork-error" if "forked" in str(e) and "spawn" in str(e) else "=other-error:" + str(e)[:300]))
        except BaseException as e:
            results.append(name + "=unexpected:" + type(e).__name__)
    if mode == "close":
        reader.close()
    else:
        del frames, reader
        gc.collect()
    os.write(w, " ".join(results).encode())
    os.close(w)
    sys.exit(0)
os.close(w)
message = b""
while chunk := os.read(r, 65536):
    message += chunk
_, status = os.waitpid(pid, 0)
rest = sum(1 for _ in frames)
reader.close()
print(status, rest + 1, message.decode())
"""
FORK_ERRORS = "open=fork-error next=fork-error stats=fork-error decoder_config=fork-error"

# 父进程只用别的方式初始化 CUDA（直接调 cuInit，不经过 vidlo）再 fork：vidlo 的进程号检查发现不了（PyTorch 也一样，
# 它只在自己初始化 CUDA 前登记 fork 处理，torch/csrc/utils/device_lazy_init.cpp:75-87），子进程里 cuInit 失败，
# 要报 vidlo.Error 并提示改用 spawn，而不是崩溃或卡住
FORK_AFTER_OTHER_CUDA_CODE = """
import ctypes, os, sys
import vidlo
path = sys.argv[1]
print("parent cuInit", ctypes.CDLL("libcuda.so.1").cuInit(0), flush=True)
pid = os.fork()
if pid == 0:
    try:
        with vidlo.VideoReader(path, format="nv12") as reader:
            print("child read", sum(1 for _ in reader), "frames", flush=True)
    except vidlo.Error as e:
        print("child vidlo.Error", str(e)[:500], flush=True)
    except BaseException as e:
        print("child", type(e).__name__, str(e)[:500], flush=True)
    os._exit(0)
_, status = os.waitpid(pid, 0)
print("child exit", status)
"""


def run_snippet(code: str, *args: str) -> tuple[str | None, str]:
    """在全新的进程里跑一段检查代码，返回 (stdout, 出错说明)；超时或异常退出时 stdout 为 None。"""
    try:
        out = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "timed out after 120 s"
    if out.returncode != 0:
        return None, f"exit {out.returncode}: {out.stderr[-1500:]}"
    return out.stdout.strip(), ""


def check_fork_after_cuda(path: Path, settings: dict[str, Any], mode: str, expected: int) -> dict[str, Any]:
    out, problem = run_snippet(FORK_AFTER_CUDA_CODE, str(path), json.dumps(settings), mode)
    if out is None:
        return {"passed": False, "problem": problem}
    status, frames, message = (out.split(" ", 2) + ["", ""])[:3]
    return {"passed": status == "0" and frames == str(expected) and message == FORK_ERRORS,
            "child_exit": status, "parent_frames": frames, "expected_frames": expected, "child_results": message[-1500:]}


def check_fork_after_other_cuda(path: Path) -> dict[str, Any]:
    out, problem = run_snippet(FORK_AFTER_OTHER_CUDA_CODE, str(path))
    if out is None:
        return {"passed": False, "problem": problem}
    lines = out.splitlines()
    child = next((line for line in lines if line.startswith("child ") and not line.startswith("child exit")), "")
    return {"passed": child.startswith("child vidlo.Error") and "spawn" in child and "child exit 0" in lines, "output": out[-1500:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    videos = args.output / "videos"
    videos.mkdir()
    paths = {name: make_video(videos, name) for name in SYNTHETIC}
    nal = {name: nal_types(path) for name, path in paths.items() if name.startswith("hevc")}

    checks: dict[str, Any] = {}
    for name in POSITIVE:
        result = subprocess.run([sys.executable, str(HERE / "compare_ffmpeg_yuv.py"), "--video", str(paths[name]),
                                 "--output", str(args.output / f"compare-{name}")], capture_output=True, text=True)
        checks[f"bit_exact_{name}"] = {"passed": result.returncode == 0, "stdout": result.stdout[-1500:], "stderr": result.stderr[-1500:]}
        result = subprocess.run([sys.executable, str(HERE / "check_index_metadata.py"), "--video", str(paths[name]),
                                 "--output", str(args.output / f"index-{name}")], capture_output=True, text=True)
        checks[f"index_metadata_{name}"] = {"passed": result.returncode == 0, "stdout": result.stdout[-1500:],
                                            "stderr": result.stderr[-1500:]}

    # SPS 与 trace_headers 逐字段对照：编码器生成的各段，加上借用多子层视频的 VPS 手工构造、用到全部语法分支的 SPS
    result = subprocess.run([sys.executable, str(HERE / "check_sps.py"), *[a for name in POSITIVE for a in ("--video", str(paths[name]))],
                             "--constructed-from", str(paths["hevc_temporal_layers"]), "--output", str(args.output / "sps")],
                            capture_output=True, text=True)
    checks["sps_matches_trace_headers"] = {"passed": result.returncode == 0, "stdout": result.stdout[-3000:], "stderr": result.stderr[-1500:]}
    sps = json.loads((args.output / "sps" / "report.json").read_text(encoding="utf-8"))["checks"]
    sps_summary = {name: {k: sps[str(paths[name])][k] for k in ("video_full_range_flag", "matrix_coefficients", "chroma_sample_loc_type")}
                   for name in POSITIVE}

    # RGB：颜色、裁剪、缩放与 swscale、OpenCV 比较（320×240：裁掉四边各 16/8，缩小到 192×160，放大到 432×336）
    for name in RGB_POSITIVE:
        result = subprocess.run([sys.executable, str(HERE / "compare_ffmpeg_rgb.py"), "--video", str(paths[name]), "--crop", "16", "8", "288", "224",
                                 "--size", "192", "160", "--upscale", "432", "336", "--formula-every", "5", "--upscale-frames", "1000",
                                 "--output", str(args.output / f"rgb-{name}")], capture_output=True, text=True)
        checks[f"rgb_{name}"] = {"passed": result.returncode == 0, "stdout": result.stdout[-2500:], "stderr": result.stderr[-1500:]}

    # 插值方式：显式指定时照用并记在 frame.interpolation，与 OpenCV 同名插值对同一份裁剪输出的结果比较。
    # 面积插值遇到放大的方向时报错：宽放大、高缩小的混合比例（自动选会选到面积插值），以及显式 area 放大
    for name, size, method, expected, cv_interp in (("explicit_linear", (192, 160), "linear", "linear", cv2.INTER_LINEAR),
                                                    ("explicit_cubic_shrink", (192, 160), "cubic", "cubic", cv2.INTER_CUBIC),
                                                    ("mixed_ratio_explicit_cubic", (400, 160), "cubic", "cubic", cv2.INTER_CUBIC)):
        checks[f"resize_{name}"] = check_resize(paths["hevc_full_bt709"], size, method, expected, cv_interp)
    checks["resize_mixed_ratio_auto_rejected"] = expect_raises(
        vidlo.UnsupportedError, lambda: vidlo.VideoReader(paths["hevc_full_bt709"], crop=(16, 8, 288, 224), width=400, height=160))
    checks["resize_area_enlarging_rejected"] = expect_raises(
        vidlo.UnsupportedError,
        lambda: vidlo.VideoReader(paths["hevc_full_bt709"], crop=(16, 8, 288, 224), width=400, height=311, interpolation="area"))
    # 只给宽：高保持裁剪后的 224
    with vidlo.VideoReader(paths["hevc_full_bt709"], crop=(16, 8, 288, 224), width=144) as video:
        first = next(iter(video))
    checks["resize_width_only_keeps_height"] = {"passed": first.image.shape == (224, 144, 3) and first.interpolation == "area",
                                                "shape": list(first.image.shape), "interpolation": first.interpolation}
    # nearest_exact 的取样与公式逐位比较在 compare_ffmpeg_rgb.py 里；这里只看 frame.interpolation 记下的名字
    with vidlo.VideoReader(paths["hevc_full_bt709"], crop=(16, 8, 288, 224), width=192, height=160, interpolation="nearest_exact") as video:
        first = next(iter(video))
    checks["resize_nearest_exact_recorded"] = {"passed": first.interpolation == "nearest_exact"}
    # 只给高：宽保持裁剪后的 288；给的尺寸与裁剪后相同：不缩放，interpolation 为 None
    with vidlo.VideoReader(paths["hevc_full_bt709"], crop=(16, 8, 288, 224), height=112) as video:
        first = next(iter(video))
    checks["resize_height_only_keeps_width"] = {"passed": first.image.shape == (112, 288, 3) and first.interpolation == "area",
                                                "shape": list(first.image.shape), "interpolation": first.interpolation}
    with vidlo.VideoReader(paths["hevc_full_bt709"], crop=(16, 8, 288, 224), width=288, height=224, interpolation="cubic") as video:
        first = next(iter(video))
    checks["resize_same_size_not_scaled"] = {"passed": first.image.shape == (224, 288, 3) and first.interpolation is None,
                                             "shape": list(first.image.shape), "interpolation": first.interpolation}
    # 实际用的矩阵：默认按录像标记（BT.709、smpte170m 即 BT.601），显式指定时不看标记
    for name, clip, options, expected in (("default_bt709", "hevc_full_bt709", {}, "bt709"), ("default_bt601", "hevc_full_bt601", {}, "bt601"),
                                          ("explicit_bt2020", "hevc_full_bt709", {"colorspace": "bt2020"}, "bt2020")):
        with vidlo.VideoReader(paths[clip], **options) as video:
            first = next(iter(video))
        checks[f"colorspace_{name}"] = {"passed": first.colorspace == expected, "colorspace": first.colorspace}

    # 选帧：顺序读按 step 或 fps 交出的帧号与画面
    nv12 = {"format": "nv12"}
    checks["select_step_nv12"] = check_selection(paths["hevc_full_bframes"], nv12, StrideSampler(3), step=3)
    checks["select_fps_nv12"] = check_selection(paths["hevc_full_bframes"], nv12, EMDownSampler(25, 10), fps=10, source_fps=25)
    checks["select_step_rgb"] = check_selection(paths["hevc_full_bt709"], {}, StrideSampler(2), step=2)
    checks["select_fps_sparse"] = check_selection(paths["hevc_full_bt709"], {}, EMDownSampler(30, 1), fps=1, source_fps=30)
    bad_selections = {"step_zero": {"step": 0}, "step_bool": {"step": True}, "step_float": {"step": 2.0},
                      "step_with_fps": {"step": 2, "fps": 30, "source_fps": 60}, "fps_alone": {"fps": 30},
                      "source_fps_alone": {"source_fps": 60}, "fps_above_source": {"fps": 60, "source_fps": 30},
                      "fps_infinite": {"fps": 30, "source_fps": float("inf")}, "fps_bool": {"fps": True, "source_fps": 60}}
    for name, selection in bad_selections.items():
        checks[f"select_rejects_{name}"] = expect_raises(ValueError, lambda s=selection: sampled(paths["hevc_full_bframes"], nv12, **s))

    # 码流里的参数集：与 codec_config 相同时照常读；改动后报错
    mismatch = videos / "hevc_inband_ps_mismatch.mkv"
    make_inband_mismatch(paths["hevc_inband_ps"], mismatch)
    checks["inband_parameter_sets_present"] = {"passed": nal["hevc_inband_ps"].get("SPS", 0) > 1, "nal_types": nal["hevc_inband_ps"]}
    checks["inband_parameter_set_mismatch"] = expect_raises(vidlo.UnsupportedError, lambda: decode_all(mismatch))

    # 容器与码流的颜色标记冲突：用 hevc_metadata 只改 SPS，容器的 Colour 元素保持原样（full range、BT.709）
    for name, change in (("matrix", "matrix_coefficients=5"), ("range", "video_full_range_flag=0")):
        conflict = videos / f"hevc_full_bt709_{name}_conflict.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(paths["hevc_full_bt709"]), "-c", "copy", "-bsf:v", f"hevc_metadata={change}",
                        str(conflict)], check=True)
        checks[f"rgb_container_{name}_conflict_rejected"] = expect_raises(vidlo.Error, lambda c=conflict: vidlo.VideoReader(c))
        checks[f"rgb_container_{name}_conflict_rejected"]["passed"] &= "disagree" in checks[f"rgb_container_{name}_conflict_rejected"]["message"]
        checks[f"yuv_container_{name}_conflict_reads"] = {"passed": decode_all(conflict) == decode_all(paths["hevc_full_bt709"])}

    # 未实现的情况与参数错误
    def nv12_open(source: Any, **options: Any) -> vidlo.VideoReader:
        return vidlo.VideoReader(source, format="nv12", **options)

    good = paths["hevc_full_bframes"]
    checks["unsupported_h264"] = expect_raises(vidlo.UnsupportedError, lambda: nv12_open(paths["h264"]))
    checks["unsupported_mp4"] = expect_raises(vidlo.UnsupportedError, lambda: nv12_open(paths["hevc_mp4"]))
    checks["unsupported_10bit"] = expect_raises(vidlo.UnsupportedError, lambda: decode_all(paths["hevc_10bit"]))
    checks["unsupported_444"] = expect_raises(vidlo.UnsupportedError, lambda: decode_all(paths["hevc_444"]))
    checks["unsupported_open_gop"] = expect_raises(vidlo.UnsupportedError, lambda: decode_all(paths["hevc_open_gop"]))
    truncated = videos / "hevc_full_bframes_truncated.mkv"
    missing_pts = make_truncated(good, truncated)
    checks["truncated_file"] = check_truncated(good, truncated, missing_pts)
    truncated_unknown = videos / "hevc_full_bframes_truncated_unknown_segment_size.mkv"
    make_unknown_size_segment(truncated, truncated_unknown)
    checks["truncated_file_unknown_segment_size"] = check_truncated(good, truncated_unknown, missing_pts)
    checks["open_gop_clip_has_cra"] = {"passed": nal["hevc_open_gop"].get("CRA_NUT", 0) > 0, "nal_types": nal["hevc_open_gop"]}
    checks["positive_clips_have_no_cra"] = {"passed": all(nal[n].get("CRA_NUT", 0) == 0 for n in POSITIVE),
                                            "nal_types": {n: nal[n] for n in POSITIVE}}
    checks["rgb_limited_range_rejected"] = expect_raises(vidlo.UnsupportedError, lambda: vidlo.VideoReader(paths["hevc_limited_bframes"]))
    checks["rgb_chromaloc2_rejected"] = expect_raises(vidlo.UnsupportedError, lambda: vidlo.VideoReader(paths["hevc_full_chromaloc2"]))
    checks["yuv_chromaloc2_reads"] = {"passed": decode_all(paths["hevc_full_chromaloc2"]) > 0}
    # 默认按录像标记转换：没标矩阵时报错，显式给矩阵就能读
    checks["rgb_default_matrix_unspecified_rejected"] = expect_raises(vidlo.UnsupportedError, lambda: vidlo.VideoReader(good))
    checks["rgb_explicit_bt601_matrix_unspecified_reads"] = {"passed": decode_all(good, colorspace="bt601") == decode_all(good)}
    checks["rgb_crop_too_large"] = expect_raises(ValueError, lambda: vidlo.VideoReader(good, colorspace="bt601", crop=(0, 0, 322, 240)))
    checks["rgb_crop_outside"] = expect_raises(ValueError, lambda: vidlo.VideoReader(good, colorspace="bt601", crop=(64, 0, 260, 240)))
    checks["remote_bad_url"] = expect_raises(ValueError, lambda: nv12_open("s3://bucket-only"))
    checks["remote_unknown_option"] = expect_raises(ValueError, lambda: nv12_open("s3://bucket/key.mkv", storage_options={"endpoint": "x"}))
    checks["oss_region_required"] = expect_raises(ValueError, lambda: nv12_open("oss://bucket/key.mkv", storage_options={}))
    checks["oss_internal_not_bool"] = expect_raises(
        ValueError, lambda: nv12_open("oss://bucket/key.mkv", storage_options={"region": "us-southeast-1", "use_internal_endpoint": "false"}))
    checks["remote_missing_credentials"] = check_missing_credentials()
    checks["bad_index"] = expect_raises(vidlo.Error, lambda: nv12_open(good, index={}))  # 按帧号读的完整检查见 check_random.py
    checks["bad_sample_index"] = expect_raises(vidlo.Error, lambda: vidlo.read_sample({}, {}, "/"))
    checks["unknown_scheme"] = expect_raises(ValueError, lambda: nv12_open("http://host/key.mkv"))
    checks["storage_options_for_local"] = expect_raises(ValueError, lambda: nv12_open(good, storage_options={"endpoint_url": "x"}))
    checks["missing_file"] = expect_raises(FileNotFoundError, lambda: nv12_open(videos / "missing.mkv"))
    for name, option in (("crop", {"crop": (0, 0, 64, 64)}), ("width", {"width": 64}), ("height", {"height": 64}),
                         ("interpolation", {"interpolation": "linear"}),
                         ("colorspace", {"colorspace": "bt709"})):
        checks[f"nv12_with_{name}"] = expect_raises(ValueError, lambda o=option: nv12_open(good, **o))
    checks["odd_crop"] = expect_raises(ValueError, lambda: vidlo.VideoReader(good, crop=(1, 0, 64, 64)))
    checks["unknown_colorspace"] = expect_raises(ValueError, lambda: vidlo.VideoReader(good, colorspace="source"))
    checks["old_nearest_name"] = expect_raises(ValueError, lambda: vidlo.VideoReader(good, interpolation="nearest"))
    checks["no_len"] = expect_raises(TypeError, lambda: len(nv12_open(good)))  # 只有按帧号读（有视频索引）时才有帧数

    # 打开、关闭的状态
    with nv12_open(good) as video:
        checks["get_frames_at_without_index"] = expect_raises(ValueError, lambda: video.get_frames_at([0]))
        checks["unsupported_submit"] = expect_raises(vidlo.UnsupportedError, lambda: video.submit([0]))
        iterator = iter(video)
        next(iterator)
        checks["read_twice"] = expect_raises(vidlo.Error, lambda: video.frames(step=2))
        checks["iter_twice"] = expect_raises(vidlo.Error, lambda: iter(video))
    checks["after_close"] = expect_raises(vidlo.Error, lambda: next(iterator))
    checks["metadata_after_close"] = {"passed": video.metadata.codec == "hevc"}
    checks["metadata_etag_local_none"] = {"passed": video.metadata.etag is None}
    native = nv12_open(good)._native
    native.close()
    checks["native_stats_after_close"] = expect_raises(vidlo.Error, native.stats)
    checks["native_config_after_close"] = expect_raises(vidlo.Error, native.decoder_config)
    checks["native_next_after_close"] = expect_raises(vidlo.Error, native.next_frame)
    checks["native_info_after_close"] = {"passed": native.info()["codec"] == "hevc"}

    # 视频索引：顺序读拼出的索引能通过检查；改坏的各种索引都被拒绝
    index = read_index(good)
    checks["index_from_sequential_read_valid"] = expect_no_error(lambda: vidlo.validate_video_index(index))
    first, second = index["decode_order"].index(0), index["decode_order"].index(1)
    bad_indexes: dict[str, Callable[[dict[str, Any]], Any]] = {
        "missing_field": lambda d: d.pop("file_size"),
        "unknown_field": lambda d: d.update(fps=60),
        "unknown_version": lambda d: d.update(version=2),
        "bad_base64": lambda d: d.update(codec_config="not base64!"),
        "length_mismatch": lambda d: d["packet_end"].pop(),
        "not_permutation": lambda d: d["decode_order"].__setitem__(second, 0),
        "start_after_order": lambda d: d["decode_start"].__setitem__(first, 1),
        "overlapping_ranges": lambda d: d["packet_end"].__setitem__(first, d["packet_start"][second] + 1),
        "past_file_size": lambda d: d.update(file_size=max(d["packet_end"]) - 1),
        "bool_as_int": lambda d: d["decode_start"].__setitem__(first, False),
    }
    for name, change in bad_indexes.items():
        checks[f"index_rejects_{name}"] = expect_raises(vidlo.Error, lambda c=change: vidlo.validate_video_index(broken(index, c)))
    checks["index_unsupported_codec"] = expect_raises(
        vidlo.UnsupportedError, lambda: vidlo.validate_video_index(broken(index, lambda d: d.update(codec="h264"))))

    # 样本索引
    sample = {"video": "PC117-1-081/20261022030227/rgb-20261022-030320.mkv", "frames": [3000, 3001, 3002]}
    checks["sample_index_valid"] = expect_no_error(lambda: vidlo.validate_sample_index(sample))
    bad_samples = {
        "absolute_path": {"video": "/data/a.mkv", "frames": [0]},
        "parent_path": {"video": "a/../../b.mkv", "frames": [0]},
        "empty_frames": {"video": "a.mkv", "frames": []},
        "negative_frame": {"video": "a.mkv", "frames": [-1]},
        "extra_field": {"video": "a.mkv", "frames": [0], "fps": 60},
    }
    for name, value in bad_samples.items():
        checks[f"sample_index_rejects_{name}"] = expect_raises(vidlo.Error, lambda v=value: vidlo.validate_sample_index(v))

    # 进程
    checks["cuda_not_initialized_at_import"] = check_cuda_not_initialized_at_import()
    checks["fork_child_decodes"] = check_fork(good, decode_all(good))
    for mode in ("close", "gc"):
        checks[f"fork_after_cuda_nv12_{mode}"] = check_fork_after_cuda(good, {"format": "nv12"}, mode, decode_all(good))
    bt709 = paths["hevc_full_bt709"]
    checks["fork_after_cuda_rgb_gc"] = check_fork_after_cuda(bt709, {"crop": [16, 8, 288, 224], "width": 192, "height": 160}, "gc",
                                                             decode_all(bt709))
    checks["fork_after_other_library_cuda"] = check_fork_after_other_cuda(good)

    report = {"vidlo_version": vidlo.__version__, "videos": {k: str(v) for k, v in paths.items()}, "nal_types": nal,
              "sps": sps_summary,
              "checks": checks, "passed": all(c["passed"] for c in checks.values())}
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    for name, check in checks.items():
        print(("PASS " if check["passed"] else "FAIL ") + name + ("" if check["passed"] else "  " + json.dumps(check, ensure_ascii=False)[:400]))
    print("ALL PASSED" if report["passed"] else "SOME FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
