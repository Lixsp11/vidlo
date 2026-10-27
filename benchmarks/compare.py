"""vidlo 与 torchcodec（CUDA）、OpenCV、decord（CPU）读同一段录像：顺序读速度、按帧号读的耗时与帧是否取对、从对象存储读一个样本的
耗时、请求数与读取量。README 的对比表来自这个脚本。每个库在装了它的环境里各跑一次（--lib），结果按行追加到 --out（JSON）。

所有库交出同样大小的画面：左右各裁 16 列（1280×720 → 1248×720），缩小到 832×480，CPU 上的 uint8 RGB（[高, 宽, 3]）。各库走自己
常用的路径，缩放算法因此不完全相同（结果里记下实际用的）：
- vidlo：VideoReader 的 crop、width、height（CV-CUDA 面积插值）；按帧号读用视频索引（--index，格式见 docs/index-format.md）。
- torchcodec：device="cuda" 解码，在 GPU 上裁剪、用 torch 的 area 插值缩放（自适应平均池化，与 OpenCV、CV-CUDA 的面积插值在
  1.5 倍缩小时不完全相同）再复制到 CPU。torchcodec 0.17 解码器自带的裁剪、缩放只支持 CPU 解码，用不上。顺序读按
  --torchcodec-batch 帧一批调 get_frames_in_range（1 为逐帧；--torchcodec-getitem 改用 decoder[i] 逐帧取）。按帧号读用 custom_frame_mappings（--mappings，ffprobe -show_frames 的 pts、duration、key_frame），与 vidlo 的视频索引一样是事先
  给出的每帧信息。
- OpenCV：cv2.VideoCapture（FFmpeg 后端）在 CPU 上解码成 BGR（不读码流的颜色标记，一律按 BT.601 转换），cv2.resize(INTER_AREA)
  后转 RGB。按帧号读：CAP_PROP_POS_FRAMES 跳到请求的第一帧，往后逐帧 grab，只对要的帧 retrieve。
- decord：VideoReader 在 CPU 上解码成 RGB，cv2.resize(INTER_AREA)。按帧号读用 get_batch。
CPU 库：--threads 同时设解码线程数与 cv2 的线程数（0 为库自己决定），所以一个进程的线程总数可能多于它。

模式（所有计时都在预热之后；本地录像先由调用方读进页缓存）：
- sequential：--procs 个进程（spawn）各自先初始化并读一帧预热，同时开始从头到尾逐帧读整段录像，总吞吐 = 各进程帧数之和 /
  (最晚结束 - 最早开始)，重复 --trials 次取中位数。每个进程交出的帧数必须等于 --frames。
- random：单进程，同一个读取器。先用同一个库顺序读一遍、记下要核对的帧的哈希（也作预热），两种请求各预热一个，再按请求清单
  （两种请求各 --count 个，每种请求一个独立的随机数序列，取前 k 个总是同一批）逐个计时。交出的每一帧都要与同一个库顺序读的
  同一帧逐位相同；帧数、形状不对直接报错。
- remote：单进程，从对象存储读：每个样本新开一个读取器、读一个 81 帧隔一帧的请求（训练时样本来自很多不同的录像），分别记下
  打开与读取的耗时、HTTP 请求数与收到的字节数（响应体）。先读 --warmup 个样本预热（建连接、CUDA），不计入。读完再与同一个库读本地
  文件的结果逐帧比较。vidlo 用自己的 storage_options；其余三个库读同一个文件对象：同一进程共用一个 SDK 客户端，关掉 SDK 的
  重试（与 vidlo 相同），对象大小由调用方给出（与 vidlo 的视频索引一样，不发 HEAD），read 时缓存里没有的部分发一个 Range GET，
  从当前位置起至少取 --readahead 字节，只保留最近一块。目前只支持 oss://（凭证取自 OSS_ACCESS_KEY_ID、OSS_ACCESS_KEY_SECRET）。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import platform
import random
import re
import statistics
import subprocess
import time
from typing import Any, Callable

CROP_X, CROP_WIDTH, SOURCE_HEIGHT = 16, 1248, 720
WIDTH, HEIGHT = 832, 480
SHAPES = {"16 consecutive": (16, 1), "81 every 2nd": (81, 2)}


def requests_for(total: int, seed: int, shape: str, count: int) -> list[list[int]]:
    """一种请求的清单：每种请求一个独立的随机数序列，取前 k 个总是同一批。"""
    n, step = SHAPES[shape]
    rng = random.Random(f"{seed}/{shape}")
    starts = [rng.randrange(total - (n - 1) * step) for _ in range(count)]
    return [list(range(s, s + (n - 1) * step + 1, step)) for s in starts]


def sha(image: Any) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    """最近秩法：排序后第 ceil(q·n) 个，不超过最大值。"""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarize(values: list[float]) -> dict:
    out = {"p50_ms": round(statistics.median(values) * 1000, 1)}
    if len(values) >= 10:
        out["p90_ms"] = round(percentile(values, 0.9) * 1000, 1)
    return out


def check_images(images: list[Any], frames: list[int]) -> None:
    if len(images) != len(frames):
        raise RuntimeError(f"got {len(images)} frames for a request of {len(frames)}")
    for image in images:
        if image.shape != (HEIGHT, WIDTH, 3) or str(image.dtype) != "uint8":
            raise RuntimeError(f"got an image of shape {image.shape} {image.dtype}")


# ---------- 各库的读法：open(源) → 读取器；frames(读取器) 逐帧交出；get(读取器, 帧号) 按帧号交出；close ----------

class Vidlo:
    def __init__(self, args: argparse.Namespace) -> None:
        import vidlo

        self.vidlo, self.args = vidlo, args
        self.index = vidlo.load_video_index(args.index) if args.index else None
        self.settings = {"crop": (CROP_X, 0, CROP_WIDTH, SOURCE_HEIGHT), "width": WIDTH, "height": HEIGHT}

    def describe(self) -> dict:
        return {"vidlo": self.vidlo.__version__, "resize": "CV-CUDA area", "by_number_metadata": "vidlo video index"}

    def open(self, source: Any, by_number: bool) -> Any:
        options = json.loads(self.args.storage_options) if isinstance(source, str) and "://" in source else None
        return self.vidlo.VideoReader(source, index=self.index if by_number else None, storage_options=options, **self.settings)

    def frames(self, reader: Any):
        for frame in reader:
            yield frame.image

    def get(self, reader: Any, numbers: list[int]) -> list[Any]:
        return list(reader.get_frames_at(numbers).images)

    def remote_counts(self, reader: Any) -> tuple[int, int]:
        stats = reader.stats
        return stats["requests"], stats["bytes_read"]

    def close(self, reader: Any) -> None:
        reader.close()


class TorchCodec:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        import torchcodec
        from torchcodec.decoders import VideoDecoder

        self.torch, self.torchcodec, self.VideoDecoder = torch, torchcodec, VideoDecoder
        self.batch, self.getitem = args.torchcodec_batch, args.torchcodec_getitem
        self.mappings = open(args.mappings).read() if args.mappings else None

    def describe(self) -> dict:
        sequential = "decoder[i]" if self.getitem else f"get_frames_in_range, {self.batch} frame(s) per call"
        return {"torchcodec": self.torchcodec.__version__, "torch": self.torch.__version__, "resize": "torch area (adaptive average pooling)",
                "sequential_calls": sequential,
                "by_number_metadata": "custom_frame_mappings from ffprobe"}

    def open(self, source: Any, by_number: bool) -> Any:
        return self.VideoDecoder(source, device="cuda", seek_mode="exact", custom_frame_mappings=self.mappings)

    def _process(self, data: Any) -> Any:
        torch = self.torch
        data = data[..., CROP_X:CROP_X + CROP_WIDTH].float()
        data = torch.nn.functional.interpolate(data, size=(HEIGHT, WIDTH), mode="area").round_().clamp_(0, 255).to(torch.uint8)
        return data.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def frames(self, decoder: Any):
        total = len(decoder)
        if self.getitem:
            for i in range(total):  # [3, 高, 宽] 在 GPU 上
                yield self._process(decoder[i][None])[0]
            return
        for start in range(0, total, self.batch):
            yield from self._process(decoder.get_frames_in_range(start, min(start + self.batch, total)).data)

    def get(self, decoder: Any, numbers: list[int]) -> list[Any]:
        return list(self._process(decoder.get_frames_at(numbers).data))

    def close(self, decoder: Any) -> None:
        if decoder.cpu_fallback:
            raise RuntimeError(f"torchcodec fell back to CPU: {decoder.cpu_fallback}")


class OpenCV:
    def __init__(self, args: argparse.Namespace) -> None:
        import cv2

        self.cv2, self.threads = cv2, args.threads
        cv2.setNumThreads(args.threads if args.threads > 0 else -1)

    def describe(self) -> dict:
        avcodec = [line.strip() for line in self.cv2.getBuildInformation().splitlines() if "avcodec:" in line]
        return {"opencv": self.cv2.__version__, "opencv_avcodec": avcodec[0] if avcodec else None, "resize": "cv2 INTER_AREA",
                "by_number_metadata": "none"}

    def open(self, source: Any, by_number: bool) -> Any:
        cv2 = self.cv2
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, self.threads])
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open the video")
        return cap

    def _process(self, image: Any) -> Any:
        cv2 = self.cv2
        image = cv2.resize(image[:, CROP_X:CROP_X + CROP_WIDTH], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def frames(self, cap: Any):
        while True:
            ok, image = cap.read()
            if not ok:
                return
            yield self._process(image)

    def get(self, cap: Any, numbers: list[int]) -> list[Any]:
        first, last, wanted, got = min(numbers), max(numbers), set(numbers), {}
        if not cap.set(self.cv2.CAP_PROP_POS_FRAMES, first):
            raise RuntimeError(f"OpenCV could not seek to frame {first}")
        for n in range(first, last + 1):
            if not cap.grab():
                raise RuntimeError(f"OpenCV could not read frame {n}")
            if n in wanted:
                ok, image = cap.retrieve()
                if not ok:
                    raise RuntimeError(f"OpenCV could not decode frame {n}")
                got[n] = self._process(image)
        return [got[n] for n in numbers]

    def close(self, cap: Any) -> None:
        cap.release()


class Decord:
    def __init__(self, args: argparse.Namespace) -> None:
        import cv2
        import decord

        self.cv2, self.decord, self.threads = cv2, decord, args.threads
        cv2.setNumThreads(args.threads if args.threads > 0 else -1)

    def describe(self) -> dict:
        return {"decord": self.decord.__version__, "opencv_for_resize": self.cv2.__version__, "resize": "cv2 INTER_AREA",
                "by_number_metadata": "none"}

    def open(self, source: Any, by_number: bool) -> Any:
        return self.decord.VideoReader(source, ctx=self.decord.cpu(0), num_threads=self.threads)

    def _process(self, image: Any) -> Any:
        cv2 = self.cv2
        return cv2.resize(image[:, CROP_X:CROP_X + CROP_WIDTH], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    def frames(self, reader: Any):
        for _ in range(len(reader)):
            yield self._process(reader.next().asnumpy())

    def get(self, reader: Any, numbers: list[int]) -> list[Any]:
        return [self._process(image) for image in reader.get_batch(numbers).asnumpy()]

    def close(self, reader: Any) -> None:
        pass


LIBS: dict[str, Callable[[argparse.Namespace], Any]] = {"vidlo": Vidlo, "torchcodec": TorchCodec, "opencv": OpenCV, "decord": Decord}


# ---------- 对象存储的文件对象（torchcodec、OpenCV、decord 用） ----------

_CLIENTS: dict[str, Any] = {}


def oss_client(storage_options: str) -> Any:
    """同一进程共用一个客户端；关掉 SDK 的重试，请求数就是实际发出的请求数（与 vidlo 相同）。"""
    if storage_options not in _CLIENTS:
        import alibabacloud_oss_v2 as oss

        options = json.loads(storage_options)
        cfg = oss.config.load_default()
        cfg.credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
        cfg.region = options["region"]
        cfg.use_internal_endpoint = options.get("use_internal_endpoint", False)
        if "endpoint" in options:
            cfg.endpoint = options["endpoint"]
        cfg.retryer = oss.retry.NopRetryer()
        _CLIENTS[storage_options] = (oss, oss.Client(cfg))
    return _CLIENTS[storage_options]


class OssFile(io.BufferedIOBase):
    """read 时缓存里没有的部分发一个 Range GET，从当前位置起至少取 readahead 字节，只保留最近一块；核对 206 与 Content-Range。"""

    def __init__(self, url: str, size: int, readahead: int, storage_options: str) -> None:
        super().__init__()
        self.oss, self.client = oss_client(storage_options)
        self.bucket, _, self.key = url.removeprefix("oss://").partition("/")
        self.size, self.readahead = size, readahead
        self.requests, self.bytes = 0, 0
        self.pos, self.cache_start, self.cache = 0, 0, b""

    def _get(self, start: int, end: int) -> bytes:
        self.requests += 1
        request = self.oss.GetObjectRequest(bucket=self.bucket, key=self.key, range_header=f"bytes={start}-{end - 1}",
                                            range_behavior="standard")
        result = self.client.get_object(request)
        data = result.body.read()
        if result.status_code != 206 or result.content_range != f"bytes {start}-{end - 1}/{self.size}" or len(data) != end - start:
            raise RuntimeError(f"range {start}-{end}: HTTP {result.status_code}, {result.content_range}, {len(data)} bytes")
        self.bytes += len(data)
        return data

    def read(self, n: int | None = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b""
        if not (self.cache_start <= self.pos and end <= self.cache_start + len(self.cache)):
            self.cache_start, self.cache = self.pos, self._get(self.pos, min(self.size, max(end, self.pos + self.readahead)))
        data = self.cache[self.pos - self.cache_start:end - self.cache_start]
        self.pos = end
        return data

    def read1(self, n: int = -1) -> bytes:
        return self.read(n)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


# ---------- 三种模式 ----------

def environment() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    with open("/proc/cpuinfo") as f:
        cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), None)
    with open("/proc/self/maps") as f:
        avcodec = sorted({line.split()[-1] for line in f if re.search(r"libavcodec", line)})
    return {"python": platform.python_version(), "gpu": gpu, "cpu": cpu, "cpus_available": len(os.sched_getaffinity(0)),
            "libavcodec_loaded": avcodec}


def sequential_worker(args: argparse.Namespace, barrier: Any, queue: Any) -> None:
    try:
        lib = LIBS[args.lib](args)
        reader = lib.open(args.video, by_number=False)  # 预热：初始化库、CUDA，读一帧
        next(iter(lib.frames(reader)))
        lib.close(reader)
        del reader
        gc.collect()
        barrier.wait(timeout=600)
        started, cpu = time.time(), time.process_time()
        reader = lib.open(args.video, by_number=False)
        n = 0
        for image in lib.frames(reader):
            n += 1
        lib.close(reader)
        del reader
        ended = time.time()
        if image.shape != (HEIGHT, WIDTH, 3):
            raise RuntimeError(f"got an image of shape {image.shape}")
        queue.put({"frames": n, "start": started, "end": ended, "cpu_s": time.process_time() - cpu, "environment": environment()})
    except BaseException as e:  # noqa: BLE001 - 报给父进程
        barrier.abort()
        queue.put({"error": f"{type(e).__name__}: {e}"})


def run_sequential(args: argparse.Namespace) -> dict:
    context = mp.get_context("spawn")
    trials = []
    for _ in range(args.trials):
        barrier, queue = context.Barrier(args.procs), context.Queue()
        workers = [context.Process(target=sequential_worker, args=(args, barrier, queue)) for _ in range(args.procs)]
        for w in workers:
            w.start()
        got = [queue.get(timeout=3600) for _ in workers]
        for w in workers:
            w.join()
        errors = [g["error"] for g in got if "error" in g]
        if errors:
            raise RuntimeError(f"sequential workers failed: {errors[:3]}")
        if any(g["frames"] != int(args.frames) for g in got):
            raise RuntimeError(f"a worker read {sorted({g['frames'] for g in got})} frames, expected {args.frames}")
        wall = max(g["end"] for g in got) - min(g["start"] for g in got)
        trials.append({"frames_per_s": round(args.procs * int(args.frames) / wall, 1), "wall_s": round(wall, 2),
                       "cpu_cores": round(sum(g["cpu_s"] for g in got) / wall, 2)})
    best = sorted(trials, key=lambda t: t["frames_per_s"])[len(trials) // 2]
    return {"frames_per_s": best["frames_per_s"], "trials": trials, "worker_environment": got[0]["environment"]}


def run_random(args: argparse.Namespace, lib: Any) -> dict:
    total = int(args.frames)
    plan = {shape: requests_for(total, args.seed, shape, args.count) for shape in SHAPES}
    needed = {n for requests in plan.values() for frames in requests for n in frames}
    reader = lib.open(args.video, by_number=False)
    reference, decoded = {}, 0
    for n, image in enumerate(lib.frames(reader)):
        if n in needed:
            reference[n] = sha(image)
        decoded += 1
    lib.close(reader)
    del reader
    if decoded != total or len(reference) != len(needed):
        raise RuntimeError(f"sequential read gave {decoded} frames, expected {total}")
    reader = lib.open(args.video, by_number=True)
    for shape in SHAPES:  # 每种请求预热一个
        lib.get(reader, requests_for(total, args.seed + 1, shape, 1)[0])
    result: dict[str, Any] = {}
    for shape, requests in plan.items():
        times, wrong = [], 0
        for frames in requests:
            started = time.perf_counter()
            images = lib.get(reader, frames)
            times.append(time.perf_counter() - started)
            check_images(images, frames)
            wrong += sum(sha(image) != reference[n] for n, image in zip(frames, images))
        result[shape] = {**summarize(times), "requests": len(requests), "frames": sum(len(f) for f in requests), "wrong_frames": wrong}
    lib.close(reader)
    return result


def run_remote(args: argparse.Namespace, lib: Any) -> dict:
    total = int(args.frames)
    shape = "81 every 2nd"
    # 预热样本与计时样本各用一个随机数序列；分几轮交替跑时，每轮用 --sample-offset 取计时清单里不同的一段
    plan = (requests_for(total, args.seed + 3, shape, args.warmup)
            + requests_for(total, args.seed + 2, shape, args.sample_offset + args.samples)[args.sample_offset:])
    rows, outputs = [], []
    for i, frames in enumerate(plan):
        started = time.perf_counter()
        source = args.source if args.lib == "vidlo" else OssFile(args.source, args.file_size, args.readahead, args.storage_options)
        reader = lib.open(source, by_number=True)
        opened = time.perf_counter()
        open_counts = lib.remote_counts(reader) if args.lib == "vidlo" else (source.requests, source.bytes)
        images = lib.get(reader, frames)
        done = time.perf_counter()
        counts = lib.remote_counts(reader) if args.lib == "vidlo" else (source.requests, source.bytes)
        lib.close(reader)
        del reader, source
        gc.collect()
        check_images(images, frames)
        if i < args.warmup:
            continue  # 预热的样本不计
        outputs.append((frames, [sha(image) for image in images]))
        rows.append({"start": frames[0], "open_s": round(opened - started, 4), "read_s": round(done - opened, 4),
                     "total_s": round(done - started, 4), "open_requests": open_counts[0], "open_bytes": open_counts[1],
                     "requests": counts[0], "bytes": counts[1]})
    wrong = 0
    reader = lib.open(args.video, by_number=True)
    for frames, hashes in outputs:  # 计时之后再与同一个库读本地文件的结果比较
        wrong += sum(h != sha(image) for h, image in zip(hashes, lib.get(reader, frames)))
    lib.close(reader)
    return {"shape": shape, "samples": len(rows), **summarize([r["total_s"] for r in rows]),
            "open_p50_ms": round(statistics.median(r["open_s"] for r in rows) * 1000, 1),
            "requests_per_sample": statistics.median(r["requests"] for r in rows),
            "open_requests_per_sample": statistics.median(r["open_requests"] for r in rows),
            "mb_per_sample": round(statistics.median(r["bytes"] for r in rows) / 1e6, 2),
            "open_mb_per_sample": round(statistics.median(r["open_bytes"] for r in rows) / 1e6, 2),
            "wrong_frames": wrong, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lib", required=True, choices=list(LIBS))
    parser.add_argument("--mode", required=True, choices=["sequential", "random", "remote"])
    parser.add_argument("--video", required=True, help="本地录像")
    parser.add_argument("--frames", required=True, help="录像的帧数")
    parser.add_argument("--file-size", type=int, help="录像的字节数（remote 的文件对象用）")
    parser.add_argument("--index", help="vidlo 的视频索引（JSON）")
    parser.add_argument("--mappings", help="torchcodec 的 custom_frame_mappings（ffprobe -show_frames 的 JSON）")
    parser.add_argument("--torchcodec-batch", type=int, default=1, help="torchcodec 顺序读每次取几帧")
    parser.add_argument("--torchcodec-getitem", action="store_true", help="torchcodec 顺序读用 decoder[i] 逐帧取")
    parser.add_argument("--procs", type=int, default=1, help="sequential 的进程数")
    parser.add_argument("--trials", type=int, default=3, help="sequential 重复几次，取中位数")
    parser.add_argument("--threads", type=int, default=0, help="CPU 库的解码与缩放线程数，0 为库自己决定")
    parser.add_argument("--count", type=int, default=30, help="random 每种请求的个数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", help="remote 的对象存储地址（oss://桶/键）")
    parser.add_argument("--storage-options", default='{"region": "us-southeast-1", "use_internal_endpoint": true}',
                        help="连接参数（JSON）：vidlo 的 storage_options，文件对象也用同样的设置")
    parser.add_argument("--samples", type=int, default=10, help="remote 计时的样本数")
    parser.add_argument("--warmup", type=int, default=1, help="remote 先读几个不计时的样本")
    parser.add_argument("--sample-offset", type=int, default=0, help="remote 从样本清单的第几个开始（分几轮交替跑时用）")
    parser.add_argument("--readahead", type=int, default=1 << 20, help="文件对象每个 Range GET 至少取的字节数")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    record: dict[str, Any] = {"lib": args.lib, "mode": args.mode, "arguments": {k: v for k, v in vars(args).items() if k != "out"}}
    if args.mode == "sequential":
        record.update(run_sequential(args))
    lib = LIBS[args.lib](args)
    record["library"] = lib.describe()
    if args.mode == "random":
        record.update(run_random(args, lib))
    elif args.mode == "remote":
        record.update(run_remote(args, lib))
    record["environment"] = environment()
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("environment", "worker_environment", "rows", "arguments")}),
          flush=True)


if __name__ == "__main__":
    main()
