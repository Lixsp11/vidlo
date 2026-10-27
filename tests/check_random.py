"""按帧号读与顺序读逐位比较，并核对每次请求只读了规划的字节区间、只解了该解的包。

先顺序读一遍（format="nv12"），按 docs/index-format.md 的做法建视频索引，同时记下每帧 NV12 的 SHA256 作为参照；顺序读与
FFmpeg CPU 解码逐位一致由 compare_ffmpeg_yuv.py 保证。再用这份索引按帧号读，逐个请求核对：交出的帧号与顺序、每帧像素的
SHA256、逐帧信息、读取的字节区间、送进解码器与跳过的包数、解码器是新建还是取自缓存。读取区间与该跳过哪些包由测试自己
按索引和文件里各包的 NAL 头算出，不用库里的规划。请求覆盖 GOP 边界、首尾帧、只有一帧、最后不满的 GOP、跨 2～5 个 GOP、
倒序、重复、负下标、16 帧连续与 81 帧隔一帧的随机片段。

配置：默认（解码器缓存容量 1、跳过没人参考的非目标帧）每段录像跑全部请求；给了两段以上录像时，几个读取器交替请求，
解码器跨文件复用；缓存与跳帧的另外三种组合（缓存关闭且不跳帧就是第 1 步的做法）各跑一部分请求。full range 的录像另比
RGB：同样的裁剪缩放（左右各裁 16 列，缩到 2/3）顺序读一遍作参照，按帧号读一部分请求逐帧比较。多时间层的录像（如 x265
temporal-layers=3）里，低层的 sub-layer non-reference 帧会被高层参考，不能跳过，报告里分层列出数量。另测应当报错的情况：
越界、类型不对、读取器模式不对、索引的文件大小不符、decode_start 指向的不是 GOP 入口（检查时发现）或不是 IDR（解码时
发现，含把可跳过的帧冒充成入口）、出错后的请求、fork（子进程报错，且拿到的是新的空缓存）。

输出目录必须不存在，结果写进 report.json。
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

import vidlo

FIELDS = ("decode_order", "decode_start", "packet_start", "packet_end")
SUCCESS = 2  # cuvidDecodeStatus_Success


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def nal_units(data: bytes, length_size: int):
    offset = 0
    while offset < len(data):
        length = int.from_bytes(data[offset:offset + length_size], "big")
        offset += length_size
        yield data[offset:offset + length]
        offset += length


def hvcc_layout(config: bytes) -> tuple[int, int]:
    """hvcC 的 NAL 长度字段字节数，以及 SPS 的 sps_max_sub_layers_minus1（最高 TemporalId）。"""
    length_size, max_tid, offset = (config[21] & 3) + 1, 0, 23
    for _ in range(config[22]):
        nal_type, count = config[offset] & 0x3F, int.from_bytes(config[offset + 1:offset + 3], "big")
        offset += 3
        for _ in range(count):
            length = int.from_bytes(config[offset:offset + 2], "big")
            if nal_type == 33:
                max_tid = max(max_tid, (config[offset + 4] >> 1) & 7)  # 2 字节长度、2 字节 NAL 头之后的第一个字节
            offset += 2 + length
    return length_size, max_tid


def unreferenced_frames(video: Path, index: dict) -> tuple[list[bool], dict[str, int]]:
    """每帧的图像是不是在最高时间层、类型为 sub-layer non-reference（H.265 表 7-1 里 14 以内的偶数类型）：没有图像参考它。
    包里还有参数集、分隔符、填充、SEI 之外的非 VCL NAL（序列结束等）时整包不能丢，按“有用”算。另按时间层数一下
    sub-layer non-reference 的帧：低于最高层的会被高层参考，不能跳。"""
    length_size, max_tid = hvcc_layout(base64.b64decode(index["codec_config"]))
    out = []
    layers: dict[str, int] = {}
    with open(video, "rb") as stream:
        for start, end in zip(index["packet_start"], index["packet_end"]):
            stream.seek(start)
            slice_type, slice_tid, stateful = None, None, False
            for nal in nal_units(stream.read(end - start), length_size):
                nal_type = (nal[0] >> 1) & 0x3F
                if nal_type <= 31:
                    if slice_type is None:
                        slice_type, slice_tid = nal_type, (nal[1] & 7) - 1
                elif nal_type not in (32, 33, 34, 35, 38, 39, 40):
                    stateful = True
            non_reference = slice_type <= 14 and slice_type % 2 == 0
            if non_reference:
                key = f"temporal_id {slice_tid}" + (" (highest, skippable)" if slice_tid == max_tid else " (referenced by higher layers)")
                layers[key] = layers.get(key, 0) + 1
            out.append(non_reference and slice_tid == max_tid and not stateful)
    return out, layers


def sequential_reference(video: Path, device: int) -> tuple[dict, list[str]]:
    arrays = {name: [] for name in FIELDS}
    hashes = []
    with vidlo.VideoReader(video, device=device, format="nv12") as reader:
        for frame in reader:
            if frame.number != len(hashes):
                raise RuntimeError(f"frame number {frame.number} at position {len(hashes)}")
            for name in FIELDS:
                arrays[name].append(getattr(frame, name))
            hashes.append(sha(frame.image))
        metadata = reader.metadata
    index = {"version": 1, "file_size": metadata.file_size, "codec": metadata.codec,
             "codec_config": base64.b64encode(metadata.codec_config).decode("ascii"), **arrays}
    vidlo.validate_video_index(index)
    return index, hashes


def requests(index: dict, seed: int, clips: int) -> list[tuple[str, list[int]]]:
    count = len(index["decode_order"])
    keys = [f for f in range(count) if index["decode_start"][f] == index["decode_order"][f]]
    rng = random.Random(seed)
    out: list[tuple[str, list[int]]] = [("first", [0]), ("last", [count - 1]), ("last then first", [count - 1, 0]),
                                        ("negative", [-1, count - 1, -count]), ("tail GOP", list(range(keys[-1], count)))]
    picked = sorted(set(keys[:3] + keys[len(keys) // 2:len(keys) // 2 + 2] + keys[-3:]))
    for k in picked:
        around = [f for f in (k - 1, k, k + 1) if 0 <= f < count]
        out += [(f"GOP start {k}", [k]), (f"GOP start {k} and neighbours", around)]
        if k > 0:
            out.append((f"before GOP start {k}", [k - 1]))
    for span in (2, 3, 5):
        i = rng.randrange(0, len(keys) - span)
        out.append((f"{span} GOPs from {keys[i]}", list(range(keys[i], keys[i + span], 7))))
    a = rng.randrange(100, count - 100)
    out += [("reverse", list(range(a + 20, a, -1))), ("repeats", [a, a, a + 30, a]),
            ("across GOPs out of order", [keys[5] + 5, keys[2] + 2, keys[5] + 5])]
    for _ in range(clips):
        s = rng.randrange(0, count - 16)
        out.append((f"16 from {s}", list(range(s, s + 16))))
    for _ in range(clips):
        s = rng.randrange(0, count - 161)
        out.append((f"81 every 2nd from {s}", list(range(s, s + 161, 2))))
    return out


def expected_plan(index: dict, numbers: list[int], unreferenced: list[bool], skip: bool) -> tuple[int, int, int, int, int]:
    """按 docs/index-format.md 的规则独立算出：解码顺序上的首尾包 s、e，读取区间 [a, b)，以及该跳过的包数。"""
    frame_at = {order: f for f, order in enumerate(index["decode_order"])}
    targets = set(numbers)
    s = min(index["decode_start"][f] for f in targets)
    e = max(index["decode_order"][f] for f in targets)
    # 索引标成 GOP 入口的包照送（库要在解码时核对它是 IDR）
    skipped = sum(1 for k in range(s, e + 1) if skip and unreferenced[frame_at[k]] and frame_at[k] not in targets
                  and index["decode_start"][frame_at[k]] != k)
    return s, e, index["packet_start"][frame_at[s]], index["packet_end"][frame_at[e]], skipped


def check_request(reader: vidlo.VideoReader, ref: dict, label: str, frames: list[int], skip: bool = True) -> dict:
    index, hashes = ref["index"], ref["hashes"]
    count = len(hashes)
    numbers = [f % count for f in frames]
    s, e, a, b, skipped = expected_plan(index, numbers, ref["unreferenced"], skip)
    before = reader.stats
    started = time.perf_counter()
    batch = reader.get_frames_at(frames)
    elapsed = time.perf_counter() - started
    after = reader.stats
    problems = []
    height, width = (reader.metadata.height * 3 // 2, reader.metadata.width)
    if batch.images.shape != (len(frames), height, width) or batch.images.dtype != np.uint8:
        problems.append(f"images shape {batch.images.shape} {batch.images.dtype}")
    if batch.number != numbers:
        problems.append(f"numbers {batch.number[:10]}")
    wrong = [n for i, n in enumerate(numbers) if i < len(batch.images) and sha(batch.images[i]) != hashes[n]]
    if wrong:
        problems.append(f"{len(wrong)} frames differ from the sequential read, first {wrong[:5]}")
    for name in FIELDS:
        if getattr(batch, name) != [index[name][n] for n in numbers]:
            problems.append(f"{name} differs from the index")
    key_frame = [index["decode_start"][n] == index["decode_order"][n] for n in numbers]
    if batch.key_frame != key_frame or batch.pts != [None] * len(numbers) or batch.decode_status != [SUCCESS] * len(numbers):
        problems.append("key_frame, pts or decode_status wrong")
    if batch.colorspace != [None] * len(numbers) or batch.interpolation != [None] * len(numbers):
        problems.append("colorspace or interpolation should be None for nv12")
    fed = e - s + 1 - skipped
    keys = ("requests", "bytes_read", "packets", "skipped", "decode_calls", "displayed", "kept")
    delta = {k: after[k] - before[k] for k in keys}
    want = {"requests": 1, "bytes_read": b - a, "packets": fed, "skipped": skipped, "decode_calls": fed, "displayed": fed,
            "kept": len(set(numbers))}
    if delta != want:
        problems.append(f"stats delta {delta}, expected {want}")
    reused = after["decoder_reuses"] - before["decoder_reuses"]
    if reused + after["decoder_creations"] - before["decoder_creations"] != 1:
        problems.append("decoder neither created nor reused once")
    if tuple(after["last_range"]) != (a, b):
        problems.append(f"read range {after['last_range']}, expected {(a, b)}")
    # 重复的帧号各占一份数据：改一份，另一份不变
    if len(set(numbers)) < len(numbers):
        first = numbers.index(numbers[-1])
        if first != len(numbers) - 1:
            keep = batch.images[first].copy()
            batch.images[-1][0, 0] ^= 0xFF
            if not np.array_equal(batch.images[first], keep):
                problems.append("repeated frames share memory")
            batch.images[-1][0, 0] ^= 0xFF
    return {"request": label, "frames": len(frames), "packets": e - s + 1, "fed": fed, "skipped": skipped, "reused_decoder": bool(reused),
            "range": [a, b], "bytes": b - a, "elapsed_s": round(elapsed, 4), "passed": not problems, "problems": problems}


def expect_error(name: str, kind: type[BaseException], action) -> dict:
    try:
        action()
    except kind as e:
        return {"check": name, "passed": True, "raised": f"{type(e).__name__}: {e}"}
    except BaseException as e:  # noqa: BLE001 - 记下意料之外的异常类型
        return {"check": name, "passed": False, "raised": f"{type(e).__name__}: {e}"}
    return {"check": name, "passed": False, "raised": None}


def error_checks(ref: dict, device: int) -> list[dict]:
    video, index = ref["video"], ref["index"]
    count = len(index["decode_order"])
    results = []
    with vidlo.VideoReader(video, device=device, format="nv12", index=index) as reader:
        stats = reader.stats
        empty = reader.get_frames_at([])
        results.append({"check": "empty request reads nothing",
                        "passed": empty.images.shape == (0, reader.metadata.height * 3 // 2, reader.metadata.width) and reader.stats == stats})
        results.append(expect_error("frame == count", IndexError, lambda: reader.get_frames_at([count])))
        results.append(expect_error("frame < -count", IndexError, lambda: reader.get_frames_at([-count - 1])))
        results.append(expect_error("bool frame", ValueError, lambda: reader.get_frames_at([True])))
        results.append(expect_error("float frame", ValueError, lambda: reader.get_frames_at([1.0])))
        results.append(expect_error("string", ValueError, lambda: reader.get_frames_at("12")))
        results.append(expect_error("zero-dimensional array", ValueError, lambda: reader.get_frames_at(np.array(1))))
        results.append(expect_error("iterate an indexed reader", vidlo.Error, lambda: next(iter(reader))))
        results.append({"check": "no I/O after bad requests", "passed": reader.stats == stats})
    with vidlo.VideoReader(video, device=device, format="nv12") as reader:
        results.append(expect_error("get_frames_at without index", ValueError, lambda: reader.get_frames_at([0])))
    results.append(expect_error("object storage", vidlo.UnsupportedError,
                                lambda: vidlo.VideoReader("oss://bucket/key.mkv", format="nv12", index=index,
                                                          storage_options={"region": "us-southeast-1"})))
    wrong_size = dict(index, file_size=index["file_size"] + 1)
    results.append(expect_error("index for another file size", vidlo.Error, lambda: vidlo.VideoReader(video, format="nv12", index=wrong_size)))

    keys = [f for f in range(count) if index["decode_start"][f] == index["decode_order"][f]]
    frame_at = {order: f for f, order in enumerate(index["decode_order"])}
    gop = 10
    d = index["decode_order"][keys[gop]]
    end = index["decode_order"][keys[gop + 1]]
    # decode_start 指向 GOP 里第二个包，但那个包所属的帧不以它自己为入口：检查时发现
    bad = copy.deepcopy(index)
    victim = frame_at[d + 2]
    bad["decode_start"][victim] = d + 1
    results.append(expect_error("decode_start not at a GOP start", vidlo.Error, lambda: vidlo.validate_video_index(bad)))
    # 把 GOP 里第二个包（P 帧）冒充成入口，后面的帧都从它开始：索引检查发现不了，解码时第一张图不是 IDR
    fake = copy.deepcopy(index)
    for order in range(d + 1, end):
        fake["decode_start"][frame_at[order]] = d + 1
    vidlo.validate_video_index(fake)
    target = frame_at[end - 1]
    with vidlo.VideoReader(video, device=device, format="nv12", index=fake) as reader:
        results.append(expect_error(f"fake GOP start at decode position {d + 1}", vidlo.Error, lambda: reader.get_frames_at([target])))
        results.append(expect_error("reader unusable after a decode error", vidlo.Error, lambda: reader.get_frames_at([0])))
        results.append(expect_error("empty request after a decode error", vidlo.Error, lambda: reader.get_frames_at([])))
    # 把 GOP 里第一个可跳过的帧冒充成入口（只有它自己以它为入口，索引检查发现不了），请求跨过它但不要它：
    # 它不能被跳过，解码时发现它不是 IDR
    g = next(frame_at[k] for k in range(d + 1, end) if ref["unreferenced"][frame_at[k]])
    fake_skippable = copy.deepcopy(index)
    fake_skippable["decode_start"][g] = index["decode_order"][g]
    vidlo.validate_video_index(fake_skippable)
    later = frame_at[index["decode_order"][g] + 1]
    with vidlo.VideoReader(video, device=device, format="nv12", index=fake_skippable) as reader:
        results.append(expect_error(f"skippable frame {g} faked as a GOP start", vidlo.Error, lambda: reader.get_frames_at([keys[gop], later])))
    results.append(fork_check(video, device, index))
    return results


def fork_check(video: Path, device: int, index: dict) -> dict:
    """本进程已经初始化过 CUDA：fork 出的子进程读帧应报错并提示改用 spawn；子进程里的解码器缓存是新的空缓存。"""
    with vidlo.VideoReader(video, device=device, format="nv12", index=index) as reader:
        reader.get_frames_at([0])  # 让父进程的缓存里有一个空闲解码器
        pid = os.fork()
        if pid == 0:
            try:
                cache = vidlo._native.decoder_cache_stats()
                fresh = cache["idle"] == 0 and cache["hits"] == 0 and cache["returned"] == 0
                reader.get_frames_at([0])
                code = 1
            except vidlo.Error as e:
                code = (0 if fresh else 4) if "spawn" in str(e) else 2
            except BaseException:  # noqa: BLE001 - 子进程里任何别的异常都算失败
                code = 3
            os._exit(code)
        _, status = os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    return {"check": "fork after CUDA was initialized", "passed": code == 0, "raised": f"child exit code {code}"}


def run_default(ref: dict, device: int, seed: int, clips: int) -> tuple[list[dict], dict, dict, dict]:
    """默认配置：全部请求。缓存里已有同格式的空闲解码器时第一个请求就复用，之后每个请求都复用。"""
    results = []
    with vidlo.VideoReader(ref["video"], device=device, format="nv12", index=ref["index"]) as reader:
        for label, frames in requests(ref["index"], seed, clips):
            results.append(check_request(reader, ref, label, frames))
        stats, decoder_config, metadata = reader.stats, reader.decoder_config, reader.metadata
    phase = {"phase": f"default {ref['video'].name}", "requests": len(results), "passed": all(r["passed"] for r in results),
             "decoder_creations": stats["decoder_creations"], "decoder_reuses": stats["decoder_reuses"], "skipped": stats["skipped"]}
    if stats["decoder_creations"] > 1:
        phase["passed"] = False
        phase["problem"] = "the cached decoder was not reused"
    return results, phase, stats, {"decoder_config": decoder_config, "metadata": metadata}


def run_alternating(refs: list[dict], device: int, seed: int, rounds: int) -> tuple[list[dict], dict]:
    """几个读取器（不同文件）交替请求：格式相同，缓存里的一个解码器跨文件复用，不新建。"""
    rng = random.Random(seed + 1)
    readers = [vidlo.VideoReader(ref["video"], device=device, format="nv12", index=ref["index"]) for ref in refs]
    results = []
    try:
        for i in range(rounds * len(refs)):
            ref, reader = refs[i % len(refs)], readers[i % len(refs)]
            s = rng.randrange(0, len(ref["hashes"]) - 16)
            results.append(check_request(reader, ref, f"{ref['video'].name} 16 from {s}", list(range(s, s + 16))))
        creations = sum(r.stats["decoder_creations"] for r in readers)
    finally:
        for reader in readers:
            reader.close()
    phase = {"phase": "alternating videos", "requests": len(results), "decoder_creations": creations,
             "passed": all(r["passed"] for r in results) and creations == 0}
    return results, phase


def run_config(ref: dict, device: int, seed: int, clips: int, count: int, capacity: int, skip: bool) -> tuple[list[dict], dict]:
    """缓存与跳帧的其他组合各跑一部分请求，便于分别定位问题；缓存关闭且不跳帧就是第 1 步的做法。"""
    vidlo._native.set_decoder_cache_capacity(capacity)
    results = []
    try:
        with vidlo.VideoReader(ref["video"], device=device, format="nv12", index=ref["index"]) as reader:
            if not skip:  # 测试专用：换成不跳帧的原生读取器
                reader._native = vidlo._native.IndexedDecoder(vidlo._native.FileInput(str(ref["video"])), reader.metadata.codec_config,
                                                              device, reader._spec, skip_unreferenced=False)
            for label, frames in requests(ref["index"], seed, clips)[:count]:
                results.append(check_request(reader, ref, label, frames, skip=skip))
            stats = reader.stats
    finally:
        vidlo._native.set_decoder_cache_capacity(1)
    reuse_ok = stats["decoder_reuses"] == 0 if capacity == 0 else stats["decoder_creations"] <= 1
    phase = {"phase": f"cache {capacity}, {'skipping' if skip else 'no skipping'}, {ref['video'].name}", "requests": len(results),
             "decoder_creations": stats["decoder_creations"], "decoder_reuses": stats["decoder_reuses"], "skipped": stats["skipped"],
             "passed": all(r["passed"] for r in results) and reuse_ok and (skip or stats["skipped"] == 0)}
    return results, phase


def run_api(ref: dict, device: int, output: Path) -> tuple[list[dict], dict]:
    """完整接口：len、下标、切片、get_frame_at、get_frames_in_range、read_sample、缓存容量设置，以及顺序读的读取器上这些
    接口的行为。取到的帧都与顺序读的哈希比较。"""
    video, index, hashes = ref["video"], ref["index"], ref["hashes"]
    count = len(hashes)
    keys = [f for f in range(count) if index["decode_start"][f] == index["decode_order"][f]]
    results = []

    def batch_check(name: str, batch: vidlo.FrameBatch, expected: list[int]) -> None:
        ok = batch.number == expected and len(batch.images) == len(expected) and all(
            sha(batch.images[i]) == hashes[n] for i, n in enumerate(expected))
        results.append({"check": name, "passed": ok, "frames": len(expected)})

    def frame_check(name: str, frame: vidlo.Frame, expected: int) -> None:
        ok = (isinstance(frame, vidlo.Frame) and frame.number == expected and sha(frame.image) == hashes[expected]
              and frame.decode_order == index["decode_order"][expected] and frame.pts is None)
        results.append({"check": name, "passed": ok})

    with vidlo.VideoReader(video, device=device, format="nv12", index=index) as reader:
        results.append({"check": "len and bool", "passed": len(reader) == count and bool(reader)})
        for i in (0, count - 1, keys[3] + 2):
            frame_check(f"reader[{i}]", reader[i], i)
        frame_check("reader[-1]", reader[-1], count - 1)
        frame_check("get_frame_at(-2)", reader.get_frame_at(-2), count - 2)
        a = keys[2]
        batch_check("slice forward", reader[a:a + 10], list(range(a, a + 10)))
        batch_check("slice backward", reader[a + 10:a:-1], list(range(a + 10, a, -1)))
        batch_check("slice tail", reader[-5:], list(range(count - 5, count)))
        batch_check("empty slice", reader[5:5], [])
        batch_check("range every 2nd", reader.get_frames_in_range(keys[4], keys[4] + 161, 2), list(range(keys[4], keys[4] + 161, 2)))
        batch_check("range clipped at the end", reader.get_frames_in_range(count - 3, count + 10), list(range(count - 3, count)))
        batch_check("range with negative bounds", reader.get_frames_in_range(-5, -1), list(range(count - 5, count - 1)))
        results.append(expect_error("range step 0", ValueError, lambda: reader.get_frames_in_range(0, 10, 0)))
        results.append(expect_error("range negative step", ValueError, lambda: reader.get_frames_in_range(10, 0, -1)))
        results.append(expect_error("range non-integer start", ValueError, lambda: reader.get_frames_in_range(0.0, 10)))
        results.append(expect_error("reader[count]", IndexError, lambda: reader[count]))
        results.append(expect_error("reader['a']", ValueError, lambda: reader["a"]))
    with vidlo.VideoReader(video, device=device, format="nv12") as reader:
        results.append(expect_error("len of a sequential reader", TypeError, lambda: len(reader)))
        results.append({"check": "bool of a sequential reader", "passed": bool(reader)})
        results.append(expect_error("reader[0] without index", ValueError, lambda: reader[0]))
        results.append(expect_error("get_frames_in_range without index", ValueError, lambda: reader.get_frames_in_range(0, 2)))

    # read_sample：根目录是录像所在目录，样本里写文件名
    root, start = video.parent, keys[6] + 3
    frames = list(range(start, start + 21))
    sample = {"video": video.name, "frames": frames}
    sample_path, index_path = output / "sample.json", output / f"index-{video.stem}.json"
    with sample_path.open("x", encoding="utf-8") as stream:
        json.dump(sample, stream)
    batch_check("read_sample (files)", vidlo.read_sample(sample_path, index_path, root, device=device, format="nv12"), frames)
    batch_check("read_sample step=2", vidlo.read_sample(sample, index, root, step=2, device=device, format="nv12"), frames[::2])
    em = vidlo._sampler.EMDownSampler(60, 30).select(frames)
    batch_check("read_sample fps=30 source_fps=60", vidlo.read_sample(sample, index, str(root), fps=30, source_fps=60, device=device,
                                                                     format="nv12"), em)
    beyond = dict(sample, frames=[start, count])  # step=2 会丢掉越界的那一项，照样要报错
    results.append(expect_error("read_sample out-of-range frame dropped by step", IndexError,
                                lambda: vidlo.read_sample(beyond, index, root, step=2, format="nv12")))
    for bad in ("../x.mkv", "a//b.mkv", "./b.mkv", "/abs.mkv", "s3://b/k.mkv"):
        results.append(expect_error(f"read_sample video path {bad!r}", vidlo.Error,
                                    lambda b=bad: vidlo.read_sample(dict(sample, video=b), index, root, format="nv12")))
    results.append(expect_error("read_sample from object storage", vidlo.UnsupportedError,
                                lambda: vidlo.read_sample(sample, index, "oss://bucket/prefix", format="nv12",
                                                          storage_options={"region": "us-southeast-1"})))
    results.append(expect_error("cache capacity -1", ValueError, lambda: vidlo.set_decoder_cache_capacity(-1)))
    results.append(expect_error("cache capacity True", ValueError, lambda: vidlo.set_decoder_cache_capacity(True)))
    vidlo.set_decoder_cache_capacity(1)
    phase = {"phase": f"interface {video.name}", "checks": len(results), "passed": all(r["passed"] for r in results)}
    return results, phase


def run_rgb(ref: dict, device: int, seed: int, clips: int, count: int) -> tuple[list[dict], dict] | None:
    """RGB 输出：同样的裁剪缩放先顺序读一遍作参照，再按帧号读一部分请求逐帧比较。limited range 的录像不能转 RGB，跳过。"""
    with vidlo.VideoReader(ref["video"], device=device, format="nv12", index=ref["index"]) as probe:
        metadata = probe.metadata  # 按帧号读的读取器打开时不解码
    if not metadata.full_range:
        return None
    width, height = metadata.width - 32, metadata.height
    options = {"crop": (16, 0, width, height), "width": width * 2 // 3 // 2 * 2, "height": height * 2 // 3 // 2 * 2}
    reference = []
    with vidlo.VideoReader(ref["video"], device=device, **options) as reader:
        for frame in reader:
            reference.append((sha(frame.image), frame.colorspace, frame.interpolation))
    results = []
    with vidlo.VideoReader(ref["video"], device=device, index=ref["index"], **options) as reader:
        for label, frames in requests(ref["index"], seed, clips)[:count]:
            numbers = [f % len(reference) for f in frames]
            batch = reader.get_frames_at(frames)
            problems = []
            if batch.images.shape != (len(frames), options["height"], options["width"], 3):
                problems.append(f"images shape {batch.images.shape}")
            wrong = [n for i, n in enumerate(numbers) if i < len(batch.images) and sha(batch.images[i]) != reference[n][0]]
            if wrong:
                problems.append(f"{len(wrong)} frames differ from the sequential RGB read, first {wrong[:5]}")
            if list(zip(batch.colorspace, batch.interpolation)) != [reference[n][1:] for n in numbers]:
                problems.append("colorspace or interpolation differ from the sequential read")
            results.append({"request": f"RGB {label}", "frames": len(frames), "passed": not problems, "problems": problems})
    phase = {"phase": f"RGB {options['crop']} -> {options['width']}x{options['height']}, {ref['video'].name}", "requests": len(results),
             "passed": all(r["passed"] for r in results)}
    return results, phase


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path, action="append", help="可给多段；两段以上时另测跨文件复用解码器")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--clips", type=int, default=25, help="16 帧连续与 81 帧隔一帧的随机片段各几个")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    refs = []
    for video in args.video:
        started = time.perf_counter()
        index, hashes = sequential_reference(video, args.device)
        unreferenced, layers = unreferenced_frames(video, index)
        refs.append({"video": video, "index": index, "hashes": hashes, "unreferenced": unreferenced, "non_reference_by_layer": layers,
                     "sequential_s": round(time.perf_counter() - started, 3)})
        with (args.output / f"index-{video.stem}.json").open("x", encoding="utf-8") as stream:
            json.dump(index, stream)

    results, phases, details = [], [], {}
    for ref in refs:
        found, phase, stats, extra = run_default(ref, args.device, args.seed, args.clips)
        results += found
        phases.append(phase)
        details[ref["video"].name] = {"stats": stats, "decoder_config": extra["decoder_config"],
                                      "metadata": {k: (base64.b64encode(v).decode() if isinstance(v, bytes) else v)
                                                   for k, v in vars(extra["metadata"]).items()}}
    if len(refs) > 1:
        found, phase = run_alternating(refs, args.device, args.seed, rounds=10)
        results += found
        phases.append(phase)
    for capacity, skip in ((1, False), (0, True), (0, False)):
        found, phase = run_config(refs[0], args.device, args.seed, args.clips, 20, capacity, skip)
        results += found
        phases.append(phase)
    rgb = run_rgb(refs[0], args.device, args.seed, args.clips, 30)
    if rgb is not None:
        results += rgb[0]
        phases.append(rgb[1])
    api, phase = run_api(refs[0], args.device, args.output)
    phases.append(phase)
    errors = error_checks(refs[0], args.device)

    failed = ([r for r in results if not r["passed"]] + [p for p in phases if not p["passed"]] + [r for r in errors if not r["passed"]]
              + [r for r in api if not r["passed"]])
    report = {
        "videos": [{"video": str(r["video"]), "file_size": r["index"]["file_size"], "frames": len(r["hashes"]),
                    "unreferenced_frames": sum(r["unreferenced"]), "non_reference_by_layer": r["non_reference_by_layer"],
                    "sequential_reference_s": r["sequential_s"]} for r in refs],
        "vidlo_version": vidlo.__version__,
        "requests": len(results),
        "requests_passed": sum(r["passed"] for r in results),
        "phases": phases,
        "error_checks": len(errors),
        "error_checks_passed": sum(r["passed"] for r in errors),
        "interface_checks": len(api),
        "interface_checks_passed": sum(r["passed"] for r in api),
        "decoder_cache": vidlo._native.decoder_cache_stats(),
        "passed": not failed,
        "failed": failed,
        "details": details,
        "results": results,
        "errors": errors,
        "interface": api,
    }
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("requests", "requests_passed", "error_checks", "error_checks_passed", "interface_checks",
                                             "interface_checks_passed", "passed")}, ensure_ascii=False))
    for p in phases:
        print(json.dumps(p, ensure_ascii=False))
    for r in failed[:10]:
        print("FAIL", json.dumps(r, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
