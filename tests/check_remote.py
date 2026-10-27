"""从 R2（s3://）或 OSS（oss://）读一段录像，与本地同一文件逐帧比较。

顺序读：两种输出各读一遍，原生 YUV 逐帧 SHA256 与写索引用的逐帧信息必须与本地完全相同；训练用的裁剪缩放 RGB 也逐帧比较哈希。
按帧号读：用本地顺序读建的视频索引，固定种子的单帧、16 帧连续、81 帧隔一帧请求各若干个，原生 YUV 与本地顺序读逐帧比较；
没有故障时每个请求正好一个 HTTP 请求，读取量等于按索引算出的各区间之和。
记下请求数、重试次数、读取量和用时（各请求的用时只供参考，测速另测）。凭证只经环境变量传给库：S3 兼容存储可用
--rclone-config 从 rclone 配置的 [R2] 段读出后放进本进程的环境变量（不打印）；OSS 用调用方环境里的 OSS_ACCESS_KEY_ID、
OSS_ACCESS_KEY_SECRET。输出目录必须不存在，结果写进 report.json。
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import vidlo
from check_random import sequential_reference, sha

CROP = (32, 0, 1248, 720)
SIZE = (832, 480)
INDEX_FIELDS = ("number", "pts", "key_frame", "decode_order", "decode_start", "packet_start", "packet_end")


def credentials_from_rclone(path: Path, remote: str) -> str:
    """把 rclone 配置里一个 S3 兼容远端的密钥放进环境变量，返回它的 endpoint。"""
    conf = configparser.ConfigParser()
    conf.read(path, encoding="utf-8")
    section = conf[remote]
    if section.get("type") != "s3":
        raise SystemExit(f"rclone remote {remote} is not an S3 remote")
    os.environ["AWS_ACCESS_KEY_ID"] = section["access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = section["secret_access_key"]
    return section["endpoint"]


def read(settings: dict[str, Any], source: str, storage_options: dict[str, Any] | None) -> dict[str, Any]:
    started = time.perf_counter()
    digests, infos = [], []
    with vidlo.VideoReader(source, storage_options=storage_options, **settings) as video:
        for frame in video:
            digests.append(hashlib.sha256(frame.image.tobytes()).hexdigest())
            infos.append({k: getattr(frame, k) for k in INDEX_FIELDS})
        stats, etag = video.stats, video.metadata.etag
    return {"digests": digests, "infos": infos, "stats": stats, "etag": etag, "elapsed_s": round(time.perf_counter() - started, 3)}


def random_reads(source: str, local: Path, storage_options: dict[str, Any], count: int) -> dict[str, Any]:
    """按帧号读：远程与本地顺序读逐帧比较，核对 HTTP 请求数与读取量。"""
    index, hashes = sequential_reference(local, 0)
    frame_at = {order: f for f, order in enumerate(index["decode_order"])}
    total, rng = len(hashes), random.Random(0)
    shapes = {"1 frame": lambda s: [s], "16 consecutive": lambda s: list(range(s, s + 16)),
              "81 every 2nd": lambda s: list(range(s, s + 161, 2))}
    requests = [(name, make(rng.randrange(total - 161))) for name, make in shapes.items() for _ in range(count)]
    expected_bytes = 0
    for _, frames in requests:
        s = min(index["decode_start"][f] for f in frames)
        e = max(index["decode_order"][f] for f in frames)
        expected_bytes += index["packet_end"][frame_at[e]] - index["packet_start"][frame_at[s]]
    times: dict[str, dict[str, list[float]]] = {}
    wrong = []
    for where, src, options in (("local", str(local), None), ("remote", source, storage_options)):
        with vidlo.VideoReader(src, format="nv12", index=index, storage_options=options) as reader:
            for name, frames in requests:
                started = time.perf_counter()
                batch = reader.get_frames_at(frames)
                times.setdefault(name, {}).setdefault(where, []).append((time.perf_counter() - started) * 1000)
                if [sha(image) for image in batch.images] != [hashes[f] for f in frames]:
                    wrong.append({"where": where, "request": name, "first": frames[0]})
            stats = reader.stats
    passed = (not wrong and stats["requests"] == len(requests) and stats["retries"] == 0 and stats["reads"] == len(requests)
              and stats["bytes_read"] == expected_bytes)
    return {"requests": len(requests), "passed": passed, "wrong": wrong[:10], "http_requests": stats["requests"],
            "retries": stats["retries"], "bytes_read": stats["bytes_read"], "expected_bytes": expected_bytes,
            "median_ms": {name: {where: round(statistics.median(v), 1) for where, v in by.items()} for name, by in times.items()},
            "remote_stats": stats}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--local", required=True, type=Path)
    parser.add_argument("--storage-options", default="{}",
                        help="JSON，如 {\"region\": \"us-southeast-1\", \"use_internal_endpoint\": true}")
    parser.add_argument("--rclone-config", type=Path, help="s3:// 时从这份 rclone 配置的 --rclone-remote 段取密钥和 endpoint")
    parser.add_argument("--rclone-remote", default="R2")
    parser.add_argument("--random-requests", type=int, default=10, help="按帧号读时每种请求的个数，0 不测")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    storage = json.loads(args.storage_options)
    if args.rclone_config is not None:
        storage["endpoint_url"] = credentials_from_rclone(args.rclone_config, args.rclone_remote)

    runs: dict[str, Any] = {}
    for name, settings in (("yuv", {"format": "nv12"}), ("rgb", {"crop": CROP, "width": SIZE[0], "height": SIZE[1]})):
        local = read(settings, str(args.local), None)
        remote = read(settings, args.source, storage)
        runs[name] = {
            "frames": len(remote["digests"]), "same_pictures": remote["digests"] == local["digests"],
            "same_index_info": remote["infos"] == local["infos"],
            "remote_etag": remote["etag"], "local_etag_is_none": local["etag"] is None,
            "local_elapsed_s": local["elapsed_s"], "remote_elapsed_s": remote["elapsed_s"],
            "remote_mib_per_s": round(remote["stats"]["bytes_read"] / 2**20 / remote["elapsed_s"], 1),
            "remote_stats": remote["stats"],
        }
    size = args.local.stat().st_size
    report = {
        "source": args.source, "local": str(args.local), "file_size": size,
        "storage_options": {k: v for k, v in storage.items() if k not in ("endpoint", "endpoint_url")},
        "endpoint_host_suffix": next((v.split("//", 1)[-1].split(".", 1)[-1] for k, v in storage.items()
                                      if k in ("endpoint", "endpoint_url")), None),
        "vidlo_version": vidlo.__version__, "runs": runs,
    }
    report["random"] = random_reads(args.source, args.local, storage, args.random_requests) if args.random_requests else None
    report["passed"] = all(r["same_pictures"] and r["same_index_info"] and r["frames"] > 0 and r["remote_stats"]["bytes_read"] == size
                           and bool(r["remote_etag"]) and r["local_etag_is_none"]
                           for r in runs.values()) and (report["random"] is None or report["random"]["passed"])
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    for name, r in runs.items():
        s = r["remote_stats"]
        print(f"{name}: frames={r['frames']} same_pictures={r['same_pictures']} same_index_info={r['same_index_info']} "
              f"remote {r['remote_elapsed_s']} s ({r['remote_mib_per_s']} MiB/s), local {r['local_elapsed_s']} s, "
              f"requests={s['requests']} retries={s['retries']}")
    if report["random"] is not None:
        r = report["random"]
        print(f"random: requests={r['requests']} passed={r['passed']} http_requests={r['http_requests']} retries={r['retries']} "
              f"bytes={r['bytes_read']} (expected {r['expected_bytes']}) median_ms={json.dumps(r['median_ms'])}")
    print("PASSED" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
