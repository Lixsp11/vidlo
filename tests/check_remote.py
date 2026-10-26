"""从 R2（s3://）或 OSS（oss://）顺序读一段录像，与本地同一文件逐帧比较。

两种输出各读一遍：原生 YUV 逐帧 SHA256 与写索引用的逐帧信息必须与本地完全相同；训练用的裁剪缩放 RGB 也逐帧比较哈希。
记下请求数、重试次数、读取量和用时。凭证只经环境变量传给库：S3 兼容存储可用 --rclone-config 从 rclone 配置的
[R2] 段读出后放进本进程的环境变量（不打印）；OSS 用调用方环境里的 OSS_ACCESS_KEY_ID、OSS_ACCESS_KEY_SECRET。
输出目录必须不存在，结果写进 report.json。
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import vidlo

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--local", required=True, type=Path)
    parser.add_argument("--storage-options", default="{}",
                        help="JSON，如 {\"region\": \"us-southeast-1\", \"use_internal_endpoint\": true}")
    parser.add_argument("--rclone-config", type=Path, help="s3:// 时从这份 rclone 配置的 --rclone-remote 段取密钥和 endpoint")
    parser.add_argument("--rclone-remote", default="R2")
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
    report["passed"] = all(r["same_pictures"] and r["same_index_info"] and r["frames"] > 0 and r["remote_stats"]["bytes_read"] == size
                           and bool(r["remote_etag"]) and r["local_etag_is_none"]
                           for r in runs.values())
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    for name, r in runs.items():
        s = r["remote_stats"]
        print(f"{name}: frames={r['frames']} same_pictures={r['same_pictures']} same_index_info={r['same_index_info']} "
              f"remote {r['remote_elapsed_s']} s ({r['remote_mib_per_s']} MiB/s), local {r['local_elapsed_s']} s, "
              f"requests={s['requests']} retries={s['retries']}")
    print("PASSED" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
