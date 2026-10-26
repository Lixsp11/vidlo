"""顺序读给出的逐帧索引信息与 ffprobe 独立核对。

对一段录像：用 vidlo 顺序读（原生 YUV，不比较像素），按第 3.1 节拼出视频索引并用 vidlo 自己的检查读回；再用 ffprobe 按解码
顺序列出每个包的 pts、pos、size、关键帧标记和负载的 SHA256，逐包核对：
- 文件中 [packet_start, packet_end) 的字节的 SHA256 等于 ffprobe 给出的负载哈希，长度等于 size；
- 第 k 个包对应 decode_order 为 k 的帧，pts 和关键帧标记一致；
- decode_start 等于第 k 个包之前（含）最近一个关键帧的位置。

输出目录必须不存在，写入 report.json 和拼出的 video.index.json。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import vidlo


def ffprobe_packets(video: Path) -> list[dict]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_data_hash", "sha256",
                          "-show_entries", "packet=pts,pos,size,flags,data_hash", "-of", "json", str(video)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["packets"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    infos = []
    error = None
    with vidlo.VideoReader(args.video, device=args.device, format="nv12") as video:
        info = video.metadata
        try:
            for frame in video:
                infos.append({k: v for k, v in vars(frame).items() if k != "image"})
        except vidlo.Error as e:
            error = f"{type(e).__name__}: {e}"
        stats = video.stats
    elapsed = time.perf_counter() - started

    index = {
        "version": 1,
        "file_size": info.file_size,
        "codec": info.codec,
        "codec_config": base64.b64encode(info.codec_config).decode(),
        "decode_order": [f["decode_order"] for f in infos],
        "decode_start": [f["decode_start"] for f in infos],
        "packet_start": [f["packet_start"] for f in infos],
        "packet_end": [f["packet_end"] for f in infos],
    }
    index_path = args.output / "video.index.json"
    with index_path.open("x", encoding="utf-8") as stream:
        json.dump(index, stream)
    validate_error = None
    try:
        vidlo.load_video_index(index_path)
    except vidlo.Error as e:
        validate_error = f"{type(e).__name__}: {e}"

    packets = ffprobe_packets(args.video)
    by_order = {f["decode_order"]: f for f in infos}
    problems = []
    latest_key = -1
    with args.video.open("rb") as stream:
        for k, packet in enumerate(packets):
            key = "K" in packet["flags"]
            if key:
                latest_key = k
            frame = by_order.get(k)
            if frame is None:
                problems.append({"decode_position": k, "problem": "no frame with this decode_order"})
                continue
            stream.seek(frame["packet_start"])
            data = stream.read(frame["packet_end"] - frame["packet_start"])
            expected_hash = packet["data_hash"].split(":", 1)[1].lower()
            checks = {
                "pts": frame["pts"] == int(packet["pts"]),
                "size": frame["packet_end"] - frame["packet_start"] == int(packet["size"]),
                "data_hash": hashlib.sha256(data).hexdigest() == expected_hash,
                "key_frame": frame["key_frame"] == key,
                "decode_start": frame["decode_start"] == latest_key,
            }
            if not all(checks.values()):
                problems.append({"decode_position": k, "failed": [name for name, ok in checks.items() if not ok],
                                 "frame": frame, "ffprobe": packet})
    # 块头长度：b1 录像的轨道号 VINT 是 1 字节，所以负载起点 = ffprobe 的 pos + 4
    header_lengths = sorted({by_order[k]["packet_start"] - int(p["pos"]) for k, p in enumerate(packets) if k in by_order})

    report = {
        "video": str(args.video),
        "vidlo_version": vidlo.__version__,
        "error": error,
        "index_validation_error": validate_error,
        "frames": len(infos),
        "ffprobe_packets": len(packets),
        "keyframes": sum(1 for f in infos if f["key_frame"]),
        "block_header_lengths": header_lengths,
        "problems": len(problems),
        "problems_first_20": problems[:20],
        "stats": stats,
        "elapsed_s": round(elapsed, 3),
    }
    report["passed"] = error is None and validate_error is None and not problems and len(infos) == len(packets)
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("video", "passed", "error", "index_validation_error", "frames", "ffprobe_packets",
                                             "keyframes", "block_header_lengths", "problems")}, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in ("bytes_read", "seeks", "first_seeks")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
