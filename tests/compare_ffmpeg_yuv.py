"""顺序读原生 YUV 与 FFmpeg CPU 解码逐帧逐位比较，并核对帧号与 PTS 的对应。

vidlo 交出显示区域的 NV12；FFmpeg 按录像自己的像素格式（yuvj420p 或 yuv420p，都是 I420）输出，不做任何转换。比较前只把
NV12 的交错色度拆成 U、V 两个平面；不让 FFmpeg 输出 nv12，因为 yuvj420p 到 nv12 在 swscale 里会做范围转换。显示顺序的时间戳
另与 ffprobe 的包时间戳排序后比较。

输出目录必须不存在，结果写进其中的 report.json。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import vidlo


def read_exact(stream, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def nv12_to_i420(nv12: np.ndarray) -> bytes:
    height = nv12.shape[0] * 2 // 3
    luma, chroma = nv12[:height], nv12[height:]
    return luma.tobytes() + chroma[:, 0::2].tobytes() + chroma[:, 1::2].tobytes()


def ffprobe_pix_fmt(video: Path) -> str:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(video)],
                         check=True, capture_output=True, text=True).stdout.strip()
    # 只比较 8 位 4:2:0；其他格式 vidlo 本来就应报错，不在这里比较
    if out not in ("yuvj420p", "yuv420p"):
        raise SystemExit(f"unsupported reference pixel format {out!r}")
    return out


def ffprobe_display_pts(video: Path) -> list[int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts", "-of", "json", str(video)],
                         check=True, capture_output=True, text=True).stdout
    return sorted(int(p["pts"]) for p in json.loads(out)["packets"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-mismatch", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    ffmpeg_cmd = ["ffmpeg", "-v", "warning", "-nostdin", "-i", str(args.video), "-map", "0:v:0", "-fps_mode", "passthrough",
                  "-f", "rawvideo", "-pix_fmt", ffprobe_pix_fmt(args.video), "-"]
    ffmpeg_version = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    ffmpeg_err = (args.output / "ffmpeg.stderr.log").open("wb")
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=ffmpeg_err, bufsize=0)

    started = time.perf_counter()
    compared, mismatches, saved = 0, [], 0
    pts_list: list[int] = []
    frame_bytes = None
    error = None
    hasher = hashlib.sha256()
    with vidlo.VideoReader(args.video, device=args.device, format="nv12") as video:
        info = asdict(video.metadata)
        try:
            for frame in video:
                ours = nv12_to_i420(frame.image)
                frame_bytes = len(ours)
                theirs = read_exact(ffmpeg.stdout, frame_bytes)
                pts_list.append(frame.pts)
                if frame.number != compared:
                    raise RuntimeError(f"frame number {frame.number} at position {compared}")
                hasher.update(ours)
                if len(theirs) != frame_bytes:
                    mismatches.append({"number": compared, "ffmpeg_bytes": len(theirs)})
                    break
                if ours != theirs:
                    a = np.frombuffer(ours, np.uint8).astype(np.int16)
                    b = np.frombuffer(theirs, np.uint8).astype(np.int16)
                    mismatches.append({"number": compared, "differing_bytes": int((a != b).sum()), "max_abs": int(np.abs(a - b).max())})
                    if saved < args.save_mismatch:
                        (args.output / f"mismatch-{compared}-vidlo.yuv").write_bytes(ours)
                        (args.output / f"mismatch-{compared}-ffmpeg.yuv").write_bytes(theirs)
                        saved += 1
                compared += 1
        except vidlo.Error as e:
            error = f"{type(e).__name__}: {e}"
        decoder_config = video.decoder_config
        stats = video.stats
    elapsed = time.perf_counter() - started

    # FFmpeg 若还有剩余帧，计数后丢弃
    extra_ffmpeg_frames = 0
    if frame_bytes:
        while len(read_exact(ffmpeg.stdout, frame_bytes)) == frame_bytes:
            extra_ffmpeg_frames += 1
    ffmpeg.stdout.close()
    ffmpeg_rc = ffmpeg.wait()
    ffmpeg_err.close()

    probe_pts = ffprobe_display_pts(args.video)
    info["codec_config"] = base64.b64encode(info["codec_config"]).decode()
    report = {
        "video": str(args.video),
        "file_size": args.video.stat().st_size,
        "vidlo_version": getattr(vidlo, "__version__", None),
        "ffmpeg_cmd": ffmpeg_cmd,
        "ffmpeg_version": ffmpeg_version,
        "ffmpeg_exit": ffmpeg_rc,
        "error": error,
        "frames_vidlo": compared,
        "frames_ffmpeg_extra": extra_ffmpeg_frames,
        "mismatching_frames": len(mismatches),
        "mismatches_first_50": mismatches[:50],
        "pts_match_ffprobe": pts_list == probe_pts,
        "ffprobe_packets": len(probe_pts),
        "vidlo_i420_sha256": hasher.hexdigest(),
        "elapsed_s": round(elapsed, 3),
        "info": info,
        "stats": stats,
        "decoder_config": decoder_config,
        "ffmpeg_stderr_tail": (args.output / "ffmpeg.stderr.log").read_text(encoding="utf-8", errors="replace")[-2000:],
    }
    passed = (error is None and ffmpeg_rc == 0 and not mismatches and extra_ffmpeg_frames == 0 and compared == len(probe_pts)
              and report["pts_match_ffprobe"])
    report["passed"] = passed
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("video", "passed", "error", "frames_vidlo", "frames_ffmpeg_extra", "mismatching_frames",
                                             "pts_match_ffprobe", "elapsed_s")}, ensure_ascii=False))
    print(json.dumps(report["stats"], ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
