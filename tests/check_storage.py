"""对象存储读取的故障注入，不访问外网。一、二是顺序读；按帧号读的有限区间见 range_checks、FORK_RANGE_CODE
和 real_http_checks 里的 range 项。

一、假客户端：代替 boto3，从本地文件按 Range 响应，按计划注入故障；走真实的 Download 线程、StreamInput 队列和 C++
解封装解码。检查：正常读与本地逐帧相同；中途断开、提前结束、503、408、TLS 连接被中途关闭后续读成功且续读带 If-Match，
第一次请求失败也会重试，连续失败计数从第一次请求一直算到读响应体；连续失败、412、Content-Range 不对、ETag 或总长
变了、第一次响应 200、404、403、证书错误（含包在 requests、urllib3 里的）都明确报错且不多发请求；字节齐全之后响应
才报错、多出字节时报错，不当作读完；截断的对象与本地一样交出完整的帧后报 TruncatedError；下载线程确实停在阻塞的
读取上时，close 或不 close 直接丢掉都会打断读取、关闭响应、停下线程；close 时恰好有请求在进行，迟到的响应也被打断
并关闭；文件头的 SeekHead 指向末尾 Tags 的大文件能照常顺序读。

二、本机 HTTP 服务：boto3 与 OSS SDK 走真实的 urllib3、requests，见 real_http_checks。
输出目录必须不存在，结果写进 report.json。
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import vidlo
from vidlo import _storage
from check_random import sequential_reference, sha
from check_synthetic import expect_raises, make_truncated, make_video

ETAG = '"fake-etag"'


class FakeClient:
    """按 plan 依次决定每次请求的响应；files 把路径映射到本地文件。responses 记下每个响应是否被打断、关闭。"""

    files: dict[str, Path] = {}
    plan: list[dict[str, Any]] = []
    seen: list[dict[str, Any]] = []
    responses: list[dict[str, threading.Event]] = []
    blocking = threading.Event()  # 有响应停在阻塞的读取上
    in_get = threading.Event()    # 有请求停在 get() 里
    release = threading.Event()   # 放行停在 get() 里的请求

    def __init__(self, bucket: str, key: str, storage: dict[str, Any] | None, shared: bool = False) -> None:
        self.data = FakeClient.files[key].read_bytes()
        self.pid = os.getpid()  # 在哪个进程里建的

    def get(self, start: int, etag: str | None, end: int | None = None) -> _storage.Response:
        """plan 里每项可选：error（状态码，抛 HttpError）、raise（抛这个异常，不带响应）、status、content_range、etag、
        stop_after（交出这么多字节后断开；quiet 时正常结束）、after_all（交完全部字节后抛出的异常）、extra（末尾多交的字节）、
        block_after（交出这么多字节后阻塞，直到响应被打断）、wait（在 get() 里等 release 之后才返回响应）、
        ignore_end（不理会请求的终点，读到对象末尾）、slow（每块之前等这么多秒）。end 是有限区间的终点（不含）。"""
        action = FakeClient.plan.pop(0) if FakeClient.plan else {}
        FakeClient.seen.append({"start": start, "if_match": etag} if end is None else {"start": start, "end": end, "if_match": etag})
        if "error" in action:
            raise _storage.HttpError(action["error"], "Injected", "injected error")
        if "raise" in action:
            raise action["raise"]
        if action.get("wait"):
            FakeClient.in_get.set()
            FakeClient.release.wait(20)
        data = self.data if end is None or action.get("ignore_end") else self.data[:end]
        stop_after = action.get("stop_after", action.get("block_after"))
        state = {"interrupted": threading.Event(), "closed": threading.Event()}
        FakeClient.responses.append(state)

        def chunks():
            position, sent = start, 0
            while position < len(data):
                if "slow" in action:
                    time.sleep(action["slow"])
                n = min(_storage.CHUNK, len(data) - position)
                if stop_after is not None and sent + n > stop_after:
                    if stop_after > sent:
                        yield data[position:position + stop_after - sent]
                    if "block_after" in action:
                        # 模拟阻塞在网络读取上：只有打断（关闭 socket 读端）才让读取出错返回，最多等 20 秒
                        FakeClient.blocking.set()
                        state["interrupted"].wait(20)
                        raise ConnectionResetError("read interrupted")
                    if action.get("quiet"):
                        return  # 连接正常关闭但字节不够
                    raise ConnectionResetError("injected connection reset")
                yield data[position:position + n]
                position += n
                sent += n
            if "extra" in action:
                yield action["extra"]
            if "after_all" in action:
                raise action["after_all"]

        def record(event: str) -> Callable[[], None]:
            def call() -> None:
                state[event].set()
                state[event + "_by"].append(threading.current_thread().name)  # type: ignore[attr-defined]
            return call

        state["closed_by"], state["interrupted_by"] = [], []  # type: ignore[assignment]
        content_range = action.get("content_range", f"bytes {start}-{len(data) - 1}/{len(self.data)}")
        return _storage.Response(action.get("status", 206), content_range, action.get("etag", ETAG), chunks(),
                                 record("closed"), record("interrupted"))


def ebml_children(data: bytes, start: int, end: int) -> list[tuple[bytes, int, int]]:
    """[start, end) 里的 EBML 元素：(ID 字节, 元素起点, 元素终点)。只处理长度已知的元素。"""
    out = []
    pos = start
    while pos < end:
        id_length = 8 - data[pos].bit_length() + 1
        element_id = data[pos:pos + id_length]
        first = data[pos + id_length]
        size_length = 8 - first.bit_length() + 1
        size = first & (0xFF >> size_length)
        for b in data[pos + id_length + 1:pos + id_length + size_length]:
            size = (size << 8) | b
        content = pos + id_length + size_length
        out.append((element_id, pos, content + size))
        pos = content + size
    return out


def ebml_content(data: bytes, start: int) -> int:
    """从 start 起的 EBML 元素的内容起点（跳过 ID 和长度）。"""
    id_length = 8 - data[start].bit_length() + 1
    return start + id_length + (8 - data[start + id_length].bit_length() + 1)


def ebml_element(element_id: bytes, content: bytes) -> bytes:
    return element_id + b"\x01" + len(content).to_bytes(7, "big") + content  # 长度统一写 8 字节


def make_tags_at_end(source: Path, target: Path) -> None:
    """重排 FFmpeg 写的 MKV：Segment 里依次为新的 SeekHead（指向 Info、Tracks 和挪到最后的 Tags）、Info、Tracks、
    全部 Cluster、Tags；去掉原来的 SeekHead、Void 和 Cues（Cues 里的 Cluster 位置会因重排失效，它本身可以没有）。
    Cluster 内部没有绝对位置，原样复制。可跳转的读法会按 SeekHead 跳到文件末尾读 Tags。"""
    data = source.read_bytes()
    (ebml_id, _, ebml_end), (segment_id, segment_start, segment_end) = ebml_children(data, 0, len(data))[:2]
    if segment_id != bytes.fromhex("18538067"):
        raise RuntimeError("unexpected top-level layout")
    first = data[segment_start + 4]
    content_start = segment_start + 4 + (8 - first.bit_length() + 1)
    children = ebml_children(data, content_start, segment_end)
    pick = {name: [data[s:e] for i, s, e in children if i == bytes.fromhex(name)] for name in ("1549A966", "1654AE6B", "1254C367", "1F43B675")}
    info, tracks, tags, clusters = pick["1549A966"][0], pick["1654AE6B"][0], pick["1254C367"][0], pick["1F43B675"]
    entries = 3
    seek_entry_size = 2 + 8 + (2 + 8 + 4) + (2 + 8 + 8)
    seekhead_size = 4 + 8 + entries * seek_entry_size
    positions = {"1549A966": seekhead_size, "1654AE6B": seekhead_size + len(info),
                 "1254C367": seekhead_size + len(info) + len(tracks) + sum(len(c) for c in clusters)}
    seekhead = ebml_element(bytes.fromhex("114D9B74"), b"".join(
        ebml_element(bytes.fromhex("4DBB"), ebml_element(bytes.fromhex("53AB"), bytes.fromhex(name))
                     + ebml_element(bytes.fromhex("53AC"), pos.to_bytes(8, "big")))
        for name, pos in positions.items()))
    assert len(seekhead) == seekhead_size
    body = seekhead + info + tracks + b"".join(clusters) + tags
    target.write_bytes(data[:ebml_end] + ebml_element(segment_id, body))


def seekhead_points_to_tags_at_end(path: Path) -> bool:
    """按元素结构核对：Segment 的第一个子元素是 SeekHead，最后一个是 Tags，前面全是 Cluster 等；SeekHead 里每一项的
    位置都正好是那个元素的起点，其中 Tags 那项指向最后一个元素，且超出 AVIO 的 256 KiB 缓冲。"""
    data = path.read_bytes()
    _, (_, segment_start, segment_end) = ebml_children(data, 0, len(data))[:2]
    content_start = ebml_content(data, segment_start)
    children = ebml_children(data, content_start, segment_end)
    if children[0][0] != bytes.fromhex("114D9B74") or children[-1][0] != bytes.fromhex("1254C367"):
        return False
    ids_at = {start - content_start: element_id for element_id, start, _ in children}  # 相对 Segment 内容起点的位置 → ID
    seekhead = children[0]
    targets = []
    for _, seek_start, seek_end in ebml_children(data, ebml_content(data, seekhead[1]), seekhead[2]):
        fields = {i: data[ebml_content(data, s):e] for i, s, e in ebml_children(data, ebml_content(data, seek_start), seek_end)}
        position = int.from_bytes(fields[bytes.fromhex("53AC")], "big")
        if ids_at.get(position) != fields[bytes.fromhex("53AB")]:
            return False
        targets.append((fields[bytes.fromhex("53AB")], position))
    tags_start = children[-1][1]
    # 必须有一项 SeekHead 指向最后那个 Tags，且它在 AVIO 的 256 KiB 缓冲之外
    return tags_start > 256 * 1024 and (bytes.fromhex("1254C367"), tags_start - content_start) in targets


def hashes(source: str | Path, limit: int | None = None, storage_options: dict[str, Any] | None = None) -> tuple[list[str], str | None, dict[str, Any]]:
    """顺序读原生 YUV，返回每帧的 SHA256、截断错误（没有为 None）和统计。"""
    result, truncated = [], None
    with vidlo.VideoReader(source, format="nv12", storage_options=storage_options) as video:
        try:
            for frame in video:
                result.append(hashlib.sha256(frame.image.tobytes()).hexdigest())
                if limit is not None and len(result) == limit:
                    break
        except vidlo.TruncatedError as e:
            truncated = str(e)
        stats = video.stats
    return result, truncated, stats


def scenario(plan: list[dict[str, Any]], check: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    FakeClient.plan = [dict(p) for p in plan]
    FakeClient.seen = []
    FakeClient.responses = []
    for event in (FakeClient.blocking, FakeClient.in_get, FakeClient.release):
        event.clear()
    result = check()
    result["requests_seen"] = list(FakeClient.seen)
    return result


def download_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "vidlo-download" and t.is_alive())


def wait_threads(limit: float = 5) -> int:
    deadline = time.perf_counter() + limit
    while download_threads() and time.perf_counter() < deadline:
        time.sleep(0.05)
    return download_threads()


# 下载线程停在“队列满”的等待上时 fork：子进程删掉读取器并回收（gc）或直接走解释器退出流程（exit），都不能卡住；
# 父进程照常读完。子进程继承的条件变量上记着父进程那个下载线程，按 glibc 的实现销毁它会一直等，所以库在子进程里不销毁它
FORK_BLOCKED_CODE = """
import gc, os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import check_storage as cs
import vidlo
from vidlo import _storage
_storage.S3Client = cs.FakeClient
_storage.CHUNK = 4096
_storage.CAPACITY = 1 << 16
cs.FakeClient.files = {"large.mkv": Path(sys.argv[2])}
mode = sys.argv[3]
reader = vidlo.VideoReader("s3://fake/large.mkv", format="nv12")
frames = iter(reader)
next(frames)
pushed = -1
while reader._download.input.pushed != pushed:
    pushed = reader._download.input.pushed
    time.sleep(0.3)
pid = os.fork()
if pid == 0:
    if mode == "gc":
        del frames, reader
        gc.collect()
        os._exit(0)
    sys.exit(0)
deadline = time.time() + 20
status = "hung"
while time.time() < deadline:
    done, code = os.waitpid(pid, os.WNOHANG)
    if done:
        status = str(code)
        break
    time.sleep(0.1)
if status == "hung":
    os.kill(pid, 9)
    os.waitpid(pid, 0)
print(status, sum(1 for _ in frames) + 1, pushed)
reader.close()
"""


# 按帧号读的读取器在父进程里打开、取过一次字节（不用 CUDA）后 fork，fork 时另一个线程正拿着共享客户端的锁：
# 子进程里继承来的读取器换用子进程自己建的客户端，照旧带父进程绑定的 ETag；再取共享客户端不能卡在继承来的锁上
FORK_RANGE_CODE = """
import json, os, sys, threading, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import check_storage as cs
import vidlo
from vidlo import _storage
_storage.S3Client = cs.FakeClient
cs.FakeClient.files = {"clip.mkv": Path(sys.argv[2])}
index = json.loads(Path(sys.argv[3]).read_text())
start, end = int(sys.argv[4]), int(sys.argv[5])
reader = vidlo.VideoReader("s3://fake/clip.mkv", format="nv12", index=index)
reader._range.fetch(start, end)
held = threading.Event()
def hold():
    with _storage._SHARED_LOCK:
        held.set()
        time.sleep(5)
threading.Thread(target=hold, daemon=True).start()
held.wait()
pid = os.fork()
if pid == 0:
    data = reader._range.fetch(start, end)
    own = reader._range._client.pid == os.getpid() and cs.FakeClient.seen[-1]["if_match"] == cs.ETAG and len(data) == end - start
    _storage._sdk_client("s3", {}, True, object)
    os._exit(0 if own else 3)
deadline = time.time() + 20
status = "hung"
while time.time() < deadline:
    done, code = os.waitpid(pid, os.WNOHANG)
    if done:
        status = str(os.waitstatus_to_exitcode(code))
        break
    time.sleep(0.1)
if status == "hung":
    os.kill(pid, 9)
    os.waitpid(pid, 0)
print(status)
"""


def check_fork_with_range_reader(clip: Path, output: Path) -> dict[str, Any]:
    index, _ = sequential_reference(clip, 0)
    path = output / "clip.index.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(index, stream)
    frame_at = {order: f for f, order in enumerate(index["decode_order"])}
    start, end = index["packet_start"][frame_at[0]], index["packet_end"][frame_at[5]]
    try:
        out = subprocess.run([sys.executable, "-c", FORK_RANGE_CODE, str(Path(__file__).resolve().parent), str(clip), str(path),
                              str(start), str(end)], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"passed": False, "problem": "timed out"}
    status = out.stdout.strip()
    return {"passed": out.returncode == 0 and status == "0", "child_exit": status, "stderr": out.stderr[-800:]}


def check_fork_with_blocked_download(large: Path, mode: str, expected: int) -> dict[str, Any]:
    try:
        out = subprocess.run([sys.executable, "-c", FORK_BLOCKED_CODE, str(Path(__file__).resolve().parent), str(large), mode],
                             capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {"passed": False, "problem": "timed out"}
    status, frames, pushed = (out.stdout.split() + ["", "", ""])[:3]
    return {"passed": out.returncode == 0 and status == "0" and frames == str(expected), "child_exit": status,
            "parent_frames": frames, "expected_frames": expected, "pushed_at_fork": pushed, "stderr": out.stderr[-800:]}


REAL_CLIENTS = {"s3": _storage.S3Client, "oss": _storage.OssClient}
HTTP_ETAG = '"real-http-etag"'


class ObjectServer(ThreadingHTTPServer):
    """本机的对象存储：对任何路径都按 Range 返回同一份字节，按 plan 注入故障，记下每个请求。不校验签名。"""

    daemon_threads = True

    def __init__(self, data: bytes) -> None:
        super().__init__(("127.0.0.1", 0), ObjectHandler)
        self.data = data
        self.etag = HTTP_ETAG
        self.checksum = False  # 每个 206 响应都带整个对象的 CRC32
        self.plan: list[dict[str, Any]] = []
        self.seen: list[dict[str, Any]] = []
        self.trickling = threading.Event()


class ObjectHandler(BaseHTTPRequestHandler):
    """HTTP/1.0，每个响应后断开。Range 可以到对象末尾（bytes=a-）或有限（bytes=a-b）。
    plan 每项可选：hangup（收到请求后直接断开）、status（返回这个错误状态和 XML 错误体）、
    drop_after（声明完整长度，只发这么多字节就断开；change_etag 时随后把对象的 ETag 改掉）、
    trickle_after（发这么多字节后每 50 ms 发 1 字节，直到客户端断开）。If-Match 与当前 ETag 不同时回 412。
    server.checksum 为真时，每个 206 响应都带整个对象的 x-amz-checksum-crc32，续读的响应也一样。"""

    server: ObjectServer

    def log_message(self, *args: Any) -> None:
        pass

    def error(self, status: int, code: str) -> None:
        body = f'<?xml version="1.0" encoding="UTF-8"?><Error><Code>{code}</Code><Message>injected</Message><RequestId>1</RequestId></Error>'.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("x-oss-request-id", "1")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        server = self.server
        action = server.plan.pop(0) if server.plan else {}
        requested, if_match = self.headers.get("Range"), self.headers.get("If-Match")
        server.seen.append({"path": self.path, "range": requested, "if_match": if_match,
                            "checksum_mode": self.headers.get("x-amz-checksum-mode")})
        if "delay" in action:
            time.sleep(action["delay"])  # 迟迟不回响应头
        if action.get("hangup"):
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        if "status" in action:
            self.error(action["status"], "SlowDown" if action["status"] == 503 else "Injected")
            return
        if if_match is not None and if_match != server.etag:
            self.error(412, "PreconditionFailed")
            return
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested or "")
        if match is None:
            self.error(400, "InvalidRange")
            return
        start = int(match[1])
        data = server.data[:int(match[2]) + 1] if match[2] else server.data  # 有限区间只发到终点
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(server.data)}")
        self.send_header("Content-Length", str(len(data) - start))
        self.send_header("ETag", server.etag)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Last-Modified", "Sat, 24 Oct 2026 00:00:00 GMT")
        self.send_header("x-oss-request-id", "1")
        if server.checksum:
            self.send_header("x-amz-checksum-crc32", base64.b64encode(zlib.crc32(server.data).to_bytes(4, "big")).decode())
        self.end_headers()
        body = data[start:]
        try:
            if "drop_after" in action:
                self.wfile.write(body[:action["drop_after"]])
                self.wfile.flush()
                if action.get("change_etag"):
                    server.etag = '"rewritten-object"'
                self.connection.shutdown(socket.SHUT_RDWR)
            elif "trickle_after" in action:
                n = action["trickle_after"]
                self.wfile.write(body[:n])
                self.wfile.flush()
                server.trickling.set()
                for i in range(n, min(len(body), n + 600)):  # 最多滴 30 秒
                    self.wfile.write(body[i:i + 1])
                    self.wfile.flush()
                    time.sleep(0.05)
            else:
                self.wfile.write(body)
        except OSError:
            pass  # 客户端已断开


def real_http_checks(clip: Path) -> dict[str, Any]:
    """boto3 与 OSS SDK 走真实的 urllib3、requests 连接本机服务：正常读、响应体中途断开、第一次请求 503、
    第一次请求直接断开都能读完且与本地相同（OSS 的请求阶段网络错误要解开 SDK 的两层包装才能判断为可重试）；
    每个 206 响应都带整个对象的校验值时，boto3 不要求校验、不自动核对，中途断开后续读照常读完；
    断开后对象被改写时续读收到 412 报错；服务器一直慢慢发数据时 close 也能很快返回（靠关闭 socket 读端打断读取，
    读超时不会触发）；续读请求还没收到响应头时 close 最多等 CLOSE_WAIT 秒，之后下载线程自己退出、不再发请求。
    按帧号读：有限区间的 Range 正确、读对；服务器一直慢慢发时到 RANGE_TIMEOUT 就打断读取报错。"""
    import alibabacloud_oss_v2 as oss

    server = ObjectServer(clip.read_bytes())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    saved = {k: os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET")}
    os.environ.update({k: "local-test" for k in saved})  # 本机服务不校验签名，这里只是占位值
    load_default = oss.config.load_default

    def path_style_config() -> Any:
        cfg = load_default()
        cfg.use_path_style = True  # 本机服务没有“桶名.主机”的地址
        return cfg

    oss.config.load_default = path_style_config
    _storage.S3Client, _storage.OssClient = REAL_CLIENTS["s3"], REAL_CLIENTS["oss"]
    chunk, _storage.CHUNK = _storage.CHUNK, 1 << 16
    size = len(server.data)
    expected, _, _ = hashes(clip)
    large_index, large_hashes = sequential_reference(clip, 0)
    checks: dict[str, Any] = {}
    try:
        for scheme, storage in (("s3", {"endpoint_url": endpoint, "region_name": "us-east-1"}), ("oss", {"endpoint": endpoint, "region": "us-east-1"})):
            source = f"{scheme}://bucket/clip.mkv"

            def run(plan: list[dict[str, Any]], requests: int, retries: int) -> dict[str, Any]:
                server.plan, server.seen = [dict(p) for p in plan], []
                try:
                    got, message, stats = hashes(source, storage_options=storage)
                except BaseException as e:  # noqa: BLE001
                    return {"passed": False, "raised": type(e).__name__, "message": str(e)[:500], "seen": server.seen}
                resumes = [r for r in server.seen if r["range"] != "bytes=0-"]
                return {"passed": got == expected and message is None and stats["requests"] == requests and stats["retries"] == retries
                        and all(r["if_match"] == HTTP_ETAG for r in resumes),
                        "requests": stats["requests"], "retries": stats["retries"], "seen": server.seen}

            checks[f"http_{scheme}_clean"] = run([], 1, 0)
            checks[f"http_{scheme}_drop_mid_body"] = run([{"drop_after": size // 3}], 2, 1)
            checks[f"http_{scheme}_first_503"] = run([{"status": 503}], 2, 1)
            checks[f"http_{scheme}_first_hangup"] = run([{"hangup": True}], 2, 1)
            if scheme == "s3":
                # 续读响应只有后半段字节，带的却是整个对象的校验值；boto3 若自动核对，读完这个响应时会误报失败
                server.checksum = True
                result = run([{"drop_after": size // 3}], 2, 1)
                server.checksum = False
                result["passed"] = result["passed"] and all(r["checksum_mode"] is None for r in result["seen"])
                checks["http_s3_full_object_checksum_on_resume"] = result
            # 断开后对象被改写（ETag 变了）：续读带旧 ETag 的 If-Match，服务器回 412，读取报错
            server.plan, server.seen = [{"drop_after": size // 3, "change_etag": True}], []
            changed = expect_raises(vidlo.Error, lambda: hashes(source, storage_options=storage))
            server.etag = HTTP_ETAG
            changed["passed"] = changed["passed"] and "412" in changed["message"] and len(server.seen) == 2
            checks[f"http_{scheme}_object_changed_412"] = changed

            server.plan, server.seen = [{"trickle_after": size - 50000}], []
            server.trickling.clear()
            video = vidlo.VideoReader(source, format="nv12", storage_options=storage)
            next(iter(video))
            entered = server.trickling.wait(10)
            time.sleep(0.3)  # 让下载线程停在读滴流数据的 recv 上
            started = time.perf_counter()
            video.close()
            seconds = time.perf_counter() - started
            checks[f"http_{scheme}_close_during_trickle"] = {"passed": entered and seconds < 2 and wait_threads() == 0 and len(server.seen) == 1,
                                                            "close_seconds": round(seconds, 3), "threads_left": download_threads()}

            # 续读请求还没收到响应头时 close：打断不了这个请求，close 最多等 CLOSE_WAIT 秒返回；
            # 请求结束后下载线程自己退出，不再发请求
            server.plan, server.seen = [{"drop_after": size // 3}, {"delay": 4}], []
            video = vidlo.VideoReader(source, format="nv12", storage_options=storage)
            next(iter(video))
            deadline = time.perf_counter() + 10
            while len(server.seen) < 2 and time.perf_counter() < deadline:
                time.sleep(0.02)
            started = time.perf_counter()
            video.close()
            seconds = time.perf_counter() - started
            left_after_close = download_threads()
            checks[f"http_{scheme}_close_before_response_headers"] = {
                "passed": len(server.seen) == 2 and seconds < _storage.CLOSE_WAIT + 0.5 and wait_threads(10) == 0 and len(server.seen) == 2,
                "close_seconds": round(seconds, 3), "threads_right_after_close": left_after_close, "requests": len(server.seen)}

            # 按帧号读：一次有限区间的请求，Range 正好是规划的区间；服务器一直慢慢发时，到 RANGE_TIMEOUT 就打断读取报错
            # （一直有字节到，SDK 的读超时不会触发）
            frames = [40, 41]
            server.plan, server.seen = [], []
            with vidlo.VideoReader(source, format="nv12", index=large_index, storage_options=storage) as video:
                batch = video.get_frames_at(frames)
                stats = video.stats
            first, last = stats["last_range"]
            checks[f"http_{scheme}_range_clean"] = {
                "passed": [sha(image) for image in batch.images] == [large_hashes[f] for f in frames] and stats["requests"] == 1
                and len(server.seen) == 1 and server.seen[0]["range"] == f"bytes={first}-{last - 1}",
                "seen": server.seen, "last_range": [first, last]}
            server.plan, server.seen = [{"trickle_after": 1000}], []
            timeout, _storage.RANGE_TIMEOUT = _storage.RANGE_TIMEOUT, 1
            try:
                video = vidlo.VideoReader(source, format="nv12", index=large_index, storage_options=storage)
                started = time.perf_counter()
                result = expect_raises(vidlo.Error, lambda: video.get_frames_at(frames))
                result["seconds"] = round(time.perf_counter() - started, 3)
                video.close()
            finally:
                _storage.RANGE_TIMEOUT = timeout
            result["passed"] = (result["passed"] and "did not finish within" in result["message"] and result["seconds"] < 2
                                and len(server.seen) == 1)
            checks[f"http_{scheme}_range_trickle_timeout"] = result
    finally:
        _storage.CHUNK = chunk
        _storage.S3Client = FakeClient  # type: ignore[misc]
        oss.config.load_default = load_default
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        server.shutdown()
    return checks


def range_checks(clip: Path, remote: str) -> dict[str, Any]:
    """按帧号读的有限区间（RangeReader），走假客户端：两次请求各一个 GET，Range 的起止正好是规划的区间，第二次带 If-Match；
    断连从断点续读、终点不变；503 重试；服务端不理会终点、总长与索引不符、200、404、412、ETag 变了、多出字节、连续失败、
    超时都明确报错，之后读取器不再可用；read_sample 从 s3:// 前缀读。"""
    index, local = sequential_reference(clip, 0)
    frame_at = {order: f for f, order in enumerate(index["decode_order"])}

    def range_of(numbers: list[int]) -> tuple[int, int]:
        s = min(index["decode_start"][f] for f in numbers)
        e = max(index["decode_order"][f] for f in numbers)
        return index["packet_start"][frame_at[s]], index["packet_end"][frame_at[e]]

    first, second = [30, 31, 33], [5, 6]
    (a1, b1), (a2, b2) = range_of(first), range_of(second)
    size = index["file_size"]

    def same(batch: vidlo.FrameBatch, numbers: list[int]) -> bool:
        return batch.number == numbers and [sha(image) for image in batch.images] == [local[n] for n in numbers]

    def ok(requests: int, retries: int, resume_at: int | None = None) -> Callable[[], dict[str, Any]]:
        """两次请求都读对；没收到第一个成功响应之前的请求不带 If-Match，之后都带；续读从断点开始、终点不变。"""
        def run() -> dict[str, Any]:
            with vidlo.VideoReader(remote, format="nv12", index=index) as video:
                pictures = same(video.get_frames_at(first), first) and same(video.get_frames_at(second), second)
                stats = video.stats
            seen = FakeClient.seen
            bound = next((i for i, r in enumerate(seen) if r["if_match"] is not None), len(seen))
            ranges = all(r["start"] >= a1 and r["end"] == b1 for r in seen[:-1]) and seen[-1] == {"start": a2, "end": b2, "if_match": ETAG}
            resumed = resume_at is None or any(r["start"] == resume_at and r["if_match"] == ETAG for r in seen)
            return {"passed": pictures and stats["requests"] == requests and stats["retries"] == retries and ranges and resumed
                    and seen[0]["start"] == a1 and all(r["if_match"] == ETAG for r in seen[bound:]),
                    "requests": stats["requests"], "retries": stats["retries"], "reads": stats["reads"]}
        return run

    def fails(kind: type[BaseException], words: str, on_second: bool = False) -> Callable[[], dict[str, Any]]:
        """出错的那次请求报预期的异常，之后读取器不再可用。"""
        def run() -> dict[str, Any]:
            with vidlo.VideoReader(remote, format="nv12", index=index) as video:
                if on_second:
                    video.get_frames_at(first)
                result = expect_raises(kind, lambda: video.get_frames_at(second if on_second else first))
                after = expect_raises(vidlo.Error, lambda: video.get_frames_at(first))
            result["passed"] = result["passed"] and words in result["message"] and after["passed"]
            return result
        return run

    half = (b1 - a1) // 2
    checks: dict[str, Any] = {}
    checks["range_clean"] = scenario([], ok(2, 0))
    checks["range_reset_resumes"] = scenario([{"stop_after": half}], ok(3, 1, resume_at=a1 + half))
    checks["range_short_read_resumes"] = scenario([{"stop_after": half, "quiet": True}], ok(3, 1, resume_at=a1 + half))
    checks["range_first_503"] = scenario([{"error": 503}], ok(3, 1))
    checks["range_end_ignored"] = scenario([{"ignore_end": True}], fails(vidlo.Error, "Content-Range"))
    checks["range_size_differs_from_index"] = scenario([{"content_range": f"bytes {a1}-{b1 - 1}/{size + 1}"}], fails(vidlo.Error, "size"))
    checks["range_200"] = scenario([{"status": 200}], fails(vidlo.Error, "expected 206"))
    checks["range_404"] = scenario([{"error": 404}], fails(FileNotFoundError, "clip.mkv"))
    checks["range_412_on_second"] = scenario([{}, {"error": 412}], fails(vidlo.Error, "HTTP 412", on_second=True))
    checks["range_etag_changed_on_second"] = scenario([{}, {"etag": '"other"'}], fails(vidlo.Error, "ETag changed", on_second=True))
    checks["range_extra_bytes"] = scenario([{"extra": b"xx"}], fails(vidlo.Error, "more bytes than requested"))
    checks["range_three_failures"] = scenario([{"error": 503}] * 3, fails(vidlo.Error, "3 attempts in a row failed"))
    def gives_up(blocked: bool) -> Callable[[], dict[str, Any]]:
        """超时报错、不重试，用时不超过时限太多：每块都慢时不延期，阻塞在读取上时到点被打断。"""
        def run() -> dict[str, Any]:
            started = time.perf_counter()
            result = fails(vidlo.Error, "did not finish within")()
            result["seconds"] = round(time.perf_counter() - started, 3)
            interrupted = not blocked or FakeClient.responses[0]["interrupted"].is_set()
            result["passed"] = result["passed"] and len(FakeClient.seen) == 1 and result["seconds"] < _storage.RANGE_TIMEOUT + 0.5 and interrupted
            return result
        return run

    timeout = _storage.RANGE_TIMEOUT
    _storage.RANGE_TIMEOUT = 0.3
    try:
        checks["range_timeout_slow_chunks"] = scenario([{"slow": 0.2}], gives_up(blocked=False))
        checks["range_timeout_blocked_read"] = scenario([{"block_after": half}], gives_up(blocked=True))
    finally:
        _storage.RANGE_TIMEOUT = timeout

    def sample_from_prefix() -> dict[str, Any]:
        batch = vidlo.read_sample({"video": "clip.mkv", "frames": first}, index, "s3://fake/", format="nv12")
        return {"passed": same(batch, first) and len(FakeClient.seen) == 1}

    checks["range_read_sample"] = scenario([], sample_from_prefix)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    clip = make_video(args.output, "hevc_full_bframes")
    truncated = args.output / "hevc_full_bframes_truncated.mkv"
    make_truncated(clip, truncated)
    size = clip.stat().st_size
    muxed = args.output / "large_ffmpeg.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=30", "-t", "6", "-c:v", "libx265",
                    "-pix_fmt", "yuvj420p", "-color_range", "pc", "-x265-params", "crf=10:keyint=30:min-keyint=30:open-gop=0:log-level=error",
                    str(muxed)], check=True)
    large = args.output / "large.mkv"
    make_tags_at_end(muxed, large)
    tags_at_end = seekhead_points_to_tags_at_end(large)
    # 用完整 FFmpeg 核对重排后的文件仍然有效：包数与 vidlo 交出的帧数相同。ffprobe 读本地文件是可跳转的，
    # 流的 DURATION 标签只在 Tags 里，它读得到就说明按 SeekHead 跳到了文件末尾（修改前的流式输入正是在这里出错）
    large_packets = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                        "stream=nb_read_packets", "-of", "csv=p=0", str(large)], check=True, capture_output=True,
                                       text=True).stdout.strip())
    # 只许 ffprobe 探测开头 64 KiB：Tags 在 2 MB 之后，读得到就只可能是读文件头时按 SeekHead 跳过去的
    duration_tag = subprocess.run(["ffprobe", "-v", "error", "-probesize", "65536", "-analyzeduration", "0", "-select_streams", "v:0",
                                   "-show_entries", "stream_tags=DURATION", "-of", "csv=p=0", str(large)],
                                  check=True, capture_output=True, text=True).stdout.strip()
    _storage.CHUNK = 4096  # 小块，让一段几十 KB 的合成视频也分成很多块、多次跨过队列
    _storage.S3Client = FakeClient  # type: ignore[misc]
    FakeClient.files = {"clip.mkv": clip, "truncated.mkv": truncated, "large.mkv": large}
    remote, remote_truncated = "s3://fake/clip.mkv", "s3://fake/truncated.mkv"
    expected, _, _ = hashes(clip)
    expected_truncated, truncated_message, _ = hashes(truncated)

    def same(requests: int, retries: int, first_attempts: int = 1) -> Callable[[], dict[str, Any]]:
        """读出的画面与本地相同；前 first_attempts 次请求从 0 开始、不带 If-Match，之后的续读都带 If-Match。"""
        def run() -> dict[str, Any]:
            got, message, stats = hashes(remote)
            first, resumes = FakeClient.seen[:first_attempts], FakeClient.seen[first_attempts:]
            return {"passed": got == expected and message is None and stats["requests"] == requests and stats["retries"] == retries
                    and all(r == {"start": 0, "if_match": None} for r in first)
                    and all(r["if_match"] == ETAG and r["start"] > 0 for r in resumes),
                    "frames": len(got), "requests": stats["requests"], "retries": stats["retries"]}
        return run

    def fails(kind: type[BaseException], words: str) -> Callable[[], dict[str, Any]]:
        def run() -> dict[str, Any]:
            result = expect_raises(kind, lambda: hashes(remote))
            result["passed"] = result["passed"] and words in result["message"]
            return result
        return run

    third = size // 3
    checks: dict[str, Any] = {}
    checks["clean"] = scenario([], same(1, 0))
    checks["reset_once"] = scenario([{"stop_after": third}], same(2, 1))
    checks["short_read_once"] = scenario([{"stop_after": third, "quiet": True}], same(2, 1))
    checks["reset_then_503"] = scenario([{"stop_after": third}, {"error": 503}], same(3, 2))
    checks["reset_then_408"] = scenario([{"stop_after": third}, {"error": 408}], same(3, 2))
    checks["three_failures_in_a_row"] = scenario([{"stop_after": third}, {"stop_after": 0}, {"stop_after": 0}],
                                                 fails(vidlo.Error, "3 attempts in a row failed"))
    checks["resume_412"] = scenario([{"stop_after": third}, {"error": 412}], fails(vidlo.Error, "HTTP 412"))
    checks["resume_wrong_range"] = scenario([{"stop_after": third}, {"content_range": f"bytes 0-{size - 1}/{size}"}],
                                            fails(vidlo.Error, "Content-Range"))
    checks["resume_etag_changed"] = scenario([{"stop_after": third}, {"etag": '"other"'}], fails(vidlo.Error, "ETag changed"))
    checks["resume_size_changed"] = scenario([{"stop_after": third}, {"content_range": f"bytes {third}-{size}/{size + 1}"}],
                                             fails(vidlo.Error, "size changed"))
    checks["first_200"] = scenario([{"status": 200}], fails(vidlo.Error, "expected 206"))
    checks["first_404"] = scenario([{"error": 404}], fails(FileNotFoundError, "clip.mkv"))
    checks["first_403"] = scenario([{"error": 403}], fails(vidlo.Error, "HTTP 403"))
    # 第一次请求也按同样的规则重试
    checks["first_503_then_ok"] = scenario([{"error": 503}], same(2, 1, first_attempts=2))
    checks["first_connection_error_then_ok"] = scenario([{"raise": ConnectionResetError("injected")}], same(2, 1, first_attempts=2))
    checks["first_three_failures"] = scenario([{"error": 503}] * 3, fails(vidlo.Error, "3 attempts in a row failed"))
    # 字节齐全之后出的错：不能当成功，也不能靠续读恢复
    checks["error_after_all_bytes"] = scenario([{"after_all": ValueError("injected checksum mismatch")}],
                                               fails(vidlo.Error, "injected checksum mismatch"))
    checks["reset_after_all_bytes"] = scenario([{"after_all": ConnectionResetError("injected reset at the end")}],
                                               fails(vidlo.Error, "injected reset at the end"))
    checks["extra_bytes"] = scenario([{"extra": b"xx"}], fails(vidlo.Error, "more bytes than the object size"))

    def certificate_not_retried() -> dict[str, Any]:
        result = fails(vidlo.Error, "SSLCertVerificationError")()
        result["passed"] = result["passed"] and len(FakeClient.seen) == 2
        return result

    checks["certificate_error_not_retried"] = scenario(
        [{"stop_after": third}, {"raise": ssl.SSLCertVerificationError("injected certificate failure")}], certificate_not_retried)

    # 第一次请求的连续失败一直算到后台线程：两次 503 后响应体在第一个字节前断开，就是连续第 3 次
    def failures_carry_over() -> dict[str, Any]:
        result = fails(vidlo.Error, "3 attempts in a row failed")()
        result["passed"] = result["passed"] and len(FakeClient.seen) == 3
        return result

    checks["first_failures_carry_into_reading"] = scenario([{"error": 503}, {"error": 503}, {"stop_after": 0}], failures_carry_over)

    # TLS：连接被中途关闭（SSLEOFError）按断连重试；证书验证失败包在 requests、urllib3 的包装里也不重试
    import requests
    import urllib3

    def wrapped(inner: BaseException) -> BaseException:
        return requests.exceptions.SSLError(urllib3.exceptions.MaxRetryError(None, "https://example/", reason=urllib3.exceptions.SSLError(inner)))

    checks["tls_eof_retried"] = scenario([{"stop_after": third}, {"raise": ssl.SSLEOFError(8, "EOF occurred in violation of protocol")}],
                                         same(3, 2))
    checks["wrapped_tls_eof_retried"] = scenario([{"stop_after": third}, {"raise": wrapped(ssl.SSLEOFError(8, "EOF"))}], same(3, 2))

    def wrapped_certificate_not_retried() -> dict[str, Any]:
        result = fails(vidlo.Error, "SSLError")()
        result["passed"] = result["passed"] and len(FakeClient.seen) == 2
        return result

    checks["wrapped_certificate_error_not_retried"] = scenario(
        [{"stop_after": third}, {"raise": wrapped(ssl.SSLCertVerificationError("injected certificate failure"))}],
        wrapped_certificate_not_retried)

    def tls_error_after_earlier_eof() -> BaseException:
        """一个不可恢复的 TLS 错误，碰巧发生在处理另一个 SSLEOFError 的时候（__context__ 是 EOF，但原因不是）。"""
        try:
            raise ssl.SSLEOFError(8, "EOF")
        except ssl.SSLEOFError:
            try:
                raise ssl.SSLError(1, "WRONG_VERSION_NUMBER")
            except ssl.SSLError as e:
                return e

    def context_not_cause() -> dict[str, Any]:
        result = fails(vidlo.Error, "WRONG_VERSION_NUMBER")()
        result["passed"] = result["passed"] and len(FakeClient.seen) == 2
        return result

    checks["tls_error_with_unrelated_eof_context_not_retried"] = scenario(
        [{"stop_after": third}, {"raise": tls_error_after_earlier_eof()}], context_not_cause)

    def closed_once_by_download(response: dict[str, Any]) -> bool:
        """打断由取消的一方做，关闭只由下载线程做一次。"""
        return (response["interrupted"].is_set() and "vidlo-download" not in response["interrupted_by"]
                and response["closed_by"] == ["vidlo-download"])

    def close_while_blocked() -> dict[str, Any]:
        video = vidlo.VideoReader(remote, format="nv12")
        next(iter(video))
        entered = FakeClient.blocking.wait(10)  # 下载线程确实停在阻塞的读取上
        started = time.perf_counter()
        video.close()
        seconds = time.perf_counter() - started
        response = FakeClient.responses[0]
        return {"passed": entered and seconds < 3 and wait_threads() == 0 and len(FakeClient.seen) == 1 and closed_once_by_download(response),
                "close_seconds": round(seconds, 3), "threads_left": download_threads(),
                "interrupted_by": response["interrupted_by"], "closed_by": response["closed_by"]}

    checks["close_while_blocked"] = scenario([{"block_after": size - 2000}], close_while_blocked)

    def dropped_while_blocked() -> dict[str, Any]:
        video = vidlo.VideoReader(remote, format="nv12")
        next(iter(video))
        entered = FakeClient.blocking.wait(10)
        del video  # 不调用 close：finalizer 要打断读取、停下线程
        gc.collect()
        response = FakeClient.responses[0]
        return {"passed": entered and wait_threads() == 0 and closed_once_by_download(response) and len(FakeClient.seen) == 1,
                "threads_left": download_threads(), "interrupted_by": response["interrupted_by"], "closed_by": response["closed_by"]}

    checks["dropped_video_stops_download"] = scenario([{"block_after": size - 2000}], dropped_while_blocked)

    def cancel_during_get() -> dict[str, Any]:
        # 第一个响应在三分之一处断开，续读的请求停在 get() 里；这时 close，0.5 s 后请求返回一个迟到的响应
        video = vidlo.VideoReader("s3://fake/large.mkv", format="nv12")
        next(iter(video))
        entered = FakeClient.in_get.wait(10)
        threading.Timer(0.5, FakeClient.release.set).start()
        started = time.perf_counter()
        video.close()
        seconds = time.perf_counter() - started
        late = FakeClient.responses[1] if len(FakeClient.responses) > 1 else None
        # 迟到的响应没人读：下载线程自己打断并关闭它，各一次
        return {"passed": entered and late is not None and late["interrupted_by"] == ["vidlo-download"] and late["closed_by"] == ["vidlo-download"]
                and wait_threads() == 0 and len(FakeClient.seen) == 2 and seconds < 3,
                "close_seconds": round(seconds, 3), "threads_left": download_threads(), "requests": len(FakeClient.seen)}

    _storage.CHUNK = 1 << 16
    checks["cancel_during_get"] = scenario([{"stop_after": large.stat().st_size // 3}, {"wait": True}], cancel_during_get)
    _storage.CHUNK = 4096

    def large_clip() -> dict[str, Any]:
        local, _, local_stats = hashes(large)
        got, message, stats = hashes("s3://fake/large.mkv")
        return {"passed": got == local and message is None and local_stats["seeks"] == 0 and stats["seeks"] == 0 and tags_at_end
                and len(got) == large_packets and bool(duration_tag),
                "frames": len(got), "ffprobe_packets": large_packets, "ffprobe_duration_tag": duration_tag,
                "size": large.stat().st_size, "tags_after_clusters": tags_at_end,
                "seeks": [local_stats["seeks"], stats["seeks"]]}

    # 文件头的 SeekHead 指向文件末尾的 Tags，超出 AVIO 缓冲：输入按只能顺序读交给 libavformat，它不会跳过去
    _storage.CHUNK = 1 << 20
    checks["large_clip_tags_at_end"] = scenario([], large_clip)
    for mode in ("gc", "exit"):
        checks[f"fork_with_blocked_download_{mode}"] = check_fork_with_blocked_download(large, mode, large_packets)
    _storage.CHUNK = 4096

    def truncated_object() -> dict[str, Any]:
        got, message, stats = hashes(remote_truncated)
        return {"passed": got == expected_truncated and message == truncated_message and message is not None,
                "frames": len(got), "message": message}

    checks["truncated_object"] = scenario([], truncated_object)

    def close_early() -> dict[str, Any]:
        got, _, _ = hashes(remote, limit=3)
        return {"passed": len(got) == 3 and wait_threads() == 0, "threads_left": download_threads()}

    checks["close_early_stops_download"] = scenario([], close_early)
    checks.update(range_checks(clip, remote))
    checks["range_fork_inherited_reader_and_held_lock"] = check_fork_with_range_reader(clip, args.output)
    checks.update(real_http_checks(large))

    report = {"vidlo_version": vidlo.__version__, "clip": str(clip), "clip_size": size, "chunk": _storage.CHUNK,
              "frames": len(expected), "checks": checks, "passed": all(c["passed"] for c in checks.values())}
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    for name, c in checks.items():
        print(("PASS " if c["passed"] else "FAIL ") + name + ("" if c["passed"] else "  " + json.dumps(c, ensure_ascii=False)[:600]))
    print("ALL PASSED" if report["passed"] else "SOME FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
