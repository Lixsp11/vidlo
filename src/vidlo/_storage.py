"""对象存储的顺序读取。

一个流式 GET（Range: bytes=0-）从头读到尾，由后台线程按大块推给 C++ 的 StreamInput；队列有上限，下载最多领先
解封装 CAPACITY 字节。连接中断、超时、限流或合适的 5xx 时从断点重新请求（Range: bytes=断点-，If-Match 带第一次
响应的 ETag，确认还是同一个对象），连续失败 MAX_ATTEMPTS 次就报错；每次响应都核对 206、Content-Range 和对象总长。
SDK 自己的重试关掉，重试次数只由这里控制。凭证只从环境变量读，不打印、不记录。
"""

from __future__ import annotations

import os
import random
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from . import _native
from ._native import Error

CHUNK = 8 << 20
CAPACITY = 64 << 20
MAX_ATTEMPTS = 3
CLOSE_WAIT = 2  # close() 等下载线程退出的最长秒数
_RETRY_STATUS = (408, 429, 500, 502, 503, 504)


class HttpError(Exception):
    """对象存储返回的错误状态。"""

    def __init__(self, status: int, code: str | None, message: str) -> None:
        super().__init__(f"HTTP {status} {code or ''}: {message}".strip())
        self.status = status
        self.code = code


class ShortRead(Exception):
    """响应体在对象末尾之前就结束了（连接提前关闭）。"""


@dataclass
class Response:
    status: int
    content_range: str | None
    etag: str | None
    chunks: Iterator[bytes]
    close: Callable[[], None]  # 由读取响应的线程调用
    interrupt: Callable[[], None]  # 从别的线程打断正在进行的读取：关闭 socket 的读端，不等待，不加锁


def _shutdown(raw: Any) -> Callable[[], None]:
    """urllib3 响应的 shutdown()（2.3 起，客户端创建时已检查版本）：关闭 socket 读端，阻塞在 recv 上的读取立即出错返回。
    不用 close()：它要等读取方持有的缓冲区锁，服务器持续少量发数据时会一直等下去。只在取消时调用；连接若已归还
    连接池，urllib3 2.5 起 shutdown() 抛异常（下面吞掉），2.3、2.4 会打断复用这条连接的请求，那也只可能是这段录像
    自己要停下的请求（每段录像有自己的客户端和连接池）。"""
    if not hasattr(raw, "shutdown"):
        raise Error(f"cannot find the urllib3 response to interrupt (got {type(raw).__name__}); the storage SDK layout changed")

    def interrupt() -> None:
        try:
            raw.shutdown()
        except Exception:  # noqa: BLE001 - 连接已归还或已关闭时没有要打断的读取
            pass
    return interrupt


def _require_urllib3_shutdown() -> None:
    import urllib3

    if not hasattr(urllib3.response.HTTPResponse, "shutdown"):
        raise Error(f"reading object storage needs urllib3>=2.3 (found {urllib3.__version__}) to interrupt reads on close")


def parse_source(source: str) -> tuple[str, str, str]:
    """"s3://桶/路径" 或 "oss://桶/路径" → (scheme, 桶, 路径)。"""
    scheme, _, rest = source.partition("://")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"remote source must look like {scheme}://bucket/key, got {source!r}")
    return scheme, bucket, key


def _environment(*names: str) -> list[str]:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise Error(f"credentials are read only from environment variables; {', '.join(missing)} not set")
    return [os.environ[name] for name in names]


def _check_options(options: dict[str, Any] | None, allowed: tuple[str, ...], required: tuple[str, ...] = ()) -> dict[str, Any]:
    """storage_options 只接受 allowed 里的键，键名与存储 SDK 的同名参数相同，原样交给 SDK；凭证不在其中。"""
    options = dict(options or {})
    unknown = sorted(set(options) - set(allowed))
    if unknown:
        raise ValueError(f"unknown storage_options keys {unknown}; allowed: {list(allowed)}")
    missing = [name for name in required if name not in options]
    if missing:
        raise ValueError(f"storage_options keys {missing} are required")
    return options


class S3Client:
    """S3 兼容存储（R2 等），boto3。storage_options 可给 boto3 客户端的 endpoint_url、region_name（R2 写 "auto"）。
    凭证只用 AWS_ACCESS_KEY_ID、AWS_SECRET_ACCESS_KEY（可选 AWS_SESSION_TOKEN），显式传给 boto3，不走它的配置文件和实例元数据。"""

    def __init__(self, bucket: str, key: str, storage_options: dict[str, Any] | None) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise Error("reading s3:// needs boto3: pip install 'vidlo[s3]'") from e

        _require_urllib3_shutdown()
        options = _check_options(storage_options, ("endpoint_url", "region_name"))
        key_id, secret = _environment("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        self.bucket, self.key = bucket, key
        # 每个客户端用自己的 session：boto3.client() 共用默认 session，多个线程同时建客户端可能出错（boto3 文档）。
        # 响应的校验和只在请求要求时才核对：boto3 1.36 起默认自动核对，续读的 Range 响应若带着整个对象的校验值，
        # 读完会误报失败；完整性靠 Content-Range、对象总长和 If-Match
        self._client = boto3.session.Session().client(
            "s3", endpoint_url=options.get("endpoint_url"), region_name=options.get("region_name"), aws_access_key_id=key_id,
            aws_secret_access_key=secret, aws_session_token=os.environ.get("AWS_SESSION_TOKEN") or None,
            config=Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10,
                          response_checksum_validation="when_required"))

    def get(self, start: int, etag: str | None) -> Response:
        from botocore.exceptions import ClientError

        args: dict[str, Any] = {"Bucket": self.bucket, "Key": self.key, "Range": f"bytes={start}-"}
        if etag is not None:
            args["IfMatch"] = etag
        try:
            result = self._client.get_object(**args)
        except ClientError as e:
            error = e.response.get("Error", {})
            raise HttpError(e.response["ResponseMetadata"]["HTTPStatusCode"], error.get("Code"), error.get("Message", "")) from e
        body = result["Body"]  # botocore StreamingBody，底层是 urllib3 的响应
        try:
            interrupt = _shutdown(getattr(body, "_raw_stream", None))
        except BaseException:
            body.close()
            raise
        return Response(result["ResponseMetadata"]["HTTPStatusCode"], result.get("ContentRange"), result.get("ETag"),
                        body.iter_chunks(CHUNK), body.close, interrupt)


class OssClient:
    """阿里云 OSS，alibabacloud-oss-v2。storage_options 给 SDK 配置的 region（必需）、endpoint、use_internal_endpoint
    （走内网地址，默认 False）。凭证只用 OSS_ACCESS_KEY_ID、OSS_ACCESS_KEY_SECRET。"""

    def __init__(self, bucket: str, key: str, storage_options: dict[str, Any] | None) -> None:
        try:
            import alibabacloud_oss_v2 as oss
        except ImportError as e:
            raise Error("reading oss:// needs alibabacloud-oss-v2: pip install 'vidlo[oss]'") from e

        _require_urllib3_shutdown()
        options = _check_options(storage_options, ("region", "endpoint", "use_internal_endpoint"), required=("region",))
        internal = options.get("use_internal_endpoint", False)
        if not isinstance(internal, bool):
            raise ValueError(f"storage_options use_internal_endpoint must be True or False, got {internal!r}")
        _environment("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET")
        cfg = oss.config.load_default()
        cfg.credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
        cfg.region = options["region"]
        cfg.use_internal_endpoint = internal
        if "endpoint" in options:
            cfg.endpoint = options["endpoint"]
        cfg.retryer = oss.retry.NopRetryer()
        cfg.connect_timeout = 2
        cfg.readwrite_timeout = 5
        self.bucket, self.key = bucket, key
        self._oss = oss
        self._client = oss.Client(cfg)

    def get(self, start: int, etag: str | None) -> Response:
        oss = self._oss
        request = oss.GetObjectRequest(bucket=self.bucket, key=self.key, range_header=f"bytes={start}-", range_behavior="standard",
                                       if_match=etag)
        try:
            result = self._client.get_object(request)
        except oss.exceptions.OperationError as e:
            # SDK 的包装：OperationError → RequestError/ResponseError → 底层的 requests 异常，或 ServiceError
            cause = e.unwrap()
            while isinstance(cause, (oss.exceptions.RequestError, oss.exceptions.ResponseError)) and cause.unwrap() is not None:
                cause = cause.unwrap()
            if isinstance(cause, oss.exceptions.ServiceError):
                raise HttpError(cause.status_code, cause.code, str(getattr(cause, "message", ""))) from e
            raise cause from e
        # body 是 StreamBodyReader → SDK 的 requests 响应包装 → requests.Response，底层是 urllib3 的响应
        raw = getattr(getattr(getattr(result.body, "_response", None), "_internal_response", None), "raw", None)
        try:
            interrupt = _shutdown(raw)
        except BaseException:
            result.body.close()
            raise
        return Response(result.status_code, result.content_range, result.etag, result.body.iter_bytes(block_size=CHUNK),
                        result.body.close, interrupt)


def check_response(response: Response, start: int, size: int | None, etag: str | None) -> int:
    """核对 Range 响应，返回对象总长。200 表示服务端忽略了 Range，不算成功。"""
    if response.status != 206:
        raise Error(f"range read from byte {start} returned HTTP {response.status}, expected 206")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.content_range or "")
    if match is None:
        raise Error(f"range read from byte {start} returned Content-Range {response.content_range!r}")
    first, last, total = (int(g) for g in match.groups())
    if first != start or last != total - 1:
        raise Error(f"range read from byte {start} returned Content-Range {response.content_range!r}")
    if size is not None and total != size:
        raise Error(f"object size changed from {size} to {total} during reading")
    if etag is not None and response.etag != etag:
        raise Error(f"object ETag changed from {etag} to {response.etag} during reading")
    return total


def _transient_errors() -> tuple[type[BaseException], ...]:
    """可以从断点重试的网络临时故障：连接失败、连接中断、超时、响应提前结束。只列具体类型，
    证书、参数、数据校验等错误不在其中。"""
    kinds: list[type[BaseException]] = [ShortRead, ConnectionError, TimeoutError]
    try:
        from botocore import exceptions as b

        kinds += [b.EndpointConnectionError, b.ConnectTimeoutError, b.ReadTimeoutError, b.ConnectionClosedError,
                  b.ResponseStreamingError, b.IncompleteReadError]
    except ImportError:
        pass
    try:
        from requests import exceptions as r

        kinds += [r.ConnectionError, r.Timeout, r.ChunkedEncodingError]
    except ImportError:
        pass
    try:
        from urllib3 import exceptions as u

        kinds += [u.ProtocolError, u.ReadTimeoutError, u.ConnectTimeoutError, u.NewConnectionError]
    except ImportError:
        pass
    return tuple(kinds)


def _causes(error: BaseException) -> list[BaseException]:
    """异常本身和它明确包着的异常：__cause__（raise … from）、urllib3 的 reason、args 里的异常、botocore 放在
    kwargs 里的 error。不看 __context__：那是处理另一个异常时碰巧发生的，不是这个异常的原因。"""
    seen: list[BaseException] = []
    pending = [error]
    while pending and len(seen) < 32:
        e = pending.pop()
        if any(e is s for s in seen):
            continue
        seen.append(e)
        kwargs = getattr(e, "kwargs", None)
        inner = [e.__cause__, getattr(e, "reason", None), *getattr(e, "args", ()),
                 *(kwargs.values() if isinstance(kwargs, dict) else ())]
        pending.extend(x for x in inner if isinstance(x, BaseException))
    return seen


def retryable(error: BaseException) -> bool:
    """请求超时（408）、限流、合适的 5xx 和网络临时故障可以重试；其他状态（404、403、412 等）和数据核对失败不重试。

    TLS 的错误要看包在里面的原因：证书验证失败不重试；TLS 连接被中途关闭（SSLEOFError、SSLZeroReturnError）
    与普通断连一样重试；其他 TLS 错误不重试。requests、botocore 的 SSLError 都是它们 ConnectionError 的子类，
    所以要在查白名单之前判断。
    """
    import ssl

    if isinstance(error, HttpError):
        return error.status in _RETRY_STATUS
    causes = _causes(error)
    if any(isinstance(e, ssl.SSLCertVerificationError) for e in causes):
        return False
    tls = [e for e in causes if isinstance(e, ssl.SSLError) or type(e).__name__ == "SSLError"]
    if tls:
        return any(isinstance(e, (ssl.SSLEOFError, ssl.SSLZeroReturnError)) for e in causes)
    return isinstance(error, _transient_errors())


class Stopped(Exception):
    """读取方已关闭，下载停止。"""


class Download:
    """一次顺序读取：首个响应在打开时取得（拿到对象总长和 ETag），其余在后台线程里读。

    请求与读取共用一个连续失败计数：临时故障退避后从断点重新请求，连续失败 MAX_ATTEMPTS 次报错，有字节推入才清零；
    第一次请求也按同样的规则重试，计数一直带到后台线程。字节已经齐全之后出的错（例如响应在末尾才报校验失败）不重试，
    直接报错。cancel() 设停止标志、关闭 StreamInput，并打断正在读的响应（关闭 socket 读端，不等待）；响应只由下载线程
    自己关闭。停止标志与当前响应在同一把锁下登记，取消时刚好返回的响应也会被打断并丢弃。
    还没收到响应头的请求打断不了（SDK 不提供这样的接口）：close() 最多等 CLOSE_WAIT 秒，下载线程在这个请求结束
    （返回或超时）后自己退出，丢掉迟到的响应，不再推数据，也不再发请求。
    """

    def __init__(self, client: Any, source: str) -> None:
        self._client = client
        self.source = source
        self._process = os.getpid()
        self.requests = 0
        self.retries = 0
        self.size: int = -1
        self.etag: str | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._response: Response | None = None
        failures = 0
        while True:
            try:
                first = self._open(0)
                break
            except Exception as e:  # noqa: BLE001 - 按类型决定是否重试
                failures = self._failed(failures, e, 0)
        try:
            if not first.etag:
                raise Error(f"{source}: response has no ETag, cannot confirm the object stays the same")
            self.etag = first.etag
            self.input = _native.StreamInput(self.size, CAPACITY)
        except BaseException:
            self._close_response()
            raise
        self._thread = threading.Thread(target=self._run, args=(first, failures), name="vidlo-download", daemon=True)
        self._thread.start()

    def _open(self, position: int) -> Response:
        """发一次范围请求并核对响应；第一次请求时记下对象总长。"""
        if self._stop.is_set():
            raise Stopped()
        self.requests += 1
        response = self._client.get(position, self.etag)
        with self._lock:
            stopped = self._stop.is_set()
            if not stopped:
                self._response = response
        if stopped:  # 请求期间已经取消：这个响应没人读，打断并关闭
            response.interrupt()
            response.close()
            raise Stopped()
        try:
            total = check_response(response, position, None if self.size < 0 else self.size, self.etag)
        except BaseException:
            self._close_response()
            raise
        if self.size < 0:
            self.size = total
        return response

    def _failed(self, failures: int, error: BaseException, position: int) -> int:
        """记一次失败：可以重试就退避并返回新的连续失败数，否则抛出。"""
        if self._stop.is_set():
            raise Stopped() from error
        if not retryable(error):
            raise error
        failures += 1
        if failures >= MAX_ATTEMPTS:
            raise Error(f"{failures} attempts in a row failed at byte {position}: {type(error).__name__}: {error}") from error
        self.retries += 1
        if self._stop.wait(0.5 * 2 ** (failures - 1) * (1 + random.random())):
            raise Stopped() from error
        return failures

    def _close_response(self) -> None:
        with self._lock:
            response, self._response = self._response, None
        if response is not None:
            try:
                response.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响已读到的数据
                pass

    def _run(self, response: Response | None, failures: int) -> None:
        position = 0
        try:
            while True:
                try:
                    if response is None:
                        response = self._open(position)
                    chunks = iter(response.chunks)
                    while True:
                        if self._stop.is_set():
                            return  # 读取方已关闭；先查再读，不为已关闭的读取方再读一块
                        chunk = next(chunks, None)
                        if chunk is None:
                            break
                        if not self.input.push(chunk):
                            return
                        position += len(chunk)
                        failures = 0
                    if position < self.size:
                        raise ShortRead(f"response ended at byte {position} of {self.size}")
                    break  # 响应正常结束，字节齐全
                except Stopped:
                    raise
                except Exception as e:  # noqa: BLE001 - 按类型决定是否重试
                    if position >= self.size and not self._stop.is_set():
                        raise  # 字节已经齐全后出的错不能靠续读恢复
                    failures = self._failed(failures, e, position)
                finally:
                    self._close_response()
                    response = None
            self.input.finish()
        except Stopped:
            return
        except BaseException as e:  # noqa: BLE001 - 交给读取方，在它读到这里时报错
            if not self._stop.is_set():
                self.input.fail(f"{self.source}: {type(e).__name__}: {e}")

    def cancel(self) -> None:
        """停止下载，不等线程退出、不阻塞（也供没有 close 就丢掉读取器时的 finalizer 调用）。"""
        if os.getpid() != self._process:
            return  # fork 出的子进程里没有下载线程；fork 时别的线程可能正持有锁，碰锁会卡住
        self._stop.set()
        self.input.close()
        with self._lock:
            response = self._response
        if response is not None:
            response.interrupt()

    def close(self) -> None:
        if os.getpid() != self._process:
            return
        self.cancel()
        self._thread.join(timeout=CLOSE_WAIT)


def open_download(source: str, storage_options: dict[str, Any] | None) -> Download:
    scheme, bucket, key = parse_source(source)
    client = S3Client(bucket, key, storage_options) if scheme == "s3" else OssClient(bucket, key, storage_options)
    try:
        return Download(client, source)
    except HttpError as e:
        if e.status == 404:
            raise FileNotFoundError(source) from e
        raise Error(f"{source}: {e}") from e
