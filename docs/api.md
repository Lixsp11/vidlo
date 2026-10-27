# Reading videos with vidlo

This page covers everything you need to read frames: opening a video, choosing the output, picking frames, reading by frame number, the information that comes with each frame, object storage, errors and multiprocessing. Index files are described in [index-format.md](index-format.md).

## Contents

- [Quick start](#quick-start)
- [What vidlo can read](#what-vidlo-can-read)
- [`VideoReader`](#videoreader)
- [Picking frames](#picking-frames)
- [Reading by frame number](#reading-by-frame-number)
- [`Frame`](#frame)
- [`reader.metadata`](#readermetadata)
- [Color conversion](#color-conversion)
- [Cropping and resizing](#cropping-and-resizing)
- [Object storage](#object-storage)
- [Errors](#errors)
- [Processes and threads](#processes-and-threads)
- [Diagnostics](#diagnostics)
- [Not implemented yet](#not-implemented-yet)

## Quick start

```python
import vidlo

with vidlo.VideoReader("game.mkv", crop=(32, 0, 1248, 720), width=832, height=480) as reader:
    print(reader.metadata.width, reader.metadata.height)  # 1280 720, size of the video itself
    for frame in reader:
        frame.image      # numpy uint8 array, shape (480, 832, 3), RGB
        frame.number     # 0, 1, 2, ... in display order
        frame.pts        # presentation timestamp, in units of reader.metadata.time_base
```

Decoding runs on the GPU (NVDEC); cropping, color conversion and resizing run on the GPU as well (CV-CUDA). Only the final image is copied to CPU memory.

## What vidlo can read

A video is read either **sequentially, from the first frame to the last**, or **by frame number** with a [video index](index-format.md#video-index) (see [Reading by frame number](#reading-by-frame-number)). Reading by frame number works on local files only in this release.

| | Supported now | Otherwise |
| --- | --- | --- |
| Container | Matroska (`.mkv`) | `UnsupportedError` |
| Codec | HEVC (H.265), 8-bit, 4:2:0, progressive | `UnsupportedError` |
| GOP structure | every GOP starts with an IDR frame (closed GOPs, B-frames allowed); one frame per packet | `UnsupportedError` |
| Parameter sets | in the container's codec configuration; copies inside the stream must be identical | `UnsupportedError` |
| RGB output | full range (0–255), chroma sample location type 0, matrix BT.709, BT.601 or BT.2020 | `UnsupportedError`; `format="nv12"` still works |
| Sources | local files, S3-compatible storage (`s3://`, e.g. Cloudflare R2), Alibaba Cloud OSS (`oss://`); reading by frame number: local files | `ValueError` for other schemes; `UnsupportedError` for reading by frame number from object storage |

A recording that was cut off (for example the recorder crashed) is read up to the last complete frame; see [Errors](#errors).

## `VideoReader`

```python
vidlo.VideoReader(
    source,
    *,
    device=0,
    crop=None,
    width=None,
    height=None,
    interpolation=None,
    colorspace=None,
    format="rgb24",
    index=None,
    storage_options=None,
)
```

Creating a `VideoReader` opens the video and reads its header; the GPU is used from the first frame on. Every argument except `source` is keyword-only.

| Argument | Meaning | Default |
| --- | --- | --- |
| `source` | Local path, `s3://bucket/key` or `oss://bucket/key` | required |
| `device` | GPU index, as seen by this process (`CUDA_VISIBLE_DEVICES` applies) | `0` |
| `crop` | `(x, y, width, height)` of the region to keep, like OpenCV's `cv::Rect`, relative to the visible picture. All four numbers must be even, because 2×2 pixels share one chroma sample | no cropping |
| `width`, `height` | Output size. Give one of them to change only that side; the other keeps the cropped size (no aspect-ratio adjustment) | no resizing |
| `interpolation` | `"nearest_exact"`, `"linear"`, `"cubic"` or `"area"`; `None` chooses automatically, see [Cropping and resizing](#cropping-and-resizing) | automatic |
| `colorspace` | Matrix for YUV → RGB: `"bt601"`, `"bt709"` or `"bt2020"`; `None` uses the matrix tagged in the video, see [Color conversion](#color-conversion) | from the video |
| `format` | `"rgb24"`: RGB images. `"nv12"`: the decoder's own output, not cropped, resized or converted; `crop`, `width`, `height`, `interpolation` and `colorspace` must then be left out | `"rgb24"` |
| `index` | [Video index](index-format.md#video-index) (path or dict) for [reading by frame number](#reading-by-frame-number). The reader then reads only by frame number and never reads the file header | sequential reading |
| `storage_options` | Connection settings for `s3://` and `oss://`, see [Object storage](#object-storage) | — |

A `VideoReader` is a context manager. `reader.close()` stops downloading and frees the decoder and GPU memory; images you already received stay valid.

## Picking frames

```python
for frame in reader: ...                          # every frame, in display order
for frame in reader.frames(step=2): ...           # frames 0, 2, 4, ...
for frame in reader.frames(fps=30, source_fps=60): ...  # frame-rate conversion, see below
```

- `step=k` keeps every k-th frame, starting with frame 0.
- `fps` together with `source_fps` keeps frames so that `source_fps` frames per second become `fps` frames per second. The choice is the same as VPF's `EMDownSampler`: it compares windows of 0.5, 1 and 3 seconds and also works for ratios that are not whole numbers (60 → 24). Note the phase: `fps=30, source_fps=60` keeps frames 1, 3, 5, ..., while `step=2` keeps 0, 2, 4, .... You pass both rates yourself; vidlo does not read the frame rate from the file.
- `step` and `fps`/`source_fps` cannot be used together.

Frames that are not kept are still decoded (later frames may depend on them) and checked, but they are not converted or copied. A `VideoReader` can be read sequentially only once; open a new one to read the video again.

## Reading by frame number

With a [video index](index-format.md#video-index), a reader fetches only the frames you ask for:

```python
index = vidlo.load_video_index("game.index.json")
with vidlo.VideoReader("game.mkv", index=index, crop=(16, 0, 1248, 720), width=832, height=480) as reader:
    len(reader)                                     # frames in the index
    batch = reader.get_frames_at([3000, 3002, 3000])
    batch.images                                    # numpy uint8, shape (3, 480, 832, 3)
    frame = reader[3000]                            # one Frame, same as reader.get_frame_at(3000)
    clip = reader.get_frames_in_range(3000, 3161, 2)  # 81 frames, every second one
```

| Call | Returns |
| --- | --- |
| `get_frames_at(frames)` | A [`FrameBatch`](#framebatch) in the order of `frames`; any order and repeats are allowed, negative numbers count from the end (`-1` is the last frame) |
| `get_frame_at(i)`, `reader[i]` | One [`Frame`](#frame) |
| `get_frames_in_range(start, stop, step=1)` | A `FrameBatch` for `range(start, stop, step)`; bounds outside the video are clipped as for a slice; `step` must be positive |
| `reader[a:b:k]` | A `FrameBatch`; the slice may go backwards |
| `len(reader)` | The number of frames in the index |

- A frame number outside the video raises `IndexError`, anything that is not an integer raises `ValueError`, both before anything is read. An empty list returns an empty batch.
- One request reads one contiguous byte range, from the key frame that starts the GOP of the earliest requested frame to the last packet needed in decoding order, and decodes that range. Frames needed only as references are decoded but not converted or copied; frames that no other frame refers to are not decoded at all unless requested. Frames far apart (for example frames 0 and 3000) make one request read and decode everything in between; ask for them separately.
- A request returns all its frames or raises; after an error the reader reads nothing more.
- The file header is never read. `metadata.container`, `time_base`, `container_range`, `container_matrix`, `container_chroma_location` and every `pts` are `None` (the index stores no timestamps); size and color tags come from the index's `codec_config`. The file size must equal the index's `file_size`.
- A reader opened with an index cannot be iterated; one opened without an index has no `len()` and no reading by frame number.
- Hardware decoders are kept for the next request, also across readers and files as long as the video format (codec, coded size, bit depth) is the same: creating and destroying one costs tens to a few hundred milliseconds. Each process keeps one idle decoder per GPU; `vidlo.set_decoder_cache_capacity(n)` changes that, and `0` turns the cache off and frees the idle decoders (each holds some GPU memory).

### `FrameBatch`

`images` is one `numpy.uint8` array: `(frames, height, width, 3)` for RGB, `(frames, height * 3 / 2, width)` for NV12. Repeated frame numbers get separate copies. The other attributes have the names of the [`Frame`](#frame) attributes and are lists with one entry per requested position.

### `vidlo.read_sample`

```python
batch = vidlo.read_sample("fight-03-a.json", "game.index.json", "/data", step=2, width=832, height=480)
```

`vidlo.read_sample(sample_index, video_index, root, *, step=None, fps=None, source_fps=None, ...)` reads one [sample index](index-format.md#sample-index): it opens `root` + the sample's `video` with the video index, picks frames from the sample's list with `step` or `fps`/`source_fps` (by position in the list, as in [Picking frames](#picking-frames)) and returns the `FrameBatch`. The other keyword arguments are those of `VideoReader` except `source` and `index`. Both indexes and every frame number of the sample are checked first, including frames that `step` would drop. `root` is a local directory, or an `s3://` or `oss://` prefix (not implemented yet for reading by frame number).

## `Frame`

Each frame is a `vidlo.Frame` with these attributes:

| Attribute | Meaning |
| --- | --- |
| `image` | `numpy.uint8` array. `rgb24`: shape `(height, width, 3)`. `nv12`: shape `(height * 3 / 2, width)`, the luma rows followed by the interleaved chroma rows |
| `number` | Frame number in display order, starting at 0; frames that were not kept are counted too |
| `pts` | Presentation timestamp from the file, an integer in units of `reader.metadata.time_base`; `None` when reading by frame number |
| `key_frame` | Whether the frame is a key frame (IDR) |
| `decode_status` | Decoder status of this frame; frames you receive always decoded successfully |
| `decode_order` | Position of this frame's packet in decoding order |
| `decode_start` | Decoding position from which this frame can be decoded correctly (its GOP's key frame) |
| `packet_start`, `packet_end` | Byte range of this frame's compressed data in the file (end exclusive), without the container's block header |
| `colorspace` | Matrix actually used for RGB (`None` for `nv12`) |
| `interpolation` | Interpolation actually used (`None` when the image was not resized, and for `nv12`) |

`decode_order`, `decode_start`, `packet_start` and `packet_end` are what a [video index](index-format.md#video-index) stores.

## `reader.metadata`

`reader.metadata` is a `vidlo.VideoMetadata` with information from the file header. It is available as soon as the reader is created, and still after it is closed. When reading by frame number the header is not read: the attributes that come from the container (`container`, `time_base`, `container_range`, `container_matrix`, `container_chroma_location`, `etag`) are `None`, the rest comes from the video index.

| Attribute | Meaning |
| --- | --- |
| `container`, `codec` | Container format (e.g. `"matroska,webm"`) and codec (`"hevc"`) |
| `codec_config` | The codec configuration from the container (bytes; for HEVC in Matroska, the `hvcC` record) |
| `time_base` | `(numerator, denominator)`; `pts * numerator / denominator` is in seconds |
| `file_size` | File size in bytes |
| `width`, `height` | Size of the visible picture in the video, not affected by `crop`, `width` and `height` |
| `full_range` | `True` for full range (0–255), `False` for limited range (16–235) |
| `matrix_coefficients` | Matrix tagged in the video (H.265 Table E.5: 1 BT.709, 5 or 6 BT.601, 9 BT.2020, 2 not tagged) |
| `chroma_location` | Chroma sample location type tagged in the video (0 when not tagged) |
| `container_range`, `container_matrix`, `container_chroma_location` | The same color information as written by the container, as FFmpeg enum values; vidlo checks that it agrees with the video |
| `etag` | ETag of the object for `s3://` and `oss://` sources (`None` for local files). vidlo uses it to make sure the object does not change while reading; you can store it to notice later that an object was replaced |

The frame count and frame rate are not in `metadata`: a Matroska file does not record them reliably before it has been read.

## Color conversion

Videos store YUV, not RGB; converting to RGB needs a matrix (BT.601, BT.709, ...).

- By default (`colorspace=None`) vidlo uses the matrix tagged in the video. A video without a tag (`matrix_coefficients` 2) or with an unsupported one raises `UnsupportedError`; pass `colorspace` to choose the matrix yourself. vidlo does not guess.
- `colorspace="bt601"`, `"bt709"` or `"bt2020"` uses that matrix regardless of the tag. `"bt601"` gives the colors of OpenCV's `cv2.VideoCapture` (which always uses BT.601) within 2 levels per channel.
- The range always follows the video. RGB output needs full range and chroma sample location type 0; anything else raises `UnsupportedError` (use `format="nv12"` to get the decoder output instead).
- If the container and the video disagree on range, matrix or chroma location, opening raises `vidlo.Error`.

## Cropping and resizing

The image is cropped first, then resized.

- With `interpolation=None`, vidlo uses `"area"` when the image gets smaller in either direction, otherwise `"cubic"`.
- `"area"` works only when no direction gets larger (CV-CUDA 0.19 computes it wrongly when enlarging). Enlarging with `"area"`, or making one side smaller and the other larger with `interpolation=None`, raises `UnsupportedError`; pass `"linear"` or `"cubic"` instead.
- `"nearest_exact"` takes for output pixel `i` the source pixel `floor((i + 0.5) * source_size / output_size)`, like OpenCV's `INTER_NEAREST_EXACT` and PIL; it is not OpenCV's `INTER_NEAREST`.
- When shrinking, `"area"` differs from OpenCV's `INTER_AREA` by at most 1 level (measured for scales 0.35–0.69); `"linear"` and `"cubic"` differ from OpenCV's `INTER_LINEAR` and `INTER_CUBIC` by at most 1.

## Object storage

```python
reader = vidlo.VideoReader(
    "s3://my-bucket/videos/game.mkv",
    storage_options={"endpoint_url": "https://<account>.r2.cloudflarestorage.com", "region_name": "auto"},
)
reader = vidlo.VideoReader(
    "oss://my-bucket/videos/game.mkv",
    storage_options={"region": "us-southeast-1", "use_internal_endpoint": True},
)
```

| Scheme | Package | `storage_options` keys | Credentials (environment variables) |
| --- | --- | --- | --- |
| `s3://` | `boto3`, `urllib3>=2.3` (extra `s3`) | `endpoint_url`, `region_name` (as in boto3) | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN` |
| `oss://` | `alibabacloud-oss-v2`, `urllib3>=2.3` (extra `oss`) | `region` (required), `endpoint`, `use_internal_endpoint` (`True`/`False`) | `OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET` |

- Credentials are read only from these environment variables; they are never passed through `storage_options`, printed or logged.
- The object is downloaded with one streaming request while decoding goes on; at most 64 MiB is buffered ahead.
- After a broken connection, timeout, rate limit or temporary server error, the download continues from where it stopped, provided the object has not changed (its ETag is checked); 3 failures in a row raise `vidlo.Error`. A missing object raises `FileNotFoundError`.

## Errors

| Exception | When |
| --- | --- |
| `vidlo.Error` (a `RuntimeError`) | Reading or decoding failed, the data is inconsistent, or the reader is used after `close()` |
| `vidlo.UnsupportedError` (a `vidlo.Error`) | The video or the settings need something that is planned but not implemented yet |
| `vidlo.TruncatedError` (a `vidlo.Error`) | The file is cut off. It is raised only after every complete frame has been delivered |
| `ValueError` | Invalid arguments or combinations of arguments, including `storage_options`; reading by frame number without a video index |
| `IndexError` | A frame number outside the video |
| `FileNotFoundError` | The local file or the remote object does not exist |

After an error (other than `TruncatedError` at the end) the reader delivers no more frames.

## Processes and threads

- Use a `VideoReader` from one thread at a time. Several readers can be open at once; each has its own decoder.
- `import vidlo` does not initialize CUDA, so worker processes started after the import can use vidlo.
- CUDA does not work in a process forked after its parent used CUDA. In such a child, opening a video or using a reader inherited from the parent raises `vidlo.Error` that tells you to use the `spawn` or `forkserver` start method, for example `DataLoader(..., multiprocessing_context="spawn")`. If the parent used CUDA through another library, vidlo cannot tell beforehand; CUDA initialization then fails in the child with the same advice.
- Pass images between processes as CPU arrays; GPU memory is never shared.
- Each process has its own cache of idle hardware decoders (see [Reading by frame number](#reading-by-frame-number)); a forked child starts with an empty one.

## Diagnostics

- `reader.stats`: packets, decoder calls, frames shown and kept, decode status counts, bytes read, seeks, where a truncated file ends, and for remote sources the number of requests and retries. When reading by frame number, totals over all requests instead: requests, bytes read and read calls, packets sent to the decoder and packets `skipped` (no frame refers to them and they were not requested), decoder calls, frames shown and kept, how often the hardware decoder was reused or created, and the byte range of the last request (`last_range`).
- `reader.decoder_config`: driver version, GPU name and UUID, and the parameters passed to NVDEC together with what it reported (available after the first frame; when reading by frame number, those of the last request).

## Not implemented yet

These are part of the interface and raise `vidlo.UnsupportedError` for now:

- Reading by frame number (`index=...`, `read_sample`) from `s3://` and `oss://` sources.
- `reader.submit(frame_numbers)`: `get_frames_at` asynchronously.
