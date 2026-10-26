# Reading videos with vidlo

This page covers everything you need to read frames: opening a video, choosing the output, picking frames, the information that comes with each frame, object storage, errors and multiprocessing. Index files are described in [index-format.md](index-format.md).

## Contents

- [Quick start](#quick-start)
- [What vidlo can read](#what-vidlo-can-read)
- [`VideoReader`](#videoreader)
- [Picking frames](#picking-frames)
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

This release reads a video **sequentially, from the first frame to the last**. Reading by frame number is designed but not implemented yet (see [Not implemented yet](#not-implemented-yet)).

| | Supported now | Otherwise |
| --- | --- | --- |
| Container | Matroska (`.mkv`) | `UnsupportedError` |
| Codec | HEVC (H.265), 8-bit, 4:2:0, progressive | `UnsupportedError` |
| GOP structure | every GOP starts with an IDR frame (closed GOPs, B-frames allowed); one frame per packet | `UnsupportedError` |
| Parameter sets | in the container's codec configuration; copies inside the stream must be identical | `UnsupportedError` |
| RGB output | full range (0–255), chroma sample location type 0, matrix BT.709, BT.601 or BT.2020 | `UnsupportedError`; `format="nv12"` still works |
| Sources | local files, S3-compatible storage (`s3://`, e.g. Cloudflare R2), Alibaba Cloud OSS (`oss://`) | `ValueError` for other schemes |

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
| `index` | Video index for reading by frame number (not implemented yet) | — |
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

## `Frame`

Each frame is a `vidlo.Frame` with these attributes:

| Attribute | Meaning |
| --- | --- |
| `image` | `numpy.uint8` array. `rgb24`: shape `(height, width, 3)`. `nv12`: shape `(height * 3 / 2, width)`, the luma rows followed by the interleaved chroma rows |
| `number` | Frame number in display order, starting at 0; frames that were not kept are counted too |
| `pts` | Presentation timestamp from the file, an integer in units of `reader.metadata.time_base` |
| `key_frame` | Whether the frame is a key frame (IDR) |
| `decode_status` | Decoder status of this frame; frames you receive always decoded successfully |
| `decode_order` | Position of this frame's packet in decoding order |
| `decode_start` | Decoding position from which this frame can be decoded correctly (its GOP's key frame) |
| `packet_start`, `packet_end` | Byte range of this frame's compressed data in the file (end exclusive), without the container's block header |
| `colorspace` | Matrix actually used for RGB (`None` for `nv12`) |
| `interpolation` | Interpolation actually used (`None` when the image was not resized, and for `nv12`) |

`decode_order`, `decode_start`, `packet_start` and `packet_end` are what a [video index](index-format.md#video-index) stores.

## `reader.metadata`

`reader.metadata` is a `vidlo.VideoMetadata` with information from the file header. It is available as soon as the reader is created, and still after it is closed.

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
| `ValueError` | Invalid arguments or combinations of arguments, including `storage_options` |
| `FileNotFoundError` | The local file or the remote object does not exist |

After an error (other than `TruncatedError` at the end) the reader delivers no more frames.

## Processes and threads

- Use a `VideoReader` from one thread at a time. Several readers can be open at once; each has its own decoder.
- `import vidlo` does not initialize CUDA, so worker processes started after the import can use vidlo.
- CUDA does not work in a process forked after its parent used CUDA. In such a child, opening a video or using a reader inherited from the parent raises `vidlo.Error` that tells you to use the `spawn` or `forkserver` start method, for example `DataLoader(..., multiprocessing_context="spawn")`. If the parent used CUDA through another library, vidlo cannot tell beforehand; CUDA initialization then fails in the child with the same advice.
- Pass images between processes as CPU arrays; GPU memory is never shared.

## Diagnostics

- `reader.stats`: packets, decoder calls, frames shown and kept, decode status counts, bytes read, seeks, where a truncated file ends, and for remote sources the number of requests and retries.
- `reader.decoder_config`: driver version, GPU name and UUID, and the parameters passed to NVDEC together with what it reported (available after the first frame).

## Not implemented yet

These are part of the interface and raise `vidlo.UnsupportedError` for now:

- `VideoReader(..., index=...)` and `reader.get_frames_at(frame_numbers)`: read given frames using a video index, in the given order, repeats allowed.
- `reader.submit(frame_numbers)`: the same, asynchronously.
- `vidlo.read_sample(sample_index, video_index, root, ...)`: read the frames listed in a sample index.
- `len(reader)` and `reader[i]` will follow together with reading by frame number.
