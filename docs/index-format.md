# Index files

vidlo reads two kinds of index files but never creates or stores them; your data pipeline does. Both are JSON. To read with them, see [Reading by frame number](api.md#reading-by-frame-number); this release reads by frame number from local files only.

- A **video index** describes one video: where each frame's compressed data is and from where it can be decoded. It lets vidlo fetch a few frames with a single byte-range request instead of reading the whole file.
- A **sample index** lists the frames of one training sample.

## Video index

One file per video, for example `game.index.json`. The arrays below are shortened to their first 6 entries:

```json
{
  "version": 1,
  "file_size": 322673914,
  "codec": "hevc",
  "codec_config": "ASFgAAAAkAAAAAAAePAA…",
  "decode_order": [0, 3, 4, 2, 5, 1, …],
  "decode_start": [0, 0, 0, 0, 0, 0, …],
  "packet_start": [828, 291832, 292273, 290491, 293638, 285471, …],
  "packet_end":   [285464, 292266, 293283, 291465, 294412, 290484, …]
}
```

- Every array is in **display order**: entry `n` belongs to frame number `n` (starting at 0).
- A packet is one frame's compressed data; "packet `k`" is the k-th video packet in decoding order (starting at 0).
- Byte positions are absolute positions in the video file; ranges include the start and exclude the end.

| Field | Meaning |
| --- | --- |
| `version` | Format version, currently `1` |
| `file_size` | Size of the video file in bytes. Checked against the object size on every remote read, so an index that belongs to another file is rejected |
| `codec` | Codec name; `"hevc"` for now |
| `codec_config` | The container's codec configuration, base64-encoded (for HEVC in Matroska, the `hvcC` record). Frames in the middle of a video do not carry the parameter sets, so decoding from there needs them |
| `decode_order` | Position of this frame's packet in decoding order |
| `decode_start` | Packet from which decoding must start to get this frame right (usually the key frame that starts its GOP) |
| `packet_start`, `packet_end` | Byte range of this frame's compressed data, without the container's block header |

To read frames `f1, f2, …`, vidlo decodes packets `min(decode_start)` to `max(decode_order)` and fetches the bytes from that first packet's `packet_start` to the last packet's `packet_end`.

### Checks

`vidlo.load_video_index(path_or_dict)` and `vidlo.validate_video_index(dict)` raise `vidlo.Error` unless:

- exactly these fields are present and `version` is known;
- all arrays have the same, non-zero length;
- `decode_order` contains each of `0 … frames-1` exactly once;
- each frame's `decode_start` is at most its `decode_order`, and points at a packet that starts a GOP (the frame of that packet has it as its own `decode_start`);
- in decoding order, packet byte ranges are non-empty, increasing, do not overlap and end within `file_size`;
- `codec_config` is valid, non-empty base64.

An unsupported `codec` raises `vidlo.UnsupportedError`.

### Building a video index

A sequential read gives every field. Read with `format="nv12"` to skip color conversion:

```python
import base64
import json

import vidlo

FIELDS = ("decode_order", "decode_start", "packet_start", "packet_end")

arrays = {name: [] for name in FIELDS}
with vidlo.VideoReader("game.mkv", format="nv12") as reader:
    for frame in reader:  # display order, frame.number == 0, 1, 2, ...
        for name in FIELDS:
            arrays[name].append(getattr(frame, name))
    metadata = reader.metadata

index = {
    "version": 1,
    "file_size": metadata.file_size,
    "codec": metadata.codec,
    "codec_config": base64.b64encode(metadata.codec_config).decode("ascii"),
    **arrays,
}
vidlo.validate_video_index(index)
with open("game.index.json", "x") as f:
    json.dump(index, f)
```

Every byte range comes from the file itself and is compared byte by byte with the demuxed packets during the read.

## Sample index

One file per sample, for example `fight-03-a.json`:

```json
{
  "video": "machine-01/20261022030227/game.mkv",
  "frames": [3000, 3001, 3002]
}
```

| Field | Meaning |
| --- | --- |
| `video` | The sample's video, as a path relative to a data root that you pass when reading (POSIX separators; not absolute; no empty, `.` or `..` parts) |
| `frames` | Frame numbers in display order (starting at 0); any order, repeats and gaps are allowed |

`vidlo.load_sample_index(path_or_dict)` and `vidlo.validate_sample_index(dict)` check the fields and raise `vidlo.Error` for a bad path or an empty or negative frame list. Whether the frame numbers exist in the video is checked when reading.

`vidlo.read_sample` reads a sample (see [`vidlo.read_sample`](api.md#vidloread_sample)). To read it at a lower frame rate, select from its frame list with `step` (every k-th entry) or `fps` and `source_fps`; see [Picking frames](api.md#picking-frames).
