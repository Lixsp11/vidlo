<p align="center">
  <img src="docs/assets/vidlo.png" width="128" alt="vidlo">
</p>

<h1 align="center">vidlo</h1>

<p align="center">
  Read video frames on the GPU: decode with NVDEC, crop, convert and resize with CV-CUDA,<br>
  from local files, S3-compatible storage or Alibaba Cloud OSS.
</p>

<p align="center">
  <img alt="Python 3.11–3.14" src="https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB?logo=python&logoColor=white">
  <img alt="C++17" src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white">
  <img alt="CUDA 12.8" src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white">
  <img alt="NVDEC" src="https://img.shields.io/badge/NVDEC-Video%20Codec%20SDK%2013.0-76B900?logo=nvidia&logoColor=white">
  <img alt="CV-CUDA 0.19" src="https://img.shields.io/badge/CV--CUDA-0.19-76B900?logo=nvidia&logoColor=white">
  <img alt="FFmpeg 7.1" src="https://img.shields.io/badge/FFmpeg-7.1-007808?logo=ffmpeg&logoColor=white">
  <img alt="Linux x86-64" src="https://img.shields.io/badge/Linux-x86--64-FCC624?logo=linux&logoColor=black">
</p>

## Features

- **GPU end to end**: NVDEC decodes, CV-CUDA crops, converts YUV to RGB and resizes; only the final image is copied to CPU memory.
- **Checked decoding**: every frame's decode status is checked, frames are matched to their packets and timestamps, and colors follow the matrix tagged in the video. Output is tested frame by frame against FFmpeg.
- **Object storage**: one streaming read from S3-compatible storage (Cloudflare R2) or Alibaba Cloud OSS, resumed after network errors while making sure the object did not change.
- **Frame selection**: every k-th frame, or frame-rate conversion compatible with VPF's `EMDownSampler`; frames that are not kept are never converted or copied.
- **Index-ready**: each frame reports the byte range of its packet and where decoding must start, so a pipeline can write [video indexes](docs/index-format.md) in one pass.
- **Reading by frame number**: with a video index, one byte-range read per request, decoding starts at the right key frame and skips frames nothing refers to; hardware decoders are reused across requests and files.

## Install

Wheels for Python 3.11–3.14 (Linux x86-64, manylinux_2_28) are attached to the [releases](https://github.com/Lixsp11/vidlo/releases).

```bash
pip install vidlo-0.1-cp312-cp312-manylinux_2_28_x86_64.whl            # local files
pip install "vidlo-0.1-cp312-cp312-manylinux_2_28_x86_64.whl[s3,oss]"  # plus R2/S3 and OSS
```

Needs an NVIDIA GPU from Turing on (the bundled CV-CUDA is built for sm_75, 80, 86, 89, 90, 100 and 120) with driver 570 or newer. In containers, `NVIDIA_DRIVER_CAPABILITIES` must include `video`.

## Usage

```python
import vidlo

with vidlo.VideoReader("game.mkv", crop=(32, 0, 1248, 720), width=832, height=480) as reader:
    for frame in reader.frames(step=2):
        frame.image   # numpy uint8, (480, 832, 3), RGB
        frame.pts     # timestamp in units of reader.metadata.time_base

with vidlo.VideoReader("game.mkv", index="game.index.json", width=832, height=480) as reader:
    batch = reader.get_frames_at([3000, 3002, 3004])
    batch.images      # numpy uint8, (3, 480, 832, 3)
```

See [docs/api.md](docs/api.md) for every option and [docs/index-format.md](docs/index-format.md) for index files.

## Built with

| | |
| --- | --- |
| [NVIDIA Video Codec SDK 13.0](https://developer.nvidia.com/video-codec-sdk) (NVDEC) | hardware HEVC decoding |
| [CV-CUDA 0.19](https://github.com/CVCUDA/CV-CUDA) | crop, color conversion and resize on the GPU; built from source with only the operators vidlo uses |
| [FFmpeg 7.1](https://ffmpeg.org/) (libavformat, LGPL build) | demuxing |
| [pybind11](https://github.com/pybind/pybind11) · [scikit-build-core](https://github.com/scikit-build/scikit-build-core) | Python bindings and build |
| [boto3](https://github.com/boto/boto3) · [alibabacloud-oss-v2](https://github.com/aliyun/alibabacloud-oss-python-sdk-v2) | object storage |
| [cibuildwheel](https://github.com/pypa/cibuildwheel) · [auditwheel](https://github.com/pypa/auditwheel) | release wheels |
