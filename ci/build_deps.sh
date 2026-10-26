#!/bin/bash
# cibuildwheel 的 before-all：在 manylinux_2_28 容器（AlmaLinux 8）里准备构建 vidlo 需要的依赖，都装到 /opt 下。
# 版本与本地开发构建相同：CUDA 12.8、FFmpeg 7.1.5 最小 LGPL 构建、CV-CUDA 0.19.0（ci/build_cvcuda.sh）、Video Codec SDK 13.0 的头文件。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=/tmp/vidlo-deps
mkdir -p "$WORK"

source "$ROOT/ci/common.sh"

# CUDA 12.8：编 CV-CUDA 用的 nvcc、静态 cudart 与 NVTX 头文件，cuda.h 与 libcuda 的链接用占位库。CV-CUDA 无条件链接
# cusolver、cublas 的静态库（CMake 目标不存在就报错），只编两个算子时用不到其中的符号，不会链进产物
dnf install -y dnf-plugins-core
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y cuda-nvcc-12-8 cuda-cudart-devel-12-8 cuda-driver-devel-12-8 cuda-nvtx-12-8 libcusolver-devel-12-8 libcublas-devel-12-8
export PATH=/usr/local/cuda-12.8/bin:$PATH

# FFmpeg 7.1.5：只用于解封装，参数与 DSW 上验证过的最小 LGPL 构建相同
download https://ffmpeg.org/releases/ffmpeg-7.1.5.tar.xz "$WORK/ffmpeg.tar.xz" \
  de668509caf9e35e3cd162473441fdb29538c6d96ed080292b3cf9e6fc5d558f
tar xJf "$WORK/ffmpeg.tar.xz" -C "$WORK"
(
  cd "$WORK/ffmpeg-7.1.5"
  ./configure --prefix=/opt/ffmpeg --enable-shared --disable-static --enable-pic --disable-programs --disable-doc \
    --disable-autodetect --disable-x86asm --disable-everything --disable-avdevice --disable-avfilter --disable-swscale \
    --disable-swresample --disable-network --enable-demuxer=matroska,mov --enable-parser=hevc,h264 \
    --enable-bsf=hevc_mp4toannexb,h264_mp4toannexb --enable-protocol=file
  make -j"$(nproc)"
  make install
)

bash "$ROOT/ci/build_cvcuda.sh" /opt/cvcuda "$WORK/cvcuda"

# Video Codec SDK：仓库里只带两个 MIT 许可的头文件；链接用的 libnvcuvid 占位库在这里编，目录结构同 SDK
mkdir -p /opt/nvsdk/Interface /opt/nvsdk/Lib/linux/stubs/x86_64
cp "$ROOT"/third_party/nvcodec/Interface/*.h /opt/nvsdk/Interface/
gcc -shared -fPIC -Wl,-soname,libnvcuvid.so.1 -o /opt/nvsdk/Lib/linux/stubs/x86_64/libnvcuvid.so "$ROOT/third_party/nvcodec/nvcuvid_stub.c"

# 测试导入用：构建机没有 GPU 驱动，用两个占位库让扩展模块能加载（导入时不调用 CUDA）
mkdir -p /opt/stubs
cp /usr/local/cuda-12.8/lib64/stubs/libcuda.so /opt/stubs/libcuda.so.1
cp /opt/nvsdk/Lib/linux/stubs/x86_64/libnvcuvid.so /opt/stubs/libnvcuvid.so.1
