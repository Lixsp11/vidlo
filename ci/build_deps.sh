#!/bin/bash
# cibuildwheel 的 before-all：在 manylinux_2_28 容器（AlmaLinux 8）里准备构建 vidlo 需要的依赖，都装到 /opt 下。
# 版本与本地开发构建相同：CUDA 12.8、FFmpeg 7.1.5 最小 LGPL 构建、CV-CUDA 0.19.0、Video Codec SDK 13.0 的头文件。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=/tmp/vidlo-deps
mkdir -p "$WORK"

download() {  # url 文件 sha256
  curl -fsSL --retry 3 -o "$2" "$1"
  echo "$3  $2" | sha256sum -c -
}

# CUDA 12.8：cuda.h 与 libcuda 的链接用占位库、cuda_runtime.h（CV-CUDA 的头文件引用它），以及 CMake 判断 CUDA 版本用的 nvcc
dnf install -y dnf-plugins-core
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y cuda-nvcc-12-8 cuda-cudart-devel-12-8 cuda-driver-devel-12-8

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

# CV-CUDA 0.19.0（CUDA 12）：解压到根目录，库落在 /opt/nvidia/cvcuda0/lib，正是 libcvcuda 自带 RPATH 指向的位置，
# auditwheel 才找得到它依赖的 libnvcv_types
CVCUDA=https://github.com/CVCUDA/CV-CUDA/releases/download/v0.19.0
download "$CVCUDA/cvcuda-lib-0.19.0-cuda12-x86_64-linux.tar.xz" "$WORK/cvcuda-lib.tar.xz" \
  939d80e14c1ed48e5634524aa01bae398df1937b7270317abb7a55c9cb7f3146
download "$CVCUDA/cvcuda-dev-0.19.0-cuda12-x86_64-linux.tar.xz" "$WORK/cvcuda-dev.tar.xz" \
  abb71db72a908930a92ab6c60936621365380d3c255f84a53bd2224335d9784b
tar xJf "$WORK/cvcuda-lib.tar.xz" -C /
tar xJf "$WORK/cvcuda-dev.tar.xz" -C /

# Video Codec SDK：仓库里只带两个 MIT 许可的头文件；链接用的 libnvcuvid 占位库在这里编，目录结构同 SDK
mkdir -p /opt/nvsdk/Interface /opt/nvsdk/Lib/linux/stubs/x86_64
cp "$ROOT"/third_party/nvcodec/Interface/*.h /opt/nvsdk/Interface/
gcc -shared -fPIC -Wl,-soname,libnvcuvid.so.1 -o /opt/nvsdk/Lib/linux/stubs/x86_64/libnvcuvid.so "$ROOT/third_party/nvcodec/nvcuvid_stub.c"

# 测试导入用：构建机没有 GPU 驱动，用两个占位库让扩展模块能加载（导入时不调用 CUDA）
mkdir -p /opt/stubs
cp /usr/local/cuda-12.8/lib64/stubs/libcuda.so /opt/stubs/libcuda.so.1
cp /opt/nvsdk/Lib/linux/stubs/x86_64/libnvcuvid.so /opt/stubs/libnvcuvid.so.1
