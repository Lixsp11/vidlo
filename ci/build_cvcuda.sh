#!/bin/bash
# 从源码编 CV-CUDA 0.19.0，只编 vidlo 用到的 Resize 与 AdvCvtColor（CV_CUDA_SRC_PATERN 是 CV-CUDA 自带的筛选变量，
# 按文件名匹配算子），装到 <安装目录> 的 include/ 与 lib/（Ubuntu 上是 lib/x86_64-linux-gnu，CV-CUDA 自己定的）。
# 完整的预编译库有 237 MB，只编这两个约 8 MB。
# GPU 架构用 CV-CUDA 的默认策略，CUDA 12.8 下是 sm_75/80/86/89/90/100/120。
# 用法：build_cvcuda.sh <安装目录> <工作目录>；nvcc 要在 PATH 里，主机编译器用 PATH 里的 gcc、g++。
set -euo pipefail

PREFIX=$1
WORK=$2
source "$(dirname "$0")/common.sh"
mkdir -p "$WORK/src"

# GitHub 的 tag 源码包，校验值与 conda-forge 的 libcvcuda 配方一致
download https://github.com/CVCUDA/CV-CUDA/archive/refs/tags/v0.19.0.tar.gz "$WORK/cvcuda.tar.gz" \
  8659124529341bdffeb11d88c2ed189a68d5817dd3f77a35248824dd29bd2fed
tar xzf "$WORK/cvcuda.tar.gz" -C "$WORK/src" --strip-components=1

# 源码包不含 Git LFS 文件，兼容老 glibc 用的占位库只是指针文本，所以关掉 ENABLE_COMPAT_OLD_GLIBC；
# 发布的 wheel 在 manylinux_2_28 容器里编，本来就按 glibc 2.28
CXX=$(command -v g++)
cmake -S "$WORK/src" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=OFF -DBUILD_TESTS=OFF \
  -DENABLE_COMPAT_OLD_GLIBC=OFF '-DCV_CUDA_SRC_PATERN=^opresize\.;^opadvcvtcolor\.' \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER="$(command -v gcc)" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CUDA_HOST_COMPILER="$CXX"
cmake --build "$WORK/build" -j"$(nproc)"
cmake --install "$WORK/build"
