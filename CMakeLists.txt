cmake_minimum_required(VERSION 3.22)
project(vidlo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(VIDLO_FFMPEG_ROOT "" CACHE PATH "FFmpeg 安装目录，含 include/ 与 lib/")
set(VIDLO_NVSDK_ROOT "" CACHE PATH "Video Codec SDK 根目录，含 Interface/ 与 Lib/")
set(VIDLO_NVSDK_VERSION "13.0.37" CACHE STRING "Video Codec SDK 版本，记进解码配置快照")
set(VIDLO_CVCUDA_ROOT "" CACHE PATH "CV-CUDA 0.19.0 开发包的解压目录，用其中的 include/")
set(VIDLO_CVCUDA_LIBS "" CACHE PATH "cvcuda-cu12 0.19.0 的 cvcuda_cu12.libs 目录，运行时也用这个包的库")
if(NOT VIDLO_FFMPEG_ROOT OR NOT VIDLO_NVSDK_ROOT OR NOT VIDLO_CVCUDA_ROOT OR NOT VIDLO_CVCUDA_LIBS)
  message(FATAL_ERROR "VIDLO_FFMPEG_ROOT, VIDLO_NVSDK_ROOT, VIDLO_CVCUDA_ROOT and VIDLO_CVCUDA_LIBS must be set")
endif()

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit 12.2 REQUIRED)

foreach(lib avformat avcodec avutil)
  find_library(VIDLO_LIB_${lib} ${lib} PATHS "${VIDLO_FFMPEG_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
endforeach()
# cvcuda-cu12 的库名带 auditwheel 加的哈希（如 libcvcuda-4e720cbb.so.0.19.0），soname 也是这个名字
foreach(lib cvcuda nvcv_types)
  file(GLOB VIDLO_LIB_${lib} "${VIDLO_CVCUDA_LIBS}/lib${lib}-*.so.0.19.0")
  list(LENGTH VIDLO_LIB_${lib} count)
  if(NOT count EQUAL 1)
    message(FATAL_ERROR "expected one lib${lib}-*.so.0.19.0 in ${VIDLO_CVCUDA_LIBS}, found: ${VIDLO_LIB_${lib}}")
  endif()
endforeach()
# 链接用 SDK 的 stub，运行时加载驱动自带的 libnvcuvid.so.1
find_library(VIDLO_LIB_nvcuvid nvcuvid PATHS "${VIDLO_NVSDK_ROOT}/Lib/linux/stubs/x86_64" NO_DEFAULT_PATH REQUIRED)

pybind11_add_module(_native csrc/module.cpp csrc/demuxer.cpp csrc/input.cpp csrc/mkv_scanner.cpp csrc/nvdec.cpp
  csrc/postprocess.cpp)
# CV-CUDA 的头文件引用 cuda_runtime.h；它的库自带静态 CUDA 运行库，本模块不链接 libcudart
target_include_directories(_native PRIVATE "${VIDLO_FFMPEG_ROOT}/include" "${VIDLO_NVSDK_ROOT}/Interface"
  "${VIDLO_CVCUDA_ROOT}/include" ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(_native PRIVATE VIDLO_NVSDK_VERSION="${VIDLO_NVSDK_VERSION}")
target_compile_options(_native PRIVATE -Wall -Wextra)
# FFmpeg 的库都写进 NEEDED，由本模块的 RUNPATH 找到：libavformat 自己没有搜索路径，若 --as-needed 去掉了 libavutil，
# 它的依赖会找不到。发布的 wheel 由 auditwheel 把 FFmpeg 改名后拷进包里、改用 $ORIGIN 查找（pyproject.toml 的
# [tool.cibuildwheel]）。CV-CUDA 不打包，用同一个 site-packages 里依赖包 cvcuda-cu12 的库。
target_link_libraries(_native PRIVATE CUDA::cuda_driver "${VIDLO_LIB_nvcuvid}"
  -Wl,--no-as-needed "${VIDLO_LIB_avformat}" "${VIDLO_LIB_avcodec}" "${VIDLO_LIB_avutil}"
  "${VIDLO_LIB_cvcuda}" "${VIDLO_LIB_nvcv_types}" -Wl,--as-needed)
set_target_properties(_native PROPERTIES
  BUILD_RPATH "${VIDLO_FFMPEG_ROOT}/lib;${VIDLO_CVCUDA_LIBS}"
  INSTALL_RPATH "${VIDLO_FFMPEG_ROOT}/lib;$ORIGIN/../cvcuda_cu12.libs")

install(TARGETS _native DESTINATION vidlo)
