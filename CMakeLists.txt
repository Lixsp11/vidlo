cmake_minimum_required(VERSION 3.22)
project(vidlo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(VIDLO_FFMPEG_ROOT "" CACHE PATH "FFmpeg 安装目录，含 include/ 与 lib/")
set(VIDLO_NVSDK_ROOT "" CACHE PATH "Video Codec SDK 根目录，含 Interface/ 与 Lib/")
set(VIDLO_NVSDK_VERSION "13.0.37" CACHE STRING "Video Codec SDK 版本，记进解码配置快照")
set(VIDLO_CVCUDA_ROOT "" CACHE PATH "ci/build_cvcuda.sh 编出的 CV-CUDA 0.19.0 安装目录，含 include/ 与 lib/")
if(NOT VIDLO_FFMPEG_ROOT OR NOT VIDLO_NVSDK_ROOT OR NOT VIDLO_CVCUDA_ROOT)
  message(FATAL_ERROR "VIDLO_FFMPEG_ROOT, VIDLO_NVSDK_ROOT and VIDLO_CVCUDA_ROOT must be set")
endif()

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit 12.2 REQUIRED)

foreach(lib avformat avcodec avutil)
  find_library(VIDLO_LIB_${lib} ${lib} PATHS "${VIDLO_FFMPEG_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
endforeach()
# CV-CUDA 把库装在 lib/${CMAKE_LIBRARY_ARCHITECTURE}：Ubuntu 上是 lib/x86_64-linux-gnu，manylinux 容器里就是 lib
foreach(lib cvcuda nvcv_types)
  find_library(VIDLO_LIB_${lib} ${lib} PATHS "${VIDLO_CVCUDA_ROOT}/lib" PATH_SUFFIXES "${CMAKE_LIBRARY_ARCHITECTURE}"
    NO_DEFAULT_PATH REQUIRED)
endforeach()
get_filename_component(VIDLO_CVCUDA_LIBDIR "${VIDLO_LIB_cvcuda}" DIRECTORY)
# 链接用 SDK 的 stub，运行时加载驱动自带的 libnvcuvid.so.1
find_library(VIDLO_LIB_nvcuvid nvcuvid PATHS "${VIDLO_NVSDK_ROOT}/Lib/linux/stubs/x86_64" NO_DEFAULT_PATH REQUIRED)

pybind11_add_module(_native csrc/module.cpp csrc/demuxer.cpp csrc/input.cpp csrc/mkv_scanner.cpp csrc/nvdec.cpp
  csrc/postprocess.cpp)
# CV-CUDA 的头文件引用 cuda_runtime.h；它的库自带静态 CUDA 运行库，本模块不链接 libcudart
target_include_directories(_native PRIVATE "${VIDLO_FFMPEG_ROOT}/include" "${VIDLO_NVSDK_ROOT}/Interface"
  "${VIDLO_CVCUDA_ROOT}/include" ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(_native PRIVATE VIDLO_NVSDK_VERSION="${VIDLO_NVSDK_VERSION}")
target_compile_options(_native PRIVATE -Wall -Wextra)
# 开发构建：FFmpeg 与 CV-CUDA 的库都写进 NEEDED，由本模块的 RUNPATH 找到；libavformat 自己没有搜索路径，
# 若 --as-needed 去掉了 libavutil，它的依赖会找不到。发布的 wheel 由 auditwheel 把这些库改名后拷进包里、
# 改用 $ORIGIN 查找（pyproject.toml 的 [tool.cibuildwheel]）。
target_link_libraries(_native PRIVATE CUDA::cuda_driver "${VIDLO_LIB_nvcuvid}"
  -Wl,--no-as-needed "${VIDLO_LIB_avformat}" "${VIDLO_LIB_avcodec}" "${VIDLO_LIB_avutil}"
  "${VIDLO_LIB_cvcuda}" "${VIDLO_LIB_nvcv_types}" -Wl,--as-needed)
set_target_properties(_native PROPERTIES
  BUILD_RPATH "${VIDLO_FFMPEG_ROOT}/lib;${VIDLO_CVCUDA_LIBDIR}"
  INSTALL_RPATH "${VIDLO_FFMPEG_ROOT}/lib;${VIDLO_CVCUDA_LIBDIR}")

install(TARGETS _native DESTINATION vidlo)
