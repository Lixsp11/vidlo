#include "postprocess.h"

#include <cvcuda/OpAdvCvtColor.hpp>
#include <cvcuda/OpResize.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <cstring>
#include <utility>

#include "error.h"

namespace vidlo {
namespace {

// 包装一块紧凑排列的显存：[1, 高, 宽, 通道]，行宽 = 宽 × 通道
nvcv::Tensor wrap(CUdeviceptr data, int height, int width, int channels) {
  nvcv::TensorDataStridedCuda::Buffer buffer{};
  buffer.strides[3] = 1;
  buffer.strides[2] = channels;
  buffer.strides[1] = static_cast<int64_t>(width) * channels;
  buffer.strides[0] = buffer.strides[1] * height;
  buffer.basePtr = reinterpret_cast<NVCVByte*>(data);
  const nvcv::TensorShape shape({1, height, width, channels}, nvcv::TENSOR_NHWC);
  return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(shape, nvcv::TYPE_U8, buffer));
}

NVCVColorSpec color_spec(const std::string& colorspace) {
  // AdvCvtColor 只做 full range，基础规格与 _ER 规格结果相同（OpAdvCvtColor.h）；这里写 _ER 表明按 full range
  if (colorspace == "bt601") {
    return NVCV_COLOR_SPEC_BT601_ER;
  }
  if (colorspace == "bt709") {
    return NVCV_COLOR_SPEC_BT709_ER;
  }
  if (colorspace == "bt2020") {
    return NVCV_COLOR_SPEC_BT2020_ER;
  }
  throw Error("unknown colorspace '" + colorspace + "'");
}

// nearest_exact：NVCV_INTERP_NEAREST 取半像素中心 floor((i + 0.5) × 源长 / 目标长)，同 OpenCV 的 INTER_NEAREST_EXACT
NVCVInterpolationType interp_type(const std::string& method) {
  if (method == "nearest_exact") {
    return NVCV_INTERP_NEAREST;
  }
  if (method == "linear") {
    return NVCV_INTERP_LINEAR;
  }
  if (method == "cubic") {
    return NVCV_INTERP_CUBIC;
  }
  if (method == "area") {
    return NVCV_INTERP_AREA;
  }
  throw Error("unknown interpolation '" + method + "'");
}

CUdeviceptr allocate(size_t bytes) {
  CUdeviceptr ptr = 0;
  check_cu(cuMemAlloc(&ptr, bytes), "cuMemAlloc");
  return ptr;
}

}  // namespace

struct FrameProcessor::Gpu {
  cvcuda::AdvCvtColor convert;
  cvcuda::Resize resize;
  nvcv::Tensor nv12;
  nvcv::Tensor rgb;
  nvcv::Tensor resized;
  NVCVColorSpec spec = NVCV_COLOR_SPEC_BT601_ER;
  NVCVInterpolationType method = NVCV_INTERP_AREA;
};

FrameProcessor::FrameProcessor(OutputSpec spec) : spec_(std::move(spec)) {}

FrameProcessor::~FrameProcessor() {
  gpu_.reset();
  for (CUdeviceptr ptr : {nv12_, rgb_, resized_}) {
    if (ptr != 0) {
      cuMemFree(ptr);
    }
  }
  if (pinned_ != nullptr) {
    cuMemFreeHost(pinned_);
  }
}

void FrameProcessor::configure(const CUVIDEOFORMAT& format) {
  const auto& area = format.display_area;
  const int display_width = area.right - area.left;
  const int display_height = area.bottom - area.top;
  crop_width_ = spec_.crop_width == 0 ? display_width : spec_.crop_width;
  crop_height_ = spec_.crop_width == 0 ? display_height : spec_.crop_height;
  if (spec_.crop_x < 0 || spec_.crop_y < 0 || crop_width_ <= 0 || crop_height_ <= 0 ||
      spec_.crop_x + crop_width_ > display_width || spec_.crop_y + crop_height_ > display_height) {
    throw Error("crop (" + std::to_string(spec_.crop_x) + ", " + std::to_string(spec_.crop_y) + ", " +
                std::to_string(crop_width_) + ", " + std::to_string(crop_height_) + ") does not fit the " +
                std::to_string(display_width) + "x" + std::to_string(display_height) + " picture");
  }
  // 4:2:0 每 2×2 像素共用一份色度：裁剪的起点和尺寸都要是偶数（显示区域的边界在建解码器时已检查为偶数）
  if ((spec_.crop_x | spec_.crop_y | crop_width_ | crop_height_) & 1) {
    throw Error("crop values must be even");
  }
  left_ = area.left + spec_.crop_x;
  top_ = area.top + spec_.crop_y;
  output_width_ = spec_.width == 0 ? crop_width_ : spec_.width;
  output_height_ = spec_.width == 0 ? crop_height_ : spec_.height;
  const bool scaled = output_width_ != crop_width_ || output_height_ != crop_height_;
  if (scaled && !spec_.rgb) {
    throw Error("raw NV12 output cannot be resized");
  }
  if (scaled) {
    // 插值方式由 Python 层按缩放比例选好并检查过（_reader._interpolation）
    if (spec_.interpolation.empty()) {
      throw Error("interpolation was not chosen");
    }
    interpolation_ = spec_.interpolation;
  }

  const size_t pixels = static_cast<size_t>(crop_width_) * crop_height_;
  nv12_ = allocate(pixels * 3 / 2);
  const size_t output_bytes = spec_.rgb ? static_cast<size_t>(output_width_) * output_height_ * 3 : pixels * 3 / 2;
  check_cu(cuMemHostAlloc(&pinned_, output_bytes, 0), "cuMemHostAlloc");
  if (!spec_.rgb) {
    return;
  }
  try {
    gpu_ = std::make_unique<Gpu>();
    gpu_->spec = color_spec(spec_.colorspace);
    // AdvCvtColor 的 NV12 输入是一个单通道 tensor：先是亮度各行，紧接着是色度各行（OpAdvCvtColor.cu 按行号取色度）
    gpu_->nv12 = wrap(nv12_, crop_height_ * 3 / 2, crop_width_, 1);
    rgb_ = allocate(pixels * 3);
    gpu_->rgb = wrap(rgb_, crop_height_, crop_width_, 3);
    if (scaled) {
      gpu_->method = interp_type(interpolation_);
      resized_ = allocate(static_cast<size_t>(output_width_) * output_height_ * 3);
      gpu_->resized = wrap(resized_, output_height_, output_width_, 3);
    }
  } catch (const nvcv::Exception& e) {
    throw Error(std::string("CV-CUDA setup failed: ") + e.what());
  }
}

void FrameProcessor::copy_in(CUdeviceptr frame, unsigned pitch, unsigned surface_height, CUstream stream) {
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = frame + static_cast<size_t>(top_) * pitch + static_cast<size_t>(left_);
  copy.srcPitch = pitch;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = nv12_;
  copy.dstPitch = static_cast<size_t>(crop_width_);
  copy.WidthInBytes = static_cast<size_t>(crop_width_);
  copy.Height = static_cast<size_t>(crop_height_);
  check_cu(cuMemcpy2DAsync(&copy, stream), "copy luma");
  // 色度紧跟在整个 surface 的亮度之后，每行是 U、V 交错，起点的列号与亮度相同
  copy.srcDevice = frame + static_cast<size_t>(surface_height) * pitch + static_cast<size_t>(top_ / 2) * pitch +
                   static_cast<size_t>(left_);
  copy.dstDevice = nv12_ + static_cast<size_t>(crop_width_) * crop_height_;
  copy.Height = static_cast<size_t>(crop_height_ / 2);
  check_cu(cuMemcpy2DAsync(&copy, stream), "copy chroma");
}

size_t FrameProcessor::output_bytes() const {
  return spec_.rgb ? static_cast<size_t>(output_width_) * output_height_ * 3 : static_cast<size_t>(crop_width_) * crop_height_ * 3 / 2;
}

std::vector<uint8_t> FrameProcessor::produce(CUstream stream) {
  const auto* data = static_cast<const uint8_t*>(to_host(stream));
  return std::vector<uint8_t>(data, data + output_bytes());
}

void FrameProcessor::produce(CUstream stream, uint8_t* out) { std::memcpy(out, to_host(stream), output_bytes()); }

const void* FrameProcessor::to_host(CUstream stream) {
  CUdeviceptr source = nv12_;
  if (spec_.rgb) {
    try {
      gpu_->convert(stream, gpu_->nv12, gpu_->rgb, NVCV_COLOR_YUV2RGB_NV12, nvcv::ColorSpec(gpu_->spec));
      source = rgb_;
      if (resized_ != 0) {
        gpu_->resize(stream, gpu_->rgb, gpu_->resized, gpu_->method);
        source = resized_;
      }
    } catch (const nvcv::Exception& e) {
      throw Error(std::string("CV-CUDA failed: ") + e.what());
    }
  }
  // 复制到锁页内存才是真正异步的；复制到普通内存时驱动内部要等一次 GPU。一帧只在这里同步一次：
  // 多个进程共用一张卡时，每次同步都要排队等 GPU 轮到本进程的 context
  check_cu(cuMemcpyDtoHAsync(pinned_, source, output_bytes(), stream), "copy to host");
  check_cu(cuStreamSynchronize(stream), "cuStreamSynchronize");
  return pinned_;
}

}  // namespace vidlo
