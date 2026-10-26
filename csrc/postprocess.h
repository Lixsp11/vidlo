#pragma once

#include <cuda.h>
#include <nvcuvid.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vidlo {

// 交给调用方的输出是什么样。参数在 Python 层检查过，这里只按显示区域再核对一次裁剪范围
struct OutputSpec {
  bool rgb = false;  // false：显示区域的原生 NV12，不裁剪、不缩放
  // 裁剪区域，坐标相对于显示区域；crop_width 为 0 表示整个显示区域
  int crop_x = 0;
  int crop_y = 0;
  int crop_width = 0;
  int crop_height = 0;
  int width = 0;  // 输出尺寸；为 0 表示不缩放
  int height = 0;
  std::string colorspace = "bt601";  // 输出 RGB 时用的矩阵："bt601"、"bt709"、"bt2020"，都按 full range
  std::string interpolation;         // 缩放时的插值："nearest_exact"、"linear"、"cubic"、"area"，由 Python 层按比例选好
};

// 把解码出的一帧变成交给调用方的输出：先把裁剪区域复制到自有的紧凑 NV12 显存（之后就能归还 surface），
// 输出 RGB 时再用 CV-CUDA 转颜色、缩放，最后复制到内存。显存在 configure 时分配一次，每帧复用。
// 所有调用（包括析构）都要在当前线程持有解码用的 CUDA context 时进行。
class FrameProcessor {
 public:
  explicit FrameProcessor(OutputSpec spec);
  ~FrameProcessor();
  FrameProcessor(const FrameProcessor&) = delete;
  FrameProcessor& operator=(const FrameProcessor&) = delete;

  // 建好解码器后调用一次：按 sequence 回调的显示区域确定裁剪位置、选插值方式、分配显存
  void configure(const CUVIDEOFORMAT& format);
  // surface 映射期间调用：复制裁剪区域，等复制完成才返回。surface_height 是 surface 里亮度的行数（色度紧随其后）
  void copy_in(CUdeviceptr frame, unsigned pitch, unsigned surface_height, CUstream stream);
  // 颜色转换、缩放，复制到内存后返回。RGB 为 [高, 宽, 3]；NV12 为 [高×3/2, 宽]
  std::vector<uint8_t> produce(CUstream stream);

  int output_width() const { return output_width_; }
  int output_height() const { return output_height_; }
  bool rgb() const { return spec_.rgb; }
  const std::string& colorspace() const { return spec_.colorspace; }
  const std::string& interpolation() const { return interpolation_; }  // 实际用的插值；不缩放时为空

 private:
  struct Gpu;  // CV-CUDA 的算子与包装好的 tensor，只在 postprocess.cpp 里用到它们的头文件

  OutputSpec spec_;
  int left_ = 0;  // 裁剪区域在 surface 里的位置
  int top_ = 0;
  int crop_width_ = 0;
  int crop_height_ = 0;
  int output_width_ = 0;
  int output_height_ = 0;
  std::string interpolation_;
  CUdeviceptr nv12_ = 0;     // 裁剪后的紧凑 NV12：crop_height 行亮度后紧跟 crop_height/2 行交错色度，行宽 crop_width
  CUdeviceptr rgb_ = 0;      // 转换后的 RGB，裁剪尺寸
  CUdeviceptr resized_ = 0;  // 缩放后的 RGB，输出尺寸
  std::unique_ptr<Gpu> gpu_;
};

}  // namespace vidlo
