#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/bsf.h>
}

namespace vidlo {

// hvcC（ISO/IEC 14496-15 的 HEVCDecoderConfigurationRecord）里解码要用的两样：NAL 长度字段的字节数，以及 VPS、SPS、PPS
struct HvcC {
  int nal_length_size = 0;
  std::vector<std::vector<uint8_t>> parameter_sets;  // 含 NAL 头
  int max_temporal_id = 0;  // SPS 的 sps_max_sub_layers_minus1，即码流里最高的 TemporalId
};

HvcC parse_hvcc(const std::vector<uint8_t>& config);

// 一张图像第一个 slice 的 NAL 类型与 TemporalId（同一图像各 slice 的类型相同）
struct AccessUnit {
  int nal_type = -1;
  int temporal_id = 0;
  // 包里有影响后续解码的非 VCL NAL：序列结束、码流结束（重置 POC 等），以及保留、未定义的类型
  bool stateful_nal = false;
};

// 这张图像能不能不解：sub-layer non-reference（TRAIL_N、TSA_N、STSA_N、RADL_N、RASL_N 等偶数类型，H.265 表 7-1 与
// 7.4.2.2）不会被同一时间层后面的图像参考，低层图像也不参考高层图像（8.3.2），在最高时间层上就没有任何图像参考它；
// 它又不是要交出的帧时，跳过不影响其他帧的解码结果（同 FFmpeg 的 skip_frame=nonref）。整包跳过，所以包里带着
// stateful_nal 时不跳
bool unreferenced(const AccessUnit& unit, const HvcC& hvcc);

// 第一期要求一个包正好是一张完整的图像：按 hvcC 的 NAL 长度字段切开负载，长度之和必须等于包的大小；
// 恰好有一个 VCL NAL 是图像的第一个 slice（first_slice_segment_in_pic_flag 为 1），且它出现在其他 VCL NAL 之前。
// 包里的参数集必须与 codec_config 里的某一个逐字节相同：按帧号读时只送 codec_config 的参数集，
// 码流中途换参数集会让两种读法解出不同的画面。where 附在报错信息后面
AccessUnit check_access_unit(const uint8_t* data, size_t size, const HvcC& hvcc, const std::string& where);

// 按帧号读时用的 hevc_mp4toannexb：直接用索引里的 hvcC 初始化，不经 libavformat。把“长度 + 数据”的 NAL 换成起始码，
// 并在每个 IRAP 包前加上 hvcC 里的参数集。一进一出，filter() 返回的数据在下一次调用之前有效
class AnnexbFilter {
 public:
  explicit AnnexbFilter(const std::vector<uint8_t>& codec_config);
  ~AnnexbFilter();
  AnnexbFilter(const AnnexbFilter&) = delete;
  AnnexbFilter& operator=(const AnnexbFilter&) = delete;

  void filter(const uint8_t* data, size_t size, const uint8_t*& out, size_t& out_size);
  void finish();  // 冲刷，确认没有剩下的包

 private:
  AVBSFContext* bsf_ = nullptr;
  AVPacket* in_ = nullptr;
  AVPacket* out_ = nullptr;
};

}  // namespace vidlo
