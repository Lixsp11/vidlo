#include "hevc.h"

#include <algorithm>
#include <climits>
#include <cstring>

#include "error.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/mem.h>
}

namespace vidlo {
namespace {

void check_av(int err, const char* what) {
  if (err < 0) {
    char text[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, text, sizeof(text));
    throw Error(std::string(what) + " failed: " + text);
  }
}

}  // namespace

// 23 字节的固定部分，后面是各类 NAL 的数组；第 0 字节 configurationVersion 为 1，第 21 字节低 2 位是 NAL 长度字段的字节数减 1
HvcC parse_hvcc(const std::vector<uint8_t>& config) {
  if (config.size() < 23 || config[0] != 1) {
    throw UnsupportedError("HEVC codec config is not an hvcC record; not supported yet");
  }
  HvcC hvcc;
  hvcc.nal_length_size = (config[21] & 3) + 1;
  size_t offset = 23;
  const int arrays = config[22];
  for (int a = 0; a < arrays; ++a) {
    if (offset + 3 > config.size()) {
      throw Error("hvcC array header runs past the end of codec config");
    }
    const int type = config[offset] & 0x3F;
    const int count = (config[offset + 1] << 8) | config[offset + 2];
    offset += 3;
    for (int i = 0; i < count; ++i) {
      if (offset + 2 > config.size()) {
        throw Error("hvcC NAL length runs past the end of codec config");
      }
      const size_t length = (static_cast<size_t>(config[offset]) << 8) | config[offset + 1];
      offset += 2;
      if (length < 2 || length > config.size() - offset) {
        throw Error("hvcC NAL unit runs past the end of codec config");
      }
      if (type >= 32 && type <= 34) {
        hvcc.parameter_sets.emplace_back(config.begin() + static_cast<std::ptrdiff_t>(offset),
                                         config.begin() + static_cast<std::ptrdiff_t>(offset + length));
      }
      // SPS 负载第一个字节：sps_video_parameter_set_id u(4)、sps_max_sub_layers_minus1 u(3)、temporal_id_nesting u(1)
      if (type == 33 && length >= 3) {
        hvcc.max_temporal_id = std::max(hvcc.max_temporal_id, (config[offset + 2] >> 1) & 7);
      }
      offset += length;
    }
  }
  if (offset != config.size()) {
    throw Error("hvcC has unexpected trailing bytes");
  }
  return hvcc;
}

bool unreferenced(const AccessUnit& unit, const HvcC& hvcc) {
  return unit.nal_type >= 0 && unit.nal_type <= 14 && unit.nal_type % 2 == 0 && unit.temporal_id == hvcc.max_temporal_id &&
         !unit.stateful_nal;
}

AccessUnit check_access_unit(const uint8_t* data, size_t size, const HvcC& hvcc, const std::string& where) {
  const auto length_size = static_cast<size_t>(hvcc.nal_length_size);
  size_t offset = 0;
  int pictures = 0;
  bool seen_slice = false;
  AccessUnit unit;
  while (offset < size) {
    if (size - offset < length_size) {
      throw Error("truncated NAL length field in packet" + where);
    }
    size_t length = 0;
    for (size_t i = 0; i < length_size; ++i) {
      length = (length << 8) | data[offset + i];
    }
    offset += length_size;
    if (length < 2 || length > size - offset) {
      throw Error("NAL length " + std::to_string(length) + " does not fit the packet" + where);
    }
    const uint8_t* nal = data + offset;
    const int type = (nal[0] >> 1) & 0x3F;
    const int layer = ((nal[0] & 1) << 5) | (nal[1] >> 3);
    if (layer != 0) {
      throw UnsupportedError("multi-layer HEVC (nuh_layer_id > 0) is not supported yet" + where);
    }
    if (type <= 31) {  // VCL NAL：slice
      if (length < 3) {
        throw Error("slice NAL shorter than its header" + where);
      }
      if (nal[2] & 0x80) {
        ++pictures;
      } else if (!seen_slice) {
        throw UnsupportedError("packet starts in the middle of a picture (a picture split across packets) is not supported yet" + where);
      }
      if (!seen_slice) {
        unit.nal_type = type;
        unit.temporal_id = (nal[1] & 7) - 1;  // nuh_temporal_id_plus1
      }
      seen_slice = true;
    } else if (type >= 32 && type <= 34) {
      const bool known = std::any_of(hvcc.parameter_sets.begin(), hvcc.parameter_sets.end(), [&](const std::vector<uint8_t>& set) {
        return set.size() == length && std::memcmp(set.data(), nal, length) == 0;
      });
      if (!known) {
        throw UnsupportedError("a parameter set inside the stream differs from the codec config; not supported yet" + where);
      }
    } else if (type != 35 && type != 38 && type != 39 && type != 40) {  // 访问单元分隔符、填充数据、SEI 之外
      unit.stateful_nal = true;
    }
    offset += length;
  }
  if (pictures != 1) {
    throw UnsupportedError("packet contains " + std::to_string(pictures) + " pictures; exactly one is supported" + where);
  }
  return unit;
}

AnnexbFilter::AnnexbFilter(const std::vector<uint8_t>& codec_config) {
  try {
    const AVBitStreamFilter* filter = av_bsf_get_by_name("hevc_mp4toannexb");
    if (filter == nullptr) {
      throw Error("FFmpeg build has no hevc_mp4toannexb bitstream filter");
    }
    check_av(av_bsf_alloc(filter, &bsf_), "av_bsf_alloc");
    AVCodecParameters* par = bsf_->par_in;
    par->codec_type = AVMEDIA_TYPE_VIDEO;
    par->codec_id = AV_CODEC_ID_HEVC;
    // extradata 要求末尾带填充（FFmpeg 的位读取可能越过末尾）；之后由 par_in 释放
    par->extradata = static_cast<uint8_t*>(av_mallocz(codec_config.size() + AV_INPUT_BUFFER_PADDING_SIZE));
    if (par->extradata == nullptr) {
      throw Error("av_mallocz failed");
    }
    std::memcpy(par->extradata, codec_config.data(), codec_config.size());
    par->extradata_size = static_cast<int>(codec_config.size());
    bsf_->time_base_in = AVRational{1, 1};  // 包不带时间戳，帧号另外传给解码器
    check_av(av_bsf_init(bsf_), "av_bsf_init");
    in_ = av_packet_alloc();
    out_ = av_packet_alloc();
    if (in_ == nullptr || out_ == nullptr) {
      throw Error("av_packet_alloc failed");
    }
  } catch (...) {
    av_packet_free(&in_);
    av_packet_free(&out_);
    av_bsf_free(&bsf_);
    throw;
  }
}

AnnexbFilter::~AnnexbFilter() {
  av_packet_free(&in_);
  av_packet_free(&out_);
  av_bsf_free(&bsf_);
}

void AnnexbFilter::filter(const uint8_t* data, size_t size, const uint8_t*& out, size_t& out_size) {
  av_packet_unref(out_);
  // FFmpeg 的包长是 int，并要在末尾留填充；更大的包（只可能来自错误的索引）在这里拒绝，不截断长度
  if (size > static_cast<size_t>(INT_MAX - AV_INPUT_BUFFER_PADDING_SIZE)) {
    throw Error("packet of " + std::to_string(size) + " bytes is too large");
  }
  check_av(av_new_packet(in_, static_cast<int>(size)), "av_new_packet");
  std::memcpy(in_->data, data, size);
  check_av(av_bsf_send_packet(bsf_, in_), "av_bsf_send_packet");  // 接管 in_ 的数据，调用后 in_ 为空
  const int err = av_bsf_receive_packet(bsf_, out_);
  if (err == AVERROR(EAGAIN)) {
    throw Error("hevc_mp4toannexb kept a packet back (it should return one packet per input)");
  }
  check_av(err, "av_bsf_receive_packet");
  out = out_->data;
  out_size = static_cast<size_t>(out_->size);
}

void AnnexbFilter::finish() {
  av_packet_unref(out_);
  check_av(av_bsf_send_packet(bsf_, nullptr), "av_bsf_send_packet(flush)");
  const int err = av_bsf_receive_packet(bsf_, out_);
  if (err == 0) {
    throw Error("hevc_mp4toannexb returned a packet after flush");
  }
  if (err != AVERROR_EOF) {
    check_av(err, "av_bsf_receive_packet");
  }
}

}  // namespace vidlo
