#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
}

#include "hevc.h"
#include "input.h"
#include "mkv_scanner.h"

namespace vidlo {

struct StreamInfo {
  std::string container;              // libavformat 的格式名，如 "matroska,webm"
  std::string codec;                  // 第一期只有 "hevc"
  std::vector<uint8_t> codec_config;  // 容器里的 extradata，MKV 为 hvcC
  int stream_index = -1;
  int time_base_num = 0;
  int time_base_den = 0;
  // 容器记录的、影响 RGB 转换的颜色信息（MKV 的 Colour 元素），FFmpeg 的枚举值；用来与 SPS 的标记核对
  int color_range = 0;      // AVColorRange：0 未指定，1 limited，2 full
  int color_matrix = 2;     // AVColorSpace，编号同 H.265 的 matrix_coefficients，2 为未指定
  int chroma_location = 0;  // AVChromaLocation：0 未指定，1 left（type 0），2 center（type 1），3 topleft（type 2）…
  int64_t file_size = -1;
};

// 送给解码器的一个包。data 是起始码格式，在下一次调用 Demuxer::next() 之前有效
struct Packet {
  const uint8_t* data = nullptr;
  size_t size = 0;
  int64_t pts = 0;
  int64_t data_start = 0;  // 这个包的压缩数据在文件中的起止字节（不含块头），已与文件内容逐字节核对
  int64_t data_end = 0;
  bool keyframe = false;  // 容器标记的关键帧
};

struct InputStats {
  int64_t bytes_read = 0;
  int64_t seeks = 0;
  std::vector<std::pair<int64_t, int64_t>> first_seeks;  // 前 32 次 seek 的（起点，目标）
};

// 顺序解封装一个视频轨，并用 hevc_mp4toannexb 转成解码器要的起始码格式。其他轨道在解封装时就丢弃。
// 文件经自定义 AVIO 从 Input 读入；同一条字节流同时交给 MkvScanner 扫描结构。读过的字节在一个有界窗口里保留到
// 核对完，用来逐个核对每个包在文件中的位置和内容，不重复读文件。name 用于报错，也供 libavformat 按扩展名探测格式。
class Demuxer {
 public:
  Demuxer(std::shared_ptr<Input> input, const std::string& name);
  ~Demuxer();
  Demuxer(const Demuxer&) = delete;
  Demuxer& operator=(const Demuxer&) = delete;

  const StreamInfo& info() const { return info_; }
  const InputStats& input_stats() const { return input_stats_; }
  // 文件被截断时不完整元素的起点，没有截断时为 -1；读到文件末尾后才确定
  int64_t truncated_at() const { return scanner_ ? scanner_->truncated_at() : -1; }

  // 取下一个视频包；读完返回 false
  bool next(Packet& out);

 private:
  struct SourceInfo {
    int64_t pts;
    int64_t data_start;
    int64_t data_end;
    bool keyframe;
  };

  static int read_input(void* opaque, uint8_t* buffer, int size);
  static int64_t seek_input(void* opaque, int64_t offset, int whence);
  void remember(int64_t start, const uint8_t* data, size_t size);
  void evict();
  void scan();  // 用窗口里已读到的字节推进结构扫描
  bool copy_from_window(int64_t start, uint8_t* out, size_t size) const;
  bool window_equals(int64_t start, const uint8_t* data, size_t size, bool& covered) const;
  SourceInfo locate_payload(const AVPacket& packet);
  bool send_next_video_packet();
  void release();

  std::shared_ptr<Input> input_;
  int64_t position_ = 0;
  AVIOContext* avio_ = nullptr;
  AVFormatContext* format_ = nullptr;
  AVBSFContext* bsf_ = nullptr;
  AVPacket* in_ = nullptr;
  AVPacket* out_ = nullptr;
  StreamInfo info_;
  InputStats input_stats_;
  int64_t track_number_ = -1;
  HvcC hvcc_;  // codec_config 的 NAL 长度字段字节数与参数集，用来逐包核对
  std::unique_ptr<MkvScanner> scanner_;
  bool scanning_ = false;  // 确认是 MKV 之后才扫描；在那之前读过的字节都留在窗口里，确认后从头补扫
  int64_t read_end_ = 0;  // 读到过的最远位置

  // 读过的字节：起点 → 数据。超过软上限时按读入顺序淘汰，但扫描器还没处理的字节、还没核对的块不淘汰
  std::map<int64_t, std::vector<uint8_t>> window_;
  std::deque<int64_t> window_order_;
  size_t window_bytes_ = 0;
  std::exception_ptr input_error_;  // AVIO 回调里出的错，av_read_frame 返回后抛出

  std::deque<SourceInfo> pending_;  // 已送进 BSF、还没取出的包；hevc_mp4toannexb 一进一出
  bool flushed_ = false;
};

}  // namespace vidlo
