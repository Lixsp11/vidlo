#pragma once

#include <cuda.h>
#include <sys/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hevc.h"
#include "input.h"
#include "nvdec.h"
#include "postprocess.h"

namespace vidlo {

// 视频索引里解码顺序上的一个包：压缩数据在文件里的起止字节（不含块头）、所属的帧号、是不是 GOP 的入口
struct IndexedPacket {
  int64_t start = 0;
  int64_t end = 0;
  int64_t number = 0;
  bool keyframe = false;
};

// 一次请求交出的一帧；画面已写到调用方给的位置
struct IndexedFrame {
  int64_t number = 0;
  int decode_status = 0;
};

// 累计各次请求；last_range_* 是最近一次请求读的字节范围
struct IndexedStats {
  int64_t reads = 0;  // 读了字节的请求数（对象存储的 HTTP 请求数另由 Python 层记）
  int64_t bytes_read = 0;
  int64_t read_calls = 0;
  int64_t packets = 0;
  int64_t decode_calls = 0;
  int64_t displayed = 0;
  int64_t kept = 0;
  int64_t skipped = 0;            // 没有图像参考、也不是目标而没送进解码器的包
  int64_t decoder_reuses = 0;     // 成功的请求里，硬件解码器取自缓存的次数
  int64_t decoder_creations = 0;  // 新建的次数
  std::array<int64_t, 16> status_counts{};
  int64_t last_range_start = -1;
  int64_t last_range_end = -1;
};

// 按视频索引读帧，不读文件头。每次请求：取文件里一个连续的字节区间（本地从 input 读，对象存储由调用方取好传进来），
// 按索引切出解码顺序上连续的一段包，从 GOP 入口
// 送到最后一个所需的包，发 EOS 排空，要的帧写进调用方按请求分配好的数组。要么交出全部要的帧，要么报错（调用方丢掉写了
// 一半的数组）；出错后这个读取器不再可用。
// 每次请求新建 parser 与码流状态，硬件解码器取自进程级缓存（DecoderCache），请求全部核对无误后放回。
// skip_unreferenced 为真时，没有图像参考、也不是目标的包不送进解码器（hevc.h 的 unreferenced）。
// 第一次请求时才初始化 CUDA：父进程初始化过 CUDA 后 fork 出的子进程里，新建会话时报错（nvdec.cpp 的进程检查）。
class IndexedDecoder {
 public:
  IndexedDecoder(std::shared_ptr<Input> input, std::vector<uint8_t> codec_config, int device, OutputSpec output,
                 bool skip_unreferenced = true);
  ~IndexedDecoder();
  IndexedDecoder(const IndexedDecoder&) = delete;
  IndexedDecoder& operator=(const IndexedDecoder&) = delete;

  // packets 按解码顺序，正好覆盖 [range_start, range_end)；targets 是要交出的帧号，不重复，都在 packets 里。
  // data 为空时从 input 读这段字节（本地文件）；不为空时就是调用方已经取到的这段字节（对象存储），长度是 range_end − range_start。
  // destinations 与 targets 一一对应，是各帧画面的写入位置，每处 frame_bytes 字节（必须等于输出一帧的大小）。
  // 按帧号升序交出各帧的帧号与解码状态
  std::vector<IndexedFrame> read(int64_t range_start, int64_t range_end, const std::vector<IndexedPacket>& packets,
                                 const std::vector<int64_t>& targets, const uint8_t* data, const std::vector<uint8_t*>& destinations,
                                 size_t frame_bytes);
  void close();

  const IndexedStats& stats() const { return stats_; }
  bool has_snapshot() const { return has_snapshot_; }
  const DecoderSnapshot& snapshot() const { return snapshot_; }  // 最近一次请求的解码配置

 private:
  std::vector<IndexedFrame> read_unchecked(int64_t range_start, int64_t range_end, const std::vector<IndexedPacket>& packets,
                                           const std::vector<int64_t>& targets, const uint8_t* data,
                                           const std::vector<uint8_t*>& destinations, size_t frame_bytes);
  void release_context();

  std::shared_ptr<Input> input_;
  // 每次请求的会话结束时释放它对 primary context 的引用；没有别的引用时 CUDA 会销毁 context，下次请求重建要几百毫秒。
  // 读取器自己持有一份引用，让 context 在两次请求之间保留。context_process_ 是加这份引用的进程：读取器可能在父进程里
  // 建好、fork 之后才在子进程里第一次读
  pid_t context_process_ = 0;
  CUdevice context_device_ = 0;
  bool context_retained_ = false;
  std::vector<uint8_t> codec_config_;
  HvcC hvcc_;
  int device_;
  OutputSpec output_;
  bool skip_unreferenced_;
  bool closed_ = false;
  bool failed_ = false;
  IndexedStats stats_;
  bool has_snapshot_ = false;
  DecoderSnapshot snapshot_;
};

}  // namespace vidlo
