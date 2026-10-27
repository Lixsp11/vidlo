#pragma once

#include <cuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>
#include <sys/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "decoder_cache.h"
#include "postprocess.h"

namespace vidlo {

// 按显示顺序交出的一帧。没选中的帧也交出（只有时间戳和解码状态），供调用方核对帧与包的对应
struct DecodedFrame {
  int64_t timestamp = 0;  // 提交包时附带的值，由显示回调按显示顺序带回
  int decode_status = 0;  // cuvidDecodeStatus
  bool kept = false;      // 是否选中；只有选中的帧做后处理
  std::vector<uint8_t> image;  // FrameProcessor::produce 的结果；选中的帧写到了 set_targets 给的位置时为空
};

// 每次 cuvidDecodePicture 一项，按解码顺序，用于核对容器的关键帧标记
struct PictureInfo {
  bool intra = false;
  bool irap = false;  // HEVC：随机访问点（IDR、CRA、BLA）
  bool idr = false;   // HEVC：IDR
};

// 实际交给 NVDEC 的参数和它报告的原始值，用于核对与排错
struct DecoderSnapshot {
  int driver_version = 0;
  int device = 0;
  std::string device_name;
  std::string device_uuid;
  CUVIDPARSERPARAMS parser{};  // 回调与 pUserData 不记录
  bool has_format = false;
  CUVIDEOFORMAT format{};
  CUVIDDECODECAPS caps{};
  CUVIDDECODECREATEINFO create{};  // vidLock 不记录
  int sequence_calls = 0;
};

struct DecoderStats {
  int64_t packets = 0;
  int64_t decode_calls = 0;
  int64_t displayed = 0;
  int64_t kept = 0;  // 选中并做了后处理的帧数
  std::array<int64_t, 16> status_counts{};  // 按 cuvidDecodeStatus 计数
  bool end_of_stream = false;
};

// 一个 NVDEC 解码会话：parser + decoder。同步基线：显示回调里映射、检查状态，选中的帧裁剪、转换、缩放并复制到内存，
// 同步之后归还 surface。
// 哪些帧选中有两种给法，二选一：
// - 顺序读：调用方事先按显示顺序给出（add_selection）。一次 feed 或 finish 交出的帧不会多于解码 surface 数，
//   调用方在 feed、finish 之前保证 selection_pending() 不少于 kSelectionReserve。
// - 按帧号读：送包时的时间戳就是帧号，set_targets 给出要的帧号，显示回调按时间戳挑。不能按显示位置数：
//   显示在目标之前、解码顺序在区间之外的 B 帧根本没送进来，位置会错开。set_targets 同时给出每帧的写入位置时，
//   后处理的结果直接写到那里（调用方按请求一次分配好最终的数组），不另分配。
// cache_decoder 为真时，硬件解码器先从 DecoderCache 取；会话正常结束、调用方核对无误后调用 keep_decoder()，析构时放回缓存，
// 否则销毁。parser、码流状态、后处理都随会话新建。
class NvdecSession {
 public:
  static constexpr size_t kSelectionReserve = 64;  // 建解码器时检查解码 surface 数小于它

  NvdecSession(int device, cudaVideoCodec codec, unsigned clock_rate, OutputSpec output, bool cache_decoder = false);
  ~NvdecSession();
  NvdecSession(const NvdecSession&) = delete;
  NvdecSession& operator=(const NvdecSession&) = delete;

  void feed(const uint8_t* data, size_t size, int64_t timestamp);
  void finish();  // 发结束标记，交出剩余的帧
  bool pop(DecodedFrame& out);
  std::vector<PictureInfo> take_pictures();  // 取走上次以来解码的图像信息
  void add_selection(const std::vector<bool>& keep);  // 接下来按显示顺序各帧是否选中
  size_t selection_pending() const { return selection_.size(); }
  // 改为按时间戳选帧，送包之前调用。destinations 不为空时与 timestamps 一一对应，每处 frame_bytes 字节，
  // 必须等于后处理的输出大小；调用方保证这些内存在会话结束前有效
  void set_targets(const std::vector<int64_t>& timestamps, const std::vector<uint8_t*>& destinations = {}, size_t frame_bytes = 0);
  void keep_decoder();  // finish() 成功之后调用：析构时把硬件解码器放回缓存
  bool reused_decoder() const { return reused_decoder_; }  // 硬件解码器是不是从缓存取的

  // 会话是在别的进程里建的（本进程是建好之后 fork 出来的）时报错：子进程不能用父进程的 CUDA
  void check_process() const;

  const DecoderSnapshot& snapshot() const { return snapshot_; }
  const DecoderStats& stats() const { return stats_; }
  const FrameProcessor& processor() const { return *processor_; }

 private:
  static int CUDAAPI on_sequence(void* user, CUVIDEOFORMAT* format);
  static int CUDAAPI on_decode(void* user, CUVIDPICPARAMS* picture);
  static int CUDAAPI on_display(void* user, CUVIDPARSERDISPINFO* display);
  int handle_sequence(const CUVIDEOFORMAT& format);
  int handle_decode(CUVIDPICPARAMS* picture);
  int handle_display(const CUVIDPARSERDISPINFO* display);
  void record_callback_error(const char* where) noexcept;
  void parse(CUVIDSOURCEDATAPACKET& packet);
  void release();

  cudaVideoCodec codec_;
  pid_t process_;  // 建会话的进程
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  bool context_retained_ = false;
  CUstream stream_ = nullptr;
  CUvideoparser parser_ = nullptr;
  std::unique_ptr<HardwareDecoder> hardware_;  // 建好或从缓存取出的硬件解码器，连同它的锁
  CUvideodecoder decoder_ = nullptr;           // hardware_->decoder
  bool cache_decoder_ = false;
  bool keep_decoder_ = false;
  bool reused_decoder_ = false;

  // 回调里出的错，回调返回后由 parse() 抛出。只保存 exception_ptr，记录过程本身不分配内存、不会再抛异常
  std::exception_ptr callback_error_;
  const char* callback_where_ = "";
  bool failed_ = false;
  bool finished_ = false;

  std::unique_ptr<FrameProcessor> processor_;  // 显存在建解码器时分配，在 release() 里持有 context 时释放
  std::deque<bool> selection_;
  bool by_timestamp_ = false;  // set_targets 之后按时间戳选帧
  std::unordered_map<int64_t, uint8_t*> targets_;  // 要的帧号 → 写入位置（没给位置时为空指针）
  size_t frame_bytes_ = 0;
  std::deque<DecodedFrame> frames_;
  std::vector<PictureInfo> pictures_;
  DecoderSnapshot snapshot_;
  DecoderStats stats_;
};

}  // namespace vidlo
