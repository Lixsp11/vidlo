#include "indexed_decoder.h"

#include <unistd.h>

#include <unordered_set>
#include <utility>

#include "error.h"

namespace vidlo {
namespace {

// 时间戳只用来认帧（送包时附带帧号，显示回调原样带回），parser 不靠它排序；时钟取一个固定值
constexpr unsigned kClockRate = 1000;

}  // namespace

IndexedDecoder::IndexedDecoder(std::shared_ptr<Input> input, std::vector<uint8_t> codec_config, int device, OutputSpec output,
                               bool skip_unreferenced)
    : input_(std::move(input)),
      codec_config_(std::move(codec_config)),
      hvcc_(parse_hvcc(codec_config_)),
      device_(device),
      output_(std::move(output)),
      skip_unreferenced_(skip_unreferenced) {}

IndexedDecoder::~IndexedDecoder() { release_context(); }

void IndexedDecoder::release_context() {
  // 引用是 fork 之前加的，子进程里不能调用 CUDA，只丢掉记录
  if (context_retained_ && context_process_ == getpid()) {
    cuDevicePrimaryCtxRelease(context_device_);
  }
  context_retained_ = false;
}

void IndexedDecoder::close() {
  closed_ = true;
  input_.reset();
  release_context();
}

std::vector<IndexedFrame> IndexedDecoder::read(int64_t range_start, int64_t range_end, const std::vector<IndexedPacket>& packets,
                                               const std::vector<int64_t>& targets) {
  if (closed_) {
    throw Error("video is closed");
  }
  if (failed_) {
    throw Error("an earlier error stopped reading this video");
  }
  try {
    return read_unchecked(range_start, range_end, packets, targets);
  } catch (...) {
    failed_ = true;
    throw;
  }
}

std::vector<IndexedFrame> IndexedDecoder::read_unchecked(int64_t range_start, int64_t range_end,
                                                         const std::vector<IndexedPacket>& packets,
                                                         const std::vector<int64_t>& targets) {
  // 请求由 Python 层按索引规划，这里只核对它自洽：包按解码顺序首尾相接地落在区间里，要的帧都在这些包里
  if (packets.empty() || targets.empty()) {
    throw Error("empty request");
  }
  if (range_start < 0 || range_start >= range_end || range_end > input_->size()) {
    throw Error("byte range [" + std::to_string(range_start) + ", " + std::to_string(range_end) + ") does not fit the " +
                std::to_string(input_->size()) + "-byte file");
  }
  if (packets.front().start != range_start || packets.back().end != range_end) {
    throw Error("the byte range does not start at the first packet and end at the last");
  }
  std::unordered_set<int64_t> numbers;
  int64_t previous_end = range_start;
  for (const IndexedPacket& packet : packets) {
    if (packet.start < previous_end || packet.end <= packet.start) {
      throw Error("packets overlap or are out of decoding order");
    }
    if (!numbers.insert(packet.number).second) {
      throw Error("frame " + std::to_string(packet.number) + " appears in two packets");
    }
    previous_end = packet.end;
  }
  if (!packets.front().keyframe) {
    throw Error("decoding must start at a keyframe");
  }
  const std::unordered_set<int64_t> wanted(targets.begin(), targets.end());
  if (wanted.size() != targets.size()) {
    throw Error("requested frames repeat");
  }
  for (int64_t number : targets) {
    if (numbers.count(number) == 0) {
      throw Error("requested frame " + std::to_string(number) + " is not among the decoded packets");
    }
  }

  // 一次读出整个区间；本地文件 pread 可能分几次返回
  std::vector<uint8_t> bytes(static_cast<size_t>(range_end - range_start));
  size_t done = 0;
  int64_t calls = 0;
  while (done < bytes.size()) {
    const size_t n = input_->read(range_start + static_cast<int64_t>(done), bytes.data() + done, bytes.size() - done);
    ++calls;
    if (n == 0) {
      throw Error("the file ends at byte " + std::to_string(range_start + static_cast<int64_t>(done)) +
                  ", before the end of the range " + std::to_string(range_end));
    }
    done += n;
  }
  ++stats_.requests;
  stats_.bytes_read += static_cast<int64_t>(bytes.size());
  stats_.read_calls += calls;
  stats_.last_range_start = range_start;
  stats_.last_range_end = range_end;

  // 会话的构造先做进程检查、初始化 CUDA，之后才在这里加一份 context 引用。硬件解码器从进程级缓存取
  NvdecSession session(device_, cudaVideoCodec_HEVC, kClockRate, output_, /*cache_decoder=*/true);
  if (!context_retained_) {
    CUcontext context = nullptr;
    check_cu(cuDeviceGet(&context_device_, device_), "cuDeviceGet");
    check_cu(cuDevicePrimaryCtxRetain(&context, context_device_), "cuDevicePrimaryCtxRetain");
    context_retained_ = true;
    context_process_ = getpid();
  }
  session.set_targets(targets);
  AnnexbFilter filter(codec_config_);
  std::vector<const IndexedPacket*> fed;  // 实际送进解码器的包，按解码顺序
  std::unordered_set<int64_t> pending;    // 送进去、还没显示的帧号
  int64_t skipped = 0;
  size_t checked = 0;
  // 每解一张图像，核对它和索引：GOP 入口必须是 IDR，其余不能是 IDR；不支持 CRA/BLA
  auto check_pictures = [&]() {
    for (const PictureInfo& picture : session.take_pictures()) {
      const size_t i = checked++;
      if (i >= fed.size()) {
        throw Error("more decoded pictures than packets");
      }
      const IndexedPacket& packet = *fed[i];
      const std::string at = " (frame " + std::to_string(packet.number) + ")";
      if (picture.irap && !picture.idr) {
        throw UnsupportedError("random access picture that is not IDR (CRA/BLA, open GOP); not supported yet" + at);
      }
      if (picture.idr != packet.keyframe) {
        throw Error(std::string(packet.keyframe ? "the index marks a GOP start that is not an IDR picture"
                                                : "the index does not mark this IDR picture as a GOP start") +
                    " (wrong decode_start in the video index?)" + at);
      }
    }
  };
  for (const IndexedPacket& packet : packets) {
    const uint8_t* data = bytes.data() + (packet.start - range_start);
    const auto size = static_cast<size_t>(packet.end - packet.start);
    const AccessUnit unit = check_access_unit(
        data, size, hvcc_, " (frame " + std::to_string(packet.number) + ", bytes " + std::to_string(packet.start) + "-" + std::to_string(packet.end) + ")");
    // 没有图像参考、也不是要交出的帧，解了也用不上。索引标成 GOP 入口的包照送：解码时核对它是不是 IDR，
    // 跳过就会漏掉索引的这个错误
    if (skip_unreferenced_ && unreferenced(unit, hvcc_) && wanted.count(packet.number) == 0 && !packet.keyframe) {
      ++skipped;
      continue;
    }
    const uint8_t* annexb = nullptr;
    size_t annexb_size = 0;
    filter.filter(data, size, annexb, annexb_size);
    fed.push_back(&packet);
    pending.insert(packet.number);
    session.feed(annexb, annexb_size, packet.number);
    check_pictures();
    if (session.stats().decode_calls != session.stats().packets) {
      throw Error("the packet of frame " + std::to_string(packet.number) + " did not produce exactly one picture");
    }
  }
  filter.finish();
  session.finish();
  check_pictures();

  // 显示的帧正好是送进去的那些帧，各一次，帧号递增（GOP 封闭，显示顺序就是帧号顺序）
  std::vector<IndexedFrame> frames;
  frames.reserve(targets.size());
  int64_t previous = -1;
  DecodedFrame frame;
  while (session.pop(frame)) {
    if (pending.erase(frame.timestamp) == 0) {
      throw Error("displayed frame " + std::to_string(frame.timestamp) + " was not sent or was displayed twice");
    }
    if (frame.timestamp <= previous) {
      throw Error("display order is not increasing: frame " + std::to_string(previous) + " then " + std::to_string(frame.timestamp));
    }
    previous = frame.timestamp;
    if (frame.kept) {
      frames.push_back({frame.timestamp, frame.decode_status, std::move(frame.image)});
    }
  }
  if (!pending.empty()) {
    throw Error(std::to_string(pending.size()) + " decoded frames were never displayed");
  }
  const DecoderStats& s = session.stats();
  const auto count = static_cast<int64_t>(fed.size());
  if (s.packets != count || s.decode_calls != count || s.displayed != count) {
    throw Error("frame count mismatch: " + std::to_string(count) + " packets, " + std::to_string(s.decode_calls) +
                " decode calls, " + std::to_string(s.displayed) + " frames displayed");
  }
  if (frames.size() != targets.size()) {
    throw Error("only " + std::to_string(frames.size()) + " of " + std::to_string(targets.size()) + " requested frames were decoded");
  }
  // 全部核对无误，硬件解码器才放回缓存
  session.keep_decoder();

  stats_.packets += s.packets;
  stats_.skipped += skipped;
  ++(session.reused_decoder() ? stats_.decoder_reuses : stats_.decoder_creations);
  stats_.decode_calls += s.decode_calls;
  stats_.displayed += s.displayed;
  stats_.kept += s.kept;
  for (size_t i = 0; i < s.status_counts.size(); ++i) {
    stats_.status_counts[i] += s.status_counts[i];
  }
  snapshot_ = session.snapshot();
  has_snapshot_ = true;
  output_width_ = session.processor().output_width();
  output_height_ = session.processor().output_height();
  return frames;
}

}  // namespace vidlo
