#include "demuxer.h"

#include <algorithm>
#include <cstring>

#include "error.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/mem.h>
}

namespace vidlo {
namespace {

constexpr int kAvioBufferSize = 256 * 1024;
constexpr size_t kWindowBytes = 64u * 1024 * 1024;      // 软上限：足够覆盖 AVIO 缓冲与解封装器的提前量
constexpr size_t kMaxWindowBytes = 512u * 1024 * 1024;  // 硬上限：单个块或提前量大到这个程度时报错
constexpr size_t kMaxRecordedSeeks = 32;

std::string av_error_text(int err) {
  char text[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(err, text, sizeof(text));
  return text;
}

void check_av(int err, const char* what) {
  if (err < 0) {
    throw Error(std::string(what) + " failed: " + av_error_text(err));
  }
}

}  // namespace

Demuxer::Demuxer(std::shared_ptr<Input> input, const std::string& name) : input_(std::move(input)) {
  try {
    info_.file_size = input_->size();
    scanner_ = std::make_unique<MkvScanner>(info_.file_size);

    auto* buffer = static_cast<unsigned char*>(av_malloc(kAvioBufferSize));
    if (buffer == nullptr) {
      throw Error("av_malloc failed");
    }
    avio_ = avio_alloc_context(buffer, kAvioBufferSize, 0, this, &Demuxer::read_input, nullptr, &Demuxer::seek_input);
    if (avio_ == nullptr) {
      av_free(buffer);
      throw Error("avio_alloc_context failed");
    }
    // 本地与远程都按只能顺序读的流交给 libavformat：它不会为 SeekHead 等跳到文件后部，向前跳过靠读入丢弃，
    // 两种来源的读取过程相同。seek 回调只回答文件大小
    avio_->seekable = 0;
    format_ = avformat_alloc_context();
    if (format_ == nullptr) {
      throw Error("avformat_alloc_context failed");
    }
    format_->pb = avio_;
    // 打开失败时 avformat_open_input 会释放 format_ 并置空；自定义的 avio_ 仍由本类释放
    const int opened = avformat_open_input(&format_, name.c_str(), nullptr, nullptr);
    if (opened < 0 && input_error_) {
      std::rethrow_exception(input_error_);
    }
    check_av(opened, "avformat_open_input");
    info_.container = format_->iformat->name;
    // 第一期只验证过 MKV 的块结构
    if (info_.container != "matroska,webm") {
      throw UnsupportedError("container '" + info_.container + "' is not supported yet (only Matroska)");
    }
    scanning_ = true;
    scan();  // 补扫打开文件时已经读过的字节；文件被截断时在这里就会发现
    evict();
    for (unsigned i = 0; i < format_->nb_streams; ++i) {
      const AVStream* stream = format_->streams[i];
      if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && !(stream->disposition & AV_DISPOSITION_ATTACHED_PIC)) {
        info_.stream_index = static_cast<int>(i);
        break;
      }
    }
    if (info_.stream_index < 0) {
      throw Error("no video stream in " + name);
    }
    const AVStream* stream = format_->streams[info_.stream_index];
    const AVCodecParameters* par = stream->codecpar;
    if (par->codec_id != AV_CODEC_ID_HEVC) {
      throw UnsupportedError(std::string("codec '") + avcodec_get_name(par->codec_id) + "' is not supported yet (only HEVC)");
    }
    info_.codec = "hevc";
    // 不调用 avformat_find_stream_info：最小 FFmpeg 没有解码器，参数集直接取 extradata
    if (par->extradata == nullptr || par->extradata_size <= 0) {
      throw Error("video stream has no codec config (extradata)");
    }
    info_.codec_config.assign(par->extradata, par->extradata + par->extradata_size);
    hvcc_ = parse_hvcc(info_.codec_config);
    info_.time_base_num = stream->time_base.num;
    info_.time_base_den = stream->time_base.den;
    if (info_.time_base_num <= 0 || info_.time_base_den <= 0) {
      throw Error("invalid stream time base");
    }
    info_.color_range = par->color_range;
    info_.color_matrix = par->color_space;
    info_.chroma_location = par->chroma_location;
    // 只放行所选视频轨：音频等轨道的块不进入包队列，避免送进解码器
    for (unsigned i = 0; i < format_->nb_streams; ++i) {
      if (static_cast<int>(i) != info_.stream_index) {
        format_->streams[i]->discard = AVDISCARD_ALL;
      }
    }

    const AVBitStreamFilter* filter = av_bsf_get_by_name("hevc_mp4toannexb");
    if (filter == nullptr) {
      throw Error("FFmpeg build has no hevc_mp4toannexb bitstream filter");
    }
    check_av(av_bsf_alloc(filter, &bsf_), "av_bsf_alloc");
    check_av(avcodec_parameters_copy(bsf_->par_in, par), "avcodec_parameters_copy");
    bsf_->time_base_in = stream->time_base;
    check_av(av_bsf_init(bsf_), "av_bsf_init");

    in_ = av_packet_alloc();
    out_ = av_packet_alloc();
    if (in_ == nullptr || out_ == nullptr) {
      throw Error("av_packet_alloc failed");
    }
  } catch (...) {
    release();
    throw;
  }
}

Demuxer::~Demuxer() { release(); }

void Demuxer::release() {
  av_packet_free(&in_);
  av_packet_free(&out_);
  av_bsf_free(&bsf_);
  avformat_close_input(&format_);  // 自定义 IO 不会被它关闭
  if (avio_ != nullptr) {
    av_freep(&avio_->buffer);
    avio_context_free(&avio_);
  }
  input_.reset();
  window_.clear();
  window_order_.clear();
  window_bytes_ = 0;
}

// AVIO 的读、seek 回调由 FFmpeg 的 C 代码调用，异常不能穿出去
int Demuxer::read_input(void* opaque, uint8_t* buffer, int size) {
  auto* self = static_cast<Demuxer*>(opaque);
  try {
    const auto n = static_cast<int64_t>(self->input_->read(self->position_, buffer, static_cast<size_t>(size)));
    if (n == 0) {
      return AVERROR_EOF;
    }
    self->remember(self->position_, buffer, static_cast<size_t>(n));
    self->read_end_ = std::max(self->read_end_, self->position_ + n);
    if (self->scanning_) {
      self->scan();
    }
    self->evict();
    self->position_ += n;
    self->input_stats_.bytes_read += n;
    return static_cast<int>(n);
  } catch (...) {
    self->input_error_ = std::current_exception();
    return AVERROR_EXTERNAL;
  }
}

int64_t Demuxer::seek_input(void* opaque, int64_t offset, int whence) {
  auto* self = static_cast<Demuxer*>(opaque);
  if (whence & AVSEEK_SIZE) {
    return self->info_.file_size;
  }
  int64_t target;
  switch (whence & ~AVSEEK_FORCE) {
    case SEEK_SET:
      target = offset;
      break;
    case SEEK_CUR:
      target = self->position_ + offset;
      break;
    case SEEK_END:
      target = self->info_.file_size + offset;
      break;
    default:
      return AVERROR(EINVAL);
  }
  if (target < 0 || target > self->info_.file_size) {
    return AVERROR(EINVAL);
  }
  if (target == self->position_) {
    return target;
  }
  // 顺序读不支持真的跳转：记下请求，返回失败（不先答应、再让下一次读出错）
  ++self->input_stats_.seeks;
  if (self->input_stats_.first_seeks.size() < kMaxRecordedSeeks) {
    try {
      self->input_stats_.first_seeks.emplace_back(self->position_, target);
    } catch (...) {
      return AVERROR(ENOMEM);
    }
  }
  return AVERROR(ESPIPE);
}

void Demuxer::remember(int64_t start, const uint8_t* data, size_t size) {
  auto it = window_.find(start);
  if (it != window_.end()) {
    // seek 后重读同一位置：文件不变，只在这次读得更长时替换
    if (size > it->second.size()) {
      window_bytes_ += size - it->second.size();
      it->second.assign(data, data + size);
    }
  } else {
    window_.emplace(start, std::vector<uint8_t>(data, data + size));
    window_order_.push_back(start);
    window_bytes_ += size;
  }
}

void Demuxer::scan() {
  scanner_->advance([this](int64_t start, uint8_t* out, size_t count) { return copy_from_window(start, out, count); },
                    read_end_);
}

void Demuxer::evict() {
  // 扫描器还没处理的字节、还没与包核对的块都要留着
  int64_t keep_from = scanner_->position();
  if (!scanner_->blocks().empty()) {
    keep_from = std::min(keep_from, scanner_->blocks().front().content_start);
  }
  while (window_bytes_ > kWindowBytes && !window_order_.empty()) {
    auto old = window_.find(window_order_.front());
    if (old->first + static_cast<int64_t>(old->second.size()) > keep_from) {
      break;
    }
    window_bytes_ -= old->second.size();
    window_.erase(old);
    window_order_.pop_front();
  }
  if (window_bytes_ > kMaxWindowBytes) {
    throw Error("read window exceeds 512 MiB: a Matroska block or the demuxer's read-ahead is too large");
  }
}

bool Demuxer::copy_from_window(int64_t start, uint8_t* out, size_t size) const {
  size_t done = 0;
  while (done < size) {
    const int64_t cur = start + static_cast<int64_t>(done);
    auto it = window_.upper_bound(cur);
    if (it == window_.begin()) {
      return false;
    }
    --it;
    const int64_t offset = cur - it->first;
    if (offset >= static_cast<int64_t>(it->second.size())) {
      return false;
    }
    const size_t n = std::min(size - done, it->second.size() - static_cast<size_t>(offset));
    std::memcpy(out + done, it->second.data() + offset, n);
    done += n;
  }
  return true;
}

bool Demuxer::window_equals(int64_t start, const uint8_t* data, size_t size, bool& covered) const {
  size_t done = 0;
  covered = true;
  while (done < size) {
    const int64_t cur = start + static_cast<int64_t>(done);
    auto it = window_.upper_bound(cur);
    if (it == window_.begin()) {
      covered = false;
      return false;
    }
    --it;
    const int64_t offset = cur - it->first;
    if (offset >= static_cast<int64_t>(it->second.size())) {
      covered = false;
      return false;
    }
    const size_t n = std::min(size - done, it->second.size() - static_cast<size_t>(offset));
    if (std::memcmp(data + done, it->second.data() + offset, n) != 0) {
      return false;
    }
    done += n;
  }
  return true;
}

// libavformat 给出的 pos 是 Block 内容的起点，即块头（libavformat/matroskadec.c：ebml_parse 在读完元素 ID 与长度后
// 记下位置）。逐个对上 MkvScanner 扫到的块，再与窗口里的文件字节逐字节比对。
Demuxer::SourceInfo Demuxer::locate_payload(const AVPacket& packet) {
  const std::string where = " (block at byte " + std::to_string(packet.pos) + ")";
  if (packet.pos < 0) {
    throw Error("video packet without file position");
  }
  // 包交出时它的字节已经读过，扫描器已经扫到它
  auto& blocks = scanner_->blocks();
  auto match = std::find_if(blocks.begin(), blocks.end(),
                            [&](const MkvScanner::Block& block) { return block.content_start >= packet.pos; });
  if (match == blocks.end() || match->content_start != packet.pos) {
    throw Error("packet position does not match any Matroska block" + where);
  }
  const MkvScanner::Block block = *match;
  for (auto it = blocks.begin(); it != match; ++it) {
    if (it->track == block.track) {
      throw Error("the demuxer skipped the video block at byte " + std::to_string(it->content_start) +
                  " (damaged file?)" + where);
    }
  }
  blocks.erase(blocks.begin(), match + 1);

  if (block.in_block_group) {
    throw UnsupportedError("Matroska BlockGroup is not supported yet (only SimpleBlock)" + where);
  }
  if (block.flags & 0x06) {
    throw UnsupportedError("Matroska lacing is not supported yet" + where);
  }
  if (track_number_ < 0) {
    track_number_ = block.track;
  } else if (block.track != track_number_) {
    throw Error("video packets come from track " + std::to_string(block.track) + " and " + std::to_string(track_number_) +
                where);
  }

  SourceInfo source{};
  source.pts = packet.pts;
  source.data_start = block.content_start + block.header_length;
  source.data_end = source.data_start + packet.size;
  source.keyframe = (packet.flags & AV_PKT_FLAG_KEY) != 0;
  if (source.data_end != block.content_end) {
    throw Error("packet size does not match its Matroska block (truncated or modified data)" + where);
  }
  bool covered = false;
  if (!window_equals(source.data_start, packet.data, static_cast<size_t>(packet.size), covered)) {
    if (!covered) {
      throw Error("packet data is no longer in the read window" + where);
    }
    // 内容压缩（如 header stripping）或其他布局时，文件里的字节不是包数据本身，无法按字节范围随机读取
    throw UnsupportedError("packet data does not match the file bytes after the block header "
                           "(content compression or unexpected layout)" + where);
  }
  check_access_unit(packet.data, static_cast<size_t>(packet.size), hvcc_, where);
  return source;
}

bool Demuxer::send_next_video_packet() {
  while (true) {
    int err = av_read_frame(format_, in_);
    if (input_error_) {
      std::rethrow_exception(input_error_);
    }
    if (err < 0 && err != AVERROR_EOF && scanner_->truncated_at() >= 0 && read_end_ == info_.file_size) {
      // 文件已读到末尾且扫描器已发现截断：libavformat 读不完最后那个元素时可能报错而不是 EOF，按读完处理
      err = AVERROR_EOF;
    }
    if (err == AVERROR_EOF) {
      // libavformat 遇到不完整的块也可能当作正常结束：结构必须扫到文件末尾，扫到的视频块都必须已经交出
      scan();
      scanner_->finish(read_end_);
      if (read_end_ == info_.file_size) {
        input_->confirm_end();  // 远程下载：字节齐全之外，响应本身也要成功结束
      }
      if (track_number_ < 0) {
        throw Error("no video packets in the file");
      }
      for (const MkvScanner::Block& block : scanner_->blocks()) {
        if (block.track == track_number_) {
          throw Error("the demuxer did not deliver the video block at byte " + std::to_string(block.content_start) +
                      " (truncated or damaged file)");
        }
      }
      return false;
    }
    check_av(err, "av_read_frame");
    if (in_->stream_index != info_.stream_index) {
      av_packet_unref(in_);
      continue;
    }
    if (in_->pts == AV_NOPTS_VALUE) {
      throw Error("video packet without pts at byte " + std::to_string(in_->pos));
    }
    // 必须在交给 BSF 之前核对：av_bsf_send_packet 接管 in_ 的引用，调用后 in_ 为空
    pending_.push_back(locate_payload(*in_));
    check_av(av_bsf_send_packet(bsf_, in_), "av_bsf_send_packet");
    return true;
  }
}

bool Demuxer::next(Packet& out) {
  av_packet_unref(out_);
  while (true) {
    int err = av_bsf_receive_packet(bsf_, out_);
    if (err == 0) {
      if (pending_.empty()) {
        throw Error("bitstream filter produced more packets than it received");
      }
      const SourceInfo source = pending_.front();
      pending_.pop_front();
      out.data = out_->data;
      out.size = static_cast<size_t>(out_->size);
      out.pts = source.pts;
      out.data_start = source.data_start;
      out.data_end = source.data_end;
      out.keyframe = source.keyframe;
      return true;
    }
    if (err == AVERROR_EOF) {
      if (!pending_.empty()) {
        throw Error("bitstream filter dropped " + std::to_string(pending_.size()) + " packets");
      }
      return false;
    }
    if (err != AVERROR(EAGAIN)) {
      check_av(err, "av_bsf_receive_packet");
    }
    if (flushed_) {
      throw Error("bitstream filter asks for input after flush");
    }
    if (!send_next_video_packet()) {
      check_av(av_bsf_send_packet(bsf_, nullptr), "av_bsf_send_packet(flush)");
      flushed_ = true;
    }
  }
}

}  // namespace vidlo
