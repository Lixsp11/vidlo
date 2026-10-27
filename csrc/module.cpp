// Python 绑定：输入（本地文件、下载流）、输出规格、顺序解码和按帧号解码这一层；接口形状和参数检查在 src/vidlo/ 的 Python 代码里。

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "demuxer.h"
#include "error.h"
#include "indexed_decoder.h"
#include "input.h"
#include "nvdec.h"
#include "postprocess.h"

namespace py = pybind11;

namespace vidlo {
namespace {

cudaVideoCodec to_cuvid_codec(const std::string& codec) {
  if (codec == "hevc") {
    return cudaVideoCodec_HEVC;
  }
  throw UnsupportedError("codec '" + codec + "' is not supported yet (only HEVC)");
}

py::dict area_dict(short left, short top, short right, short bottom) {
  py::dict d;
  d["left"] = left;
  d["top"] = top;
  d["right"] = right;
  d["bottom"] = bottom;
  return d;
}

py::dict parser_dict(const CUVIDPARSERPARAMS& p) {
  py::dict d;
  d["CodecType"] = static_cast<int>(p.CodecType);
  d["ulMaxNumDecodeSurfaces"] = p.ulMaxNumDecodeSurfaces;
  d["ulClockRate"] = p.ulClockRate;
  d["ulErrorThreshold"] = p.ulErrorThreshold;
  d["ulMaxDisplayDelay"] = p.ulMaxDisplayDelay;
  d["bAnnexb"] = static_cast<unsigned>(p.bAnnexb);
  d["bMemoryOptimize"] = static_cast<unsigned>(p.bMemoryOptimize);
  d["pfnGetOperatingPoint_set"] = p.pfnGetOperatingPoint != nullptr;
  d["pfnGetSEIMsg_set"] = p.pfnGetSEIMsg != nullptr;
  d["pExtVideoInfo_set"] = p.pExtVideoInfo != nullptr;
  return d;
}

py::dict format_dict(const CUVIDEOFORMAT& f) {
  py::dict d;
  d["codec"] = static_cast<int>(f.codec);
  d["frame_rate"] = py::make_tuple(f.frame_rate.numerator, f.frame_rate.denominator);
  d["progressive_sequence"] = f.progressive_sequence;
  d["bit_depth_luma_minus8"] = f.bit_depth_luma_minus8;
  d["bit_depth_chroma_minus8"] = f.bit_depth_chroma_minus8;
  d["min_num_decode_surfaces"] = f.min_num_decode_surfaces;
  d["coded_width"] = f.coded_width;
  d["coded_height"] = f.coded_height;
  d["display_area"] = area_dict(static_cast<short>(f.display_area.left), static_cast<short>(f.display_area.top),
                                static_cast<short>(f.display_area.right), static_cast<short>(f.display_area.bottom));
  d["chroma_format"] = static_cast<int>(f.chroma_format);
  d["bitrate"] = f.bitrate;
  d["display_aspect_ratio"] = py::make_tuple(f.display_aspect_ratio.x, f.display_aspect_ratio.y);
  py::dict v;
  v["video_format"] = static_cast<unsigned>(f.video_signal_description.video_format);
  v["video_full_range_flag"] = static_cast<unsigned>(f.video_signal_description.video_full_range_flag);
  v["color_primaries"] = f.video_signal_description.color_primaries;
  v["transfer_characteristics"] = f.video_signal_description.transfer_characteristics;
  v["matrix_coefficients"] = f.video_signal_description.matrix_coefficients;
  d["video_signal_description"] = v;
  d["seqhdr_data_length"] = f.seqhdr_data_length;
  return d;
}

py::dict caps_dict(const CUVIDDECODECAPS& c) {
  py::dict d;
  d["eCodecType"] = static_cast<int>(c.eCodecType);
  d["eChromaFormat"] = static_cast<int>(c.eChromaFormat);
  d["nBitDepthMinus8"] = c.nBitDepthMinus8;
  d["bIsSupported"] = c.bIsSupported;
  d["nNumNVDECs"] = c.nNumNVDECs;
  d["nOutputFormatMask"] = c.nOutputFormatMask;
  d["nMaxWidth"] = c.nMaxWidth;
  d["nMaxHeight"] = c.nMaxHeight;
  d["nMaxMBCount"] = c.nMaxMBCount;
  d["nMinWidth"] = c.nMinWidth;
  d["nMinHeight"] = c.nMinHeight;
  d["bIsHistogramSupported"] = c.bIsHistogramSupported;
  d["nCounterBitDepth"] = c.nCounterBitDepth;
  d["nMaxHistogramBins"] = c.nMaxHistogramBins;
  return d;
}

py::dict create_dict(const CUVIDDECODECREATEINFO& c);

py::dict snapshot_dict(const DecoderSnapshot& s) {
  py::dict d;
  d["driver_version"] = s.driver_version;
  d["device"] = s.device;
  d["device_name"] = s.device_name;
  d["device_uuid"] = s.device_uuid;
  d["sdk_headers"] = VIDLO_NVSDK_VERSION;  // 头文件里没有版本宏，由构建时传入
  d["parser"] = parser_dict(s.parser);
  d["sequence_calls"] = s.sequence_calls;
  if (s.has_format) {
    d["format"] = format_dict(s.format);
    d["caps"] = caps_dict(s.caps);
    d["create"] = create_dict(s.create);
  }
  return d;
}

py::dict status_dict(const std::array<int64_t, 16>& counts) {
  py::dict status;
  for (size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] != 0) {
      status[py::int_(i)] = counts[i];
    }
  }
  return status;
}

// 交出的画面：数组直接用这块内存，由 capsule 持有，不再复制。RGB 为 [高, 宽, 3]，NV12 为 [高×3/2, 宽]
py::array_t<uint8_t> image_array(std::vector<uint8_t>&& image, int width, int height, bool rgb) {
  auto* buffer = new std::vector<uint8_t>(std::move(image));
  py::capsule owner(buffer, [](void* p) { delete static_cast<std::vector<uint8_t>*>(p); });
  const auto w = static_cast<py::ssize_t>(width);
  const auto h = static_cast<py::ssize_t>(height);
  if (rgb) {
    return py::array_t<uint8_t>({h, w, static_cast<py::ssize_t>(3)}, {w * 3, static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(1)},
                                buffer->data(), owner);
  }
  return py::array_t<uint8_t>({h * 3 / 2, w}, {w, static_cast<py::ssize_t>(1)}, buffer->data(), owner);
}

py::dict create_dict(const CUVIDDECODECREATEINFO& c) {
  py::dict d;
  d["ulWidth"] = c.ulWidth;
  d["ulHeight"] = c.ulHeight;
  d["ulNumDecodeSurfaces"] = c.ulNumDecodeSurfaces;
  d["CodecType"] = static_cast<int>(c.CodecType);
  d["ChromaFormat"] = static_cast<int>(c.ChromaFormat);
  d["ulCreationFlags"] = c.ulCreationFlags;
  d["bitDepthMinus8"] = c.bitDepthMinus8;
  d["ulIntraDecodeOnly"] = c.ulIntraDecodeOnly;
  d["ulMaxWidth"] = c.ulMaxWidth;
  d["ulMaxHeight"] = c.ulMaxHeight;
  d["display_area"] = area_dict(c.display_area.left, c.display_area.top, c.display_area.right, c.display_area.bottom);
  d["OutputFormat"] = static_cast<int>(c.OutputFormat);
  d["DeinterlaceMode"] = static_cast<int>(c.DeinterlaceMode);
  d["ulTargetWidth"] = c.ulTargetWidth;
  d["ulTargetHeight"] = c.ulTargetHeight;
  d["ulNumOutputSurfaces"] = c.ulNumOutputSurfaces;
  d["target_rect"] = area_dict(c.target_rect.left, c.target_rect.top, c.target_rect.right, c.target_rect.bottom);
  d["enableHistogram"] = c.enableHistogram;
  return d;
}


// 顺序读一段录像：解封装 → NVDEC → 选中的帧裁剪、转换、缩放 → 内存。按显示顺序交出选中的帧，并给出写视频索引用的逐帧信息。
// 构造时只读文件头；start() 之后才初始化 CUDA、建解码会话，Python 层在两者之间按 codec_config 检查输出参数。
class SequentialDecoder {
 public:
  SequentialDecoder(std::shared_ptr<Input> input, const std::string& name)
      : demuxer_(std::make_unique<Demuxer>(std::move(input), name)), stream_(demuxer_->info()) {}

  void start(int device, const OutputSpec& output) {
    if (!demuxer_) {
      throw Error("video is closed");
    }
    if (session_) {
      throw Error("decoder already started");
    }
    // 时间戳按 pts × 时间基分子提交，parser 时钟取分母；1/1000 时无损
    session_ = std::make_unique<NvdecSession>(device, to_cuvid_codec(stream_.codec), static_cast<unsigned>(stream_.time_base_den),
                                              output);
  }

  py::dict info() const {
    const StreamInfo& s = stream_;
    py::dict d;
    d["container"] = s.container;
    d["codec"] = s.codec;
    d["codec_config"] = py::bytes(reinterpret_cast<const char*>(s.codec_config.data()), s.codec_config.size());
    d["time_base"] = py::make_tuple(s.time_base_num, s.time_base_den);
    d["file_size"] = s.file_size;
    py::dict color;
    color["range"] = s.color_range;
    color["matrix"] = s.color_matrix;
    color["chroma_location"] = s.chroma_location;
    d["container_color"] = color;
    return d;
  }

  // 接下来按显示顺序各帧是否选中，每个字节一帧，非零为选中
  void add_selection(const py::bytes& keep) {
    require_open();
    const std::string_view bytes = keep;
    std::vector<bool> flags(bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i) {
      flags[i] = bytes[i] != 0;
    }
    session_->add_selection(flags);
  }

  // 下一个选中的帧 (image, info)；读完返回 None；选帧决定不够时返回 False，调用方 add_selection 后再调用
  py::object next_frame() {
    require_open();
    if (failed_) {
      throw Error("an earlier error stopped reading this video");
    }
    try {
      return next_frame_unchecked();
    } catch (...) {
      // 任何一步出错后整段录像都不再交出帧，避免错位
      failed_ = true;
      throw;
    }
  }

  py::dict decoder_config() const {
    require_open();
    return decoder_config_dict();
  }

  py::dict stats() const {
    require_open();
    return stats_dict();
  }

  void close() {
    session_.reset();
    demuxer_.reset();
  }

 private:
  struct PacketRecord {
    int64_t decode_order;
    int64_t decode_start;
    int64_t data_start;
    int64_t data_end;
    bool keyframe;
  };

  // 一帧在显示顺序中的身份，每个显示的帧（不论是否选中）都要核对
  struct FrameIdentity {
    int64_t number;
    int64_t pts;
    PacketRecord record;
  };

  enum class Step { kFrame, kEnd, kNeedSelection };

  void require_open() const {
    if (!session_ || !demuxer_) {
      throw Error(demuxer_ ? "decoder not started" : "video is closed");
    }
    session_->check_process();  // 从父进程继承来的读取器，在子进程里一律不能用
  }

  py::object next_frame_unchecked() {
    DecodedFrame frame;
    FrameIdentity identity{};
    Step step;
    {
      py::gil_scoped_release release;
      step = advance(frame, identity);
    }
    if (step == Step::kNeedSelection) {
      return py::bool_(false);
    }
    if (step == Step::kEnd) {
      return py::none();
    }
    const FrameProcessor& processor = session_->processor();
    py::array_t<uint8_t> image = image_array(std::move(frame.image), processor.output_width(), processor.output_height(), processor.rgb());
    py::dict meta;
    meta["number"] = identity.number;
    meta["pts"] = identity.pts;
    meta["decode_status"] = frame.decode_status;
    meta["key_frame"] = identity.record.keyframe;
    meta["decode_order"] = identity.record.decode_order;
    meta["decode_start"] = identity.record.decode_start;
    meta["packet_start"] = identity.record.data_start;
    meta["packet_end"] = identity.record.data_end;
    if (processor.rgb()) {
      meta["colorspace"] = processor.colorspace();
      meta["interpolation"] = processor.interpolation().empty() ? py::object(py::none()) : py::str(processor.interpolation());
    }
    return py::make_tuple(image, meta);
  }

  // 显示的每一帧都核对时间戳、对应的包和 GOP 结构；只有选中的帧才交给调用方
  Step advance(DecodedFrame& frame, FrameIdentity& identity) {
    while (true) {
      while (session_->pop(frame)) {
        identity = account(frame);
        if (frame.kept) {
          return Step::kFrame;
        }
      }
      if (done_) {
        if (!records_.empty()) {
          throw Error(std::to_string(records_.size()) + " packets were never displayed");
        }
        // 所有完整写入的帧都已交出，最后明确报告截断，不让调用方误以为录像完整
        if (demuxer_->truncated_at() >= 0) {
          throw TruncatedError("the file is truncated: data from byte " + std::to_string(demuxer_->truncated_at()) + " of " +
                               std::to_string(stream_.file_size) + " is incomplete or missing; all " +
                               std::to_string(next_number_) + " complete frames were delivered");
        }
        return Step::kEnd;
      }
      // 一次送包或收尾交出的帧数不超过解码 surface 数，留够选帧决定才往下送
      if (session_->selection_pending() < NvdecSession::kSelectionReserve) {
        return Step::kNeedSelection;
      }
      Packet packet;
      if (demuxer_->next(packet)) {
        feed(packet);
        continue;
      }
      session_->finish();
      check_pictures();
      done_ = true;
      const DecoderStats& s = session_->stats();
      // 一帧一个包：解码次数、交出的帧数都要等于包数
      if (s.decode_calls != s.packets || s.displayed != s.packets) {
        throw Error("frame count mismatch: " + std::to_string(s.packets) + " packets, " + std::to_string(s.decode_calls) +
                    " decode calls, " + std::to_string(s.displayed) + " frames displayed");
      }
    }
  }

  void feed(const Packet& packet) {
    int64_t timestamp = 0;
    if (__builtin_mul_overflow(packet.pts, static_cast<int64_t>(stream_.time_base_num), &timestamp)) {
      throw Error("timestamp overflow: pts " + std::to_string(packet.pts) + " x time base numerator " +
                  std::to_string(stream_.time_base_num));
    }
    const auto decode_order = static_cast<int64_t>(keyframes_.size());
    if (packet.keyframe) {
      gop_start_ = decode_order;
    } else if (gop_start_ < 0) {
      throw UnsupportedError("video does not start with a keyframe");
    }
    const PacketRecord record{decode_order, gop_start_, packet.data_start, packet.data_end, packet.keyframe};
    if (!records_.emplace(timestamp, record).second) {
      throw Error("two video packets share pts " + std::to_string(packet.pts));
    }
    keyframes_.push_back(packet.keyframe);
    session_->feed(packet.data, packet.size, timestamp);
    check_pictures();
    if (session_->stats().decode_calls != session_->stats().packets) {
      throw Error("packet at decode position " + std::to_string(decode_order) + " did not produce exactly one picture (" +
                  std::to_string(session_->stats().decode_calls) + " pictures for " + std::to_string(session_->stats().packets) +
                  " packets)");
    }
  }

  FrameIdentity account(const DecodedFrame& frame) {
    const int64_t num = stream_.time_base_num;
    if (frame.timestamp % num != 0) {
      throw Error("display timestamp " + std::to_string(frame.timestamp) + " is not a multiple of the time base numerator");
    }
    const int64_t pts = frame.timestamp / num;
    if (has_last_pts_ && pts <= last_pts_) {
      throw Error("display timestamps are not increasing: " + std::to_string(last_pts_) + " then " + std::to_string(pts));
    }
    has_last_pts_ = true;
    last_pts_ = pts;

    auto found = records_.find(frame.timestamp);
    if (found == records_.end()) {
      throw Error("displayed frame with pts " + std::to_string(pts) + " matches no packet");
    }
    const PacketRecord record = found->second;
    records_.erase(found);
    // 第一期的 GOP 结构：显示顺序里 decode_start 不减，进入新 GOP 的第一帧必须是这个 GOP 的关键帧本身
    const std::string at = " (pts " + std::to_string(pts) + ", decode position " + std::to_string(record.decode_order) + ")";
    if (record.decode_start < last_decode_start_) {
      throw UnsupportedError("a frame of an earlier GOP is displayed after a later GOP has started; not supported yet" + at);
    }
    if (record.decode_start > last_decode_start_ && record.decode_order != record.decode_start) {
      throw UnsupportedError("a frame is displayed before its GOP's keyframe (leading picture); not supported yet" + at);
    }
    last_decode_start_ = record.decode_start;
    return {next_number_++, pts, record};
  }

  // 解码器每解一张图像，核对它和对应包的关键帧标记：第一期的关键帧必须都是 IDR，不支持 CRA/BLA
  void check_pictures() {
    for (const PictureInfo& picture : session_->take_pictures()) {
      const size_t i = checked_pictures_++;
      const std::string at = " (decode position " + std::to_string(i) + ")";
      if (i >= keyframes_.size()) {
        throw Error("more decoded pictures than packets" + at);
      }
      if (picture.irap && !picture.idr) {
        throw UnsupportedError("random access picture that is not IDR (CRA/BLA, open GOP); not supported yet" + at);
      }
      if (keyframes_[i] && !picture.idr) {
        throw UnsupportedError("container keyframe is not an IDR picture; not supported yet" + at);
      }
      if (picture.idr && !keyframes_[i]) {
        throw UnsupportedError("IDR picture is not marked as a keyframe by the container; not supported yet" + at);
      }
    }
  }

  py::dict decoder_config_dict() const { return snapshot_dict(session_->snapshot()); }

  py::dict stats_dict() const {
    const DecoderStats& s = session_->stats();
    py::dict d;
    d["packets"] = s.packets;
    d["decode_calls"] = s.decode_calls;
    d["displayed"] = s.displayed;
    d["kept"] = s.kept;
    d["end_of_stream"] = s.end_of_stream;
    d["decode_status_counts"] = status_dict(s.status_counts);
    const InputStats& input = demuxer_->input_stats();
    d["bytes_read"] = input.bytes_read;
    d["seeks"] = input.seeks;
    py::list seeks;
    for (const auto& [from, to] : input.first_seeks) {
      seeks.append(py::make_tuple(from, to));
    }
    d["first_seeks"] = seeks;
    d["truncated_at"] = demuxer_->truncated_at();
    return d;
  }

  std::unique_ptr<Demuxer> demuxer_;
  StreamInfo stream_;  // 关闭后 info() 仍可用
  std::unique_ptr<NvdecSession> session_;
  bool done_ = false;
  bool failed_ = false;
  std::unordered_map<int64_t, PacketRecord> records_;  // 提交时的时间戳 → 包，显示后删除
  std::vector<bool> keyframes_;                        // 按解码顺序，每个包一项
  size_t checked_pictures_ = 0;
  int64_t gop_start_ = -1;
  int64_t last_decode_start_ = -1;
  bool has_last_pts_ = false;
  int64_t last_pts_ = 0;
  int64_t next_number_ = 0;  // 下一个显示的帧的帧号，不论是否选中
};

using Int64Array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

// 按帧号读的一次请求。packets 为 [n, 4]，按解码顺序每行是 (packet_start, packet_end, 帧号, 是否 GOP 入口)；targets 为要的帧号。
// 返回按帧号升序的 [(帧号, 画面, 解码状态), ...]
py::list read_indexed(IndexedDecoder& self, int64_t range_start, int64_t range_end, const Int64Array& packets, const Int64Array& targets) {
  if (packets.ndim() != 2 || packets.shape(1) != 4 || targets.ndim() != 1) {
    throw Error("packets must have shape [n, 4] and targets must be one-dimensional");
  }
  const auto rows = packets.unchecked<2>();
  std::vector<IndexedPacket> list(static_cast<size_t>(packets.shape(0)));
  for (py::ssize_t i = 0; i < packets.shape(0); ++i) {
    list[static_cast<size_t>(i)] = {rows(i, 0), rows(i, 1), rows(i, 2), rows(i, 3) != 0};
  }
  const std::vector<int64_t> wanted(targets.data(), targets.data() + targets.size());
  std::vector<IndexedFrame> frames;
  {
    py::gil_scoped_release release;
    frames = self.read(range_start, range_end, list, wanted);
  }
  py::list out;
  for (IndexedFrame& frame : frames) {
    out.append(py::make_tuple(frame.number, image_array(std::move(frame.image), self.output_width(), self.output_height(), self.rgb()),
                              frame.decode_status));
  }
  return out;
}

py::dict indexed_stats(const IndexedDecoder& self) {
  const IndexedStats& s = self.stats();
  py::dict d;
  d["requests"] = s.requests;
  d["bytes_read"] = s.bytes_read;
  d["read_calls"] = s.read_calls;
  d["packets"] = s.packets;
  d["decode_calls"] = s.decode_calls;
  d["displayed"] = s.displayed;
  d["kept"] = s.kept;
  d["skipped"] = s.skipped;
  d["decoder_reuses"] = s.decoder_reuses;
  d["decoder_creations"] = s.decoder_creations;
  d["decode_status_counts"] = status_dict(s.status_counts);
  d["last_range"] = s.requests == 0 ? py::object(py::none()) : py::object(py::make_tuple(s.last_range_start, s.last_range_end));
  return d;
}

py::dict decoder_cache_dict() {
  const DecoderCache& cache = DecoderCache::instance();
  const DecoderCacheStats s = cache.stats();
  py::dict d;
  d["capacity"] = cache.capacity();
  d["idle"] = cache.idle();
  d["hits"] = s.hits;
  d["misses"] = s.misses;
  d["returned"] = s.returned;
  d["evicted"] = s.evicted;
  return d;
}

}  // namespace
}  // namespace vidlo

PYBIND11_MODULE(_native, m) {
  m.doc() = "vidlo native layer";
  auto& error = py::register_exception<vidlo::Error>(m, "Error", PyExc_RuntimeError);
  py::register_exception<vidlo::UnsupportedError>(m, "UnsupportedError", error.ptr());
  py::register_exception<vidlo::TruncatedError>(m, "TruncatedError", error.ptr());

  py::class_<vidlo::Input, std::shared_ptr<vidlo::Input>>(m, "Input").def_property_readonly("size", &vidlo::Input::size);
  py::class_<vidlo::FileInput, vidlo::Input, std::shared_ptr<vidlo::FileInput>>(m, "FileInput")
      .def(py::init<const std::string&>(), py::arg("path"));
  py::class_<vidlo::StreamInput, vidlo::Input, std::shared_ptr<vidlo::StreamInput>>(m, "StreamInput")
      .def(py::init<int64_t, size_t>(), py::arg("size"), py::arg("capacity"))
      .def(
          "push",
          [](vidlo::StreamInput& self, const py::buffer& data) {
            const py::buffer_info info = data.request();
            if (info.ndim != 1 || info.itemsize != 1 || info.strides[0] != 1) {
              throw vidlo::Error("push expects contiguous bytes");
            }
            const auto* begin = static_cast<const uint8_t*>(info.ptr);
            std::vector<uint8_t> chunk(begin, begin + info.size);
            py::gil_scoped_release release;  // 队列满时在这里等，不能占着 GIL
            return self.push(std::move(chunk));
          },
          py::arg("data"))
      .def("finish", &vidlo::StreamInput::finish)
      .def("fail", &vidlo::StreamInput::fail, py::arg("reason"))
      .def("close", &vidlo::StreamInput::close)
      .def_property_readonly("pushed", &vidlo::StreamInput::pushed);

  py::class_<vidlo::OutputSpec>(m, "OutputSpec")
      .def(py::init<>())
      .def_readwrite("rgb", &vidlo::OutputSpec::rgb)
      .def_readwrite("crop_x", &vidlo::OutputSpec::crop_x)
      .def_readwrite("crop_y", &vidlo::OutputSpec::crop_y)
      .def_readwrite("crop_width", &vidlo::OutputSpec::crop_width)
      .def_readwrite("crop_height", &vidlo::OutputSpec::crop_height)
      .def_readwrite("width", &vidlo::OutputSpec::width)
      .def_readwrite("height", &vidlo::OutputSpec::height)
      .def_readwrite("colorspace", &vidlo::OutputSpec::colorspace)
      .def_readwrite("interpolation", &vidlo::OutputSpec::interpolation);

  py::class_<vidlo::SequentialDecoder>(m, "SequentialDecoder")
      // 读文件头时可能要等下载线程推数据，不能占着 GIL
      .def(py::init<std::shared_ptr<vidlo::Input>, const std::string&>(), py::arg("input"), py::arg("name"),
           py::call_guard<py::gil_scoped_release>())
      .def("start", &vidlo::SequentialDecoder::start, py::arg("device"), py::arg("output"))
      .def("info", &vidlo::SequentialDecoder::info)
      .def("add_selection", &vidlo::SequentialDecoder::add_selection, py::arg("keep"))
      .def("next_frame", &vidlo::SequentialDecoder::next_frame)
      .def("decoder_config", &vidlo::SequentialDecoder::decoder_config)
      .def("stats", &vidlo::SequentialDecoder::stats)
      .def("close", &vidlo::SequentialDecoder::close);

  py::class_<vidlo::IndexedDecoder>(m, "IndexedDecoder")
      .def(py::init([](std::shared_ptr<vidlo::Input> input, const py::bytes& codec_config, int device, const vidlo::OutputSpec& output,
                       bool skip_unreferenced) {
             const std::string_view config = codec_config;
             return std::make_unique<vidlo::IndexedDecoder>(std::move(input), std::vector<uint8_t>(config.begin(), config.end()), device,
                                                            output, skip_unreferenced);
           }),
           py::arg("input"), py::arg("codec_config"), py::arg("device"), py::arg("output"), py::arg("skip_unreferenced") = true)
      .def("read", &vidlo::read_indexed, py::arg("range_start"), py::arg("range_end"), py::arg("packets"), py::arg("targets"))
      .def("stats", &vidlo::indexed_stats)
      // 解出第一批帧之前还没有解码配置
      .def("decoder_config",
           [](const vidlo::IndexedDecoder& self) { return self.has_snapshot() ? vidlo::snapshot_dict(self.snapshot()) : py::dict(); })
      .def("close", &vidlo::IndexedDecoder::close);

  // 进程级的空闲硬件解码器缓存（按帧号读用），按设备计容量，默认 1；0 关闭并销毁现有的
  m.def("set_decoder_cache_capacity", [](size_t capacity) { vidlo::DecoderCache::instance().set_capacity(capacity); },
        py::arg("capacity"), py::call_guard<py::gil_scoped_release>());
  m.def("clear_decoder_cache", []() { vidlo::DecoderCache::instance().clear(); }, py::call_guard<py::gil_scoped_release>());
  m.def("decoder_cache_stats", &vidlo::decoder_cache_dict);
}
