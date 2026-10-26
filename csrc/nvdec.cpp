#include "nvdec.h"

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <utility>

#include "error.h"

namespace vidlo {
namespace {

// 第一个初始化 CUDA 的进程号。CUDA 不支持 fork：父进程初始化过 CUDA 后 fork 出的子进程继承了这个值，
// 与自己的进程号不同，据此在调用任何 CUDA 函数之前报错（PyTorch 初始化 CUDA 时也检查这种情况）
std::atomic<pid_t> cuda_process{0};

std::string fork_error(pid_t parent) {
  return "CUDA was initialized in process " + std::to_string(parent) + " before this process (" + std::to_string(getpid()) +
         ") was forked from it, and CUDA cannot be used after fork. Start worker processes with the 'spawn' or 'forkserver' "
         "method (for a PyTorch DataLoader, multiprocessing_context='spawn'), or open videos only inside the workers";
}

// 让当前线程在作用域内使用指定的 CUDA context
class ContextScope {
 public:
  explicit ContextScope(CUcontext context) { check_cu(cuCtxPushCurrent(context), "cuCtxPushCurrent"); }
  ~ContextScope() {
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
  }
  ContextScope(const ContextScope&) = delete;
  ContextScope& operator=(const ContextScope&) = delete;
};

std::string format_uuid(const CUuuid& uuid) {
  const auto* b = reinterpret_cast<const unsigned char*>(uuid.bytes);
  char text[64];
  std::snprintf(text, sizeof(text), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1],
                b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
  return text;
}

bool same_format(const CUVIDEOFORMAT& a, const CUVIDEOFORMAT& b) {
  return a.codec == b.codec && a.coded_width == b.coded_width && a.coded_height == b.coded_height &&
         a.chroma_format == b.chroma_format && a.bit_depth_luma_minus8 == b.bit_depth_luma_minus8 &&
         a.bit_depth_chroma_minus8 == b.bit_depth_chroma_minus8 && a.progressive_sequence == b.progressive_sequence &&
         std::memcmp(&a.display_area, &b.display_area, sizeof(a.display_area)) == 0 &&
         std::memcmp(&a.video_signal_description, &b.video_signal_description, sizeof(a.video_signal_description)) == 0;
}

}  // namespace

NvdecSession::NvdecSession(int device, cudaVideoCodec codec, unsigned clock_rate, OutputSpec output)
    : codec_(codec), process_(getpid()), processor_(std::make_unique<FrameProcessor>(std::move(output))) {
  try {
    pid_t first = 0;
    if (!cuda_process.compare_exchange_strong(first, process_) && first != process_) {
      throw Error(fork_error(first));
    }
    const CUresult init = cuInit(0);
    if (init != CUDA_SUCCESS) {
      // 父进程用别的库初始化过 CUDA 再 fork 时，上面的检查发现不了，cuInit 会失败
      throw Error("cuInit failed: " + cu_error_text(init) +
                  "; if the parent process used CUDA before forking this process, start workers with 'spawn' or 'forkserver'");
    }
    check_cu(cuDriverGetVersion(&snapshot_.driver_version), "cuDriverGetVersion");
    int count = 0;
    check_cu(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (device < 0 || device >= count) {
      throw Error("device " + std::to_string(device) + " out of range (" + std::to_string(count) + " devices)");
    }
    check_cu(cuDeviceGet(&device_, device), "cuDeviceGet");
    snapshot_.device = device;
    char name[256] = {0};
    check_cu(cuDeviceGetName(name, sizeof(name), device_), "cuDeviceGetName");
    snapshot_.device_name = name;
    CUuuid uuid{};
    check_cu(cuDeviceGetUuid(&uuid, device_), "cuDeviceGetUuid");
    snapshot_.device_uuid = format_uuid(uuid);

    // 每个进程每张卡共用 primary context
    check_cu(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
    context_retained_ = true;
    {
      ContextScope scope(context_);
      check_cu(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
      check_cu(cuvidCtxLockCreate(&lock_, context_), "cuvidCtxLockCreate");
    }

    CUVIDPARSERPARAMS params{};
    params.CodecType = codec;
    params.ulMaxNumDecodeSurfaces = 1;  // 由 sequence 回调返回实际需要的数量
    params.ulClockRate = clock_rate;
    params.ulErrorThreshold = 0;   // 码流有损坏的图像不解码，解码次数核对会发现
    params.ulMaxDisplayDelay = 1;  // 与 SDK 的 NvDecoder 默认值相同
    params.pUserData = this;
    params.pfnSequenceCallback = &NvdecSession::on_sequence;
    params.pfnDecodePicture = &NvdecSession::on_decode;
    params.pfnDisplayPicture = &NvdecSession::on_display;
    check_cu(cuvidCreateVideoParser(&parser_, &params), "cuvidCreateVideoParser");
    snapshot_.parser = params;
    snapshot_.parser.pUserData = nullptr;
    snapshot_.parser.pfnSequenceCallback = nullptr;
    snapshot_.parser.pfnDecodePicture = nullptr;
    snapshot_.parser.pfnDisplayPicture = nullptr;
  } catch (...) {
    release();
    throw;
  }
}

NvdecSession::~NvdecSession() { release(); }

void NvdecSession::check_process() const {
  if (process_ != getpid()) {
    throw Error(fork_error(process_));
  }
}

void NvdecSession::release() {
  if (process_ != getpid()) {
    // fork 出的子进程里：这些句柄属于父进程的 CUDA，子进程不能调用 CUDA 去释放，只丢掉句柄；
    // 持有显存的 FrameProcessor 也不析构（析构会调用 CUDA）
    (void)processor_.release();
    frames_.clear();
    parser_ = nullptr;
    decoder_ = nullptr;
    stream_ = nullptr;
    lock_ = nullptr;
    context_ = nullptr;
    context_retained_ = false;
    return;
  }
  if (parser_ != nullptr) {
    cuvidDestroyVideoParser(parser_);
    parser_ = nullptr;
  }
  frames_.clear();
  if (context_ != nullptr && (decoder_ != nullptr || stream_ != nullptr || processor_ != nullptr)) {
    CUcontext popped = nullptr;
    if (cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
      processor_.reset();  // 释放它的显存需要 context
      if (decoder_ != nullptr) {
        cuvidDestroyDecoder(decoder_);
      }
      if (stream_ != nullptr) {
        cuStreamDestroy(stream_);
      }
      cuCtxPopCurrent(&popped);
    }
    decoder_ = nullptr;
    stream_ = nullptr;
  }
  if (lock_ != nullptr) {
    cuvidCtxLockDestroy(lock_);
    lock_ = nullptr;
  }
  if (context_retained_) {
    cuDevicePrimaryCtxRelease(device_);
    context_retained_ = false;
  }
  context_ = nullptr;
}

void NvdecSession::feed(const uint8_t* data, size_t size, int64_t timestamp) {
  if (failed_) {
    throw Error("decoder session already failed");
  }
  if (finished_) {
    throw Error("packet fed after end of stream");
  }
  CUVIDSOURCEDATAPACKET packet{};
  // TIMESTAMP：不带时 parser 不使用 timestamp。ENDOFPICTURE：每个包正好是一张完整图像（解封装时已核对），
  // parser 不必等下一个包就解码，于是每送一个包就恰好调用一次 cuvidDecodePicture，可以逐包核对
  packet.flags = CUVID_PKT_TIMESTAMP | CUVID_PKT_ENDOFPICTURE;
  packet.payload_size = static_cast<unsigned long>(size);
  packet.payload = data;
  packet.timestamp = timestamp;
  ++stats_.packets;
  parse(packet);
}

void NvdecSession::finish() {
  if (failed_) {
    throw Error("decoder session already failed");
  }
  if (finished_) {
    return;
  }
  CUVIDSOURCEDATAPACKET packet{};
  // NOTIFY_EOS 让 parser 在交出最后一帧后再调一次显示回调（参数为空），据此确认已排空
  packet.flags = CUVID_PKT_ENDOFSTREAM | CUVID_PKT_NOTIFY_EOS;
  parse(packet);
  finished_ = true;
  if (!stats_.end_of_stream) {
    failed_ = true;
    throw Error("parser did not confirm end of stream");
  }
}

bool NvdecSession::pop(DecodedFrame& out) {
  if (failed_ || frames_.empty()) {
    return false;
  }
  out = std::move(frames_.front());
  frames_.pop_front();
  return true;
}

std::vector<PictureInfo> NvdecSession::take_pictures() { return std::exchange(pictures_, {}); }

void NvdecSession::add_selection(const std::vector<bool>& keep) { selection_.insert(selection_.end(), keep.begin(), keep.end()); }

void NvdecSession::parse(CUVIDSOURCEDATAPACKET& packet) {
  CUresult result;
  {
    ContextScope scope(context_);
    result = cuvidParseVideoData(parser_, &packet);
  }
  if (callback_error_) {
    failed_ = true;
    frames_.clear();  // 出错后不再交出任何帧
    std::exception_ptr error = callback_error_;
    callback_error_ = nullptr;
    try {
      std::rethrow_exception(error);
    } catch (const UnsupportedError&) {
      throw;
    } catch (const std::exception& e) {
      throw Error(std::string(callback_where_) + ": " + e.what());
    } catch (...) {
      throw Error(std::string(callback_where_) + ": unknown error");
    }
  }
  if (result != CUDA_SUCCESS) {
    failed_ = true;
    frames_.clear();
    check_cu(result, "cuvidParseVideoData");
  }
}

void NvdecSession::record_callback_error(const char* where) noexcept {
  callback_error_ = std::current_exception();
  callback_where_ = where;
}

// 三个回调由 cuvidParseVideoData 在调用线程上同步调用。C++ 异常不能穿过 nvcuvid 的 C 回调，
// 所以在这里全部捕获，返回 0 让 parser 停下，再由 parse() 抛出。
int CUDAAPI NvdecSession::on_sequence(void* user, CUVIDEOFORMAT* format) {
  auto* self = static_cast<NvdecSession*>(user);
  if (self->callback_error_) {
    return 0;
  }
  try {
    return self->handle_sequence(*format);
  } catch (...) {
    self->record_callback_error("sequence callback");
    return 0;
  }
}

int CUDAAPI NvdecSession::on_decode(void* user, CUVIDPICPARAMS* picture) {
  auto* self = static_cast<NvdecSession*>(user);
  if (self->callback_error_) {
    return 0;
  }
  try {
    return self->handle_decode(picture);
  } catch (...) {
    self->record_callback_error("decode callback");
    return 0;
  }
}

int CUDAAPI NvdecSession::on_display(void* user, CUVIDPARSERDISPINFO* display) {
  auto* self = static_cast<NvdecSession*>(user);
  if (self->callback_error_) {
    return 0;
  }
  try {
    return self->handle_display(display);
  } catch (...) {
    self->record_callback_error("display callback");
    return 0;
  }
}

int NvdecSession::handle_sequence(const CUVIDEOFORMAT& format) {
  ++snapshot_.sequence_calls;
  if (decoder_ != nullptr) {
    // parser 遇到参数集会再次回调；格式没变就继续，变了属于未实现的情况（不做 reconfigure）
    if (same_format(format, snapshot_.format) && format.min_num_decode_surfaces <= snapshot_.create.ulNumDecodeSurfaces) {
      return 1;
    }
    throw UnsupportedError("video format changes in the middle of the stream; not supported yet");
  }
  snapshot_.format = format;
  snapshot_.has_format = true;

  if (format.codec != codec_) {
    throw Error("parser reported codec " + std::to_string(format.codec) + ", expected " + std::to_string(codec_));
  }
  if (format.chroma_format != cudaVideoChromaFormat_420) {
    throw UnsupportedError("chroma format " + std::to_string(format.chroma_format) + " is not supported yet (only 4:2:0)");
  }
  if (format.bit_depth_luma_minus8 != 0 || format.bit_depth_chroma_minus8 != 0) {
    throw UnsupportedError("bit depth " + std::to_string(8 + format.bit_depth_luma_minus8) +
                           " is not supported yet (only 8-bit)");
  }
  if (!format.progressive_sequence) {
    throw UnsupportedError("interlaced video is not supported yet");
  }
  const auto& area = format.display_area;
  if (area.left < 0 || area.top < 0 || area.right <= area.left || area.bottom <= area.top ||
      static_cast<unsigned>(area.right) > format.coded_width || static_cast<unsigned>(area.bottom) > format.coded_height) {
    throw Error("invalid display area");
  }
  // 4:2:0 每 2×2 像素共用一份色度，显示区域的边界须落在偶数上
  if (area.left % 2 != 0 || area.top % 2 != 0 || area.right % 2 != 0 || area.bottom % 2 != 0) {
    throw UnsupportedError("display area with odd coordinates is not supported yet");
  }
  // 颜色范围与色度位置只影响 RGB 输出，在颜色转换那一步检查；原生 YUV 输出不涉及

  CUVIDDECODECAPS caps{};
  caps.eCodecType = format.codec;
  caps.eChromaFormat = format.chroma_format;
  caps.nBitDepthMinus8 = format.bit_depth_luma_minus8;
  check_cu(cuvidGetDecoderCaps(&caps), "cuvidGetDecoderCaps");
  snapshot_.caps = caps;
  if (!caps.bIsSupported) {
    throw UnsupportedError("this GPU cannot decode this codec / chroma format / bit depth");
  }
  if (!(caps.nOutputFormatMask & (1u << cudaVideoSurfaceFormat_NV12))) {
    throw UnsupportedError("this GPU cannot output NV12 for this format");
  }
  if (format.coded_width < caps.nMinWidth || format.coded_height < caps.nMinHeight || format.coded_width > caps.nMaxWidth ||
      format.coded_height > caps.nMaxHeight) {
    throw UnsupportedError("coded size " + std::to_string(format.coded_width) + "x" + std::to_string(format.coded_height) +
                           " is outside what this GPU supports");
  }
  if ((format.coded_width >> 4) * (format.coded_height >> 4) > caps.nMaxMBCount) {
    throw UnsupportedError("macroblock count exceeds what this GPU supports");
  }

  CUVIDDECODECREATEINFO create{};
  create.ulWidth = format.coded_width;
  create.ulHeight = format.coded_height;
  create.ulNumDecodeSurfaces = format.min_num_decode_surfaces;
  create.CodecType = format.codec;
  create.ChromaFormat = format.chroma_format;
  create.ulCreationFlags = cudaVideoCreate_PreferCUVID;
  create.bitDepthMinus8 = format.bit_depth_luma_minus8;
  create.ulIntraDecodeOnly = 0;
  create.ulMaxWidth = format.coded_width;
  create.ulMaxHeight = format.coded_height;
  // display_area、target_rect 保持为零：解码器输出整个编码尺寸，显示区域在后续步骤取出
  create.OutputFormat = cudaVideoSurfaceFormat_NV12;
  create.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
  create.ulTargetWidth = format.coded_width;
  create.ulTargetHeight = format.coded_height;
  create.ulNumOutputSurfaces = 2;
  create.vidLock = lock_;
  create.enableHistogram = 0;
  check_cu(cuvidCreateDecoder(&decoder_, &create), "cuvidCreateDecoder");
  snapshot_.create = create;
  snapshot_.create.vidLock = nullptr;
  if (create.ulNumDecodeSurfaces >= kSelectionReserve) {
    throw Error("decoder needs " + std::to_string(create.ulNumDecodeSurfaces) + " surfaces, more than the selection reserve");
  }
  processor_->configure(format);
  return static_cast<int>(create.ulNumDecodeSurfaces);
}

int NvdecSession::handle_decode(CUVIDPICPARAMS* picture) {
  if (decoder_ == nullptr) {
    throw Error("decode callback before the decoder was created");
  }
  PictureInfo info;
  info.intra = picture->intra_pic_flag != 0;
  if (codec_ == cudaVideoCodec_HEVC) {
    info.irap = picture->CodecSpecific.hevc.IrapPicFlag != 0;
    info.idr = picture->CodecSpecific.hevc.IdrPicFlag != 0;
  }
  pictures_.push_back(info);
  check_cu(cuvidDecodePicture(decoder_, picture), "cuvidDecodePicture");
  ++stats_.decode_calls;
  return 1;
}

int NvdecSession::handle_display(const CUVIDPARSERDISPINFO* display) {
  if (display == nullptr) {
    stats_.end_of_stream = true;
    return 1;
  }
  if (!display->progressive_frame) {
    throw UnsupportedError("interlaced frames are not supported yet");
  }
  if (selection_.empty()) {
    throw Error("no selection decision for the displayed frame (caller did not keep the selection reserve)");
  }
  const bool keep = selection_.front();
  selection_.pop_front();
  CUVIDPROCPARAMS proc{};
  proc.progressive_frame = display->progressive_frame;
  proc.second_field = 0;
  proc.top_field_first = display->top_field_first;
  proc.unpaired_field = display->repeat_first_field < 0;
  proc.output_stream = stream_;

  unsigned long long frame = 0;
  unsigned int pitch = 0;
  check_cu(cuvidMapVideoFrame64(decoder_, display->picture_index, &frame, &pitch, &proc), "cuvidMapVideoFrame64");
  // 映射成功后无论后面是否出错都要归还 surface；正常路径显式归还并检查结果
  struct MappedFrame {
    CUvideodecoder decoder;
    unsigned long long frame;
    bool mapped = true;
    CUresult unmap() {
      mapped = false;
      return cuvidUnmapVideoFrame64(decoder, frame);
    }
    ~MappedFrame() {
      if (mapped) {
        cuvidUnmapVideoFrame64(decoder, frame);
      }
    }
  } mapped{decoder_, frame};

  // 映射会等这帧解码完成，所以在映射之后查状态，避免查到 InProgress
  CUVIDGETDECODESTATUS status{};
  check_cu(cuvidGetDecodeStatus(decoder_, display->picture_index, &status), "cuvidGetDecodeStatus");
  ++stats_.status_counts[static_cast<unsigned>(status.decodeStatus) & 15u];
  if (status.decodeStatus != cuvidDecodeStatus_Success) {
    throw Error("frame with timestamp " + std::to_string(display->timestamp) + " has decode status " +
                std::to_string(status.decodeStatus));
  }

  DecodedFrame out;
  out.timestamp = display->timestamp;
  out.decode_status = status.decodeStatus;
  out.kept = keep;
  if (keep) {
    // 色度紧跟在整个 surface 的亮度之后：surface 高度是编码高度（b1 为 736），不是显示高度
    const auto surface_height = static_cast<unsigned>((snapshot_.create.ulTargetHeight + 1) & ~1ul);
    try {
      processor_->copy_in(frame, pitch, surface_height, stream_);
    } catch (...) {
      cuStreamSynchronize(stream_);  // 复制可能还在读 surface：等它结束再归还
      throw;
    }
  }
  // 选中的帧已复制到自有显存，没选中的帧只检查状态：都可以归还 surface 了
  check_cu(mapped.unmap(), "cuvidUnmapVideoFrame64");
  if (keep) {
    out.image = processor_->produce(stream_);
    ++stats_.kept;
  }

  frames_.push_back(std::move(out));
  ++stats_.displayed;
  return 1;
}

}  // namespace vidlo
