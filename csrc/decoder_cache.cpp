#include "decoder_cache.h"

#include <pthread.h>
#include <unistd.h>

#include <iterator>
#include <utility>

#include "error.h"

namespace vidlo {
namespace {

template <typename Rect>
bool same_rect(const Rect& a, const Rect& b) {
  return a.left == b.left && a.top == b.top && a.right == b.right && a.bottom == b.bottom;
}

// 建立参数逐项相同才能复用（vidLock 除外）
bool same_create(const CUVIDDECODECREATEINFO& a, const CUVIDDECODECREATEINFO& b) {
  return a.ulWidth == b.ulWidth && a.ulHeight == b.ulHeight && a.ulNumDecodeSurfaces == b.ulNumDecodeSurfaces &&
         a.CodecType == b.CodecType && a.ChromaFormat == b.ChromaFormat && a.ulCreationFlags == b.ulCreationFlags &&
         a.bitDepthMinus8 == b.bitDepthMinus8 && a.ulIntraDecodeOnly == b.ulIntraDecodeOnly && a.ulMaxWidth == b.ulMaxWidth &&
         a.ulMaxHeight == b.ulMaxHeight && same_rect(a.display_area, b.display_area) && a.OutputFormat == b.OutputFormat &&
         a.DeinterlaceMode == b.DeinterlaceMode && a.ulTargetWidth == b.ulTargetWidth && a.ulTargetHeight == b.ulTargetHeight &&
         a.ulNumOutputSurfaces == b.ulNumOutputSurfaces && same_rect(a.target_rect, b.target_rect) &&
         a.enableHistogram == b.enableHistogram;
}

}  // namespace

HardwareDecoder::HardwareDecoder(CUcontext context_in, CUdevice device_in, const CUVIDDECODECREATEINFO& create_in,
                                 const CUVIDDECODECAPS& caps_in)
    : context(context_in), device(device_in), create(create_in), caps(caps_in), process(getpid()) {
  create.vidLock = nullptr;
  try {
    CUcontext retained = nullptr;
    check_cu(cuDevicePrimaryCtxRetain(&retained, device), "cuDevicePrimaryCtxRetain");
    context_retained = true;
    check_cu(cuvidCtxLockCreate(&lock, context), "cuvidCtxLockCreate");
    CUVIDDECODECREATEINFO params = create;
    params.vidLock = lock;
    // 调用方（parser 的 sequence 回调）已压入这个 context
    check_cu(cuvidCreateDecoder(&decoder, &params), "cuvidCreateDecoder");
  } catch (...) {
    if (lock != nullptr) {
      cuvidCtxLockDestroy(lock);
    }
    if (context_retained) {
      cuDevicePrimaryCtxRelease(device);
    }
    throw;
  }
}

HardwareDecoder::~HardwareDecoder() {
  if (process != getpid()) {
    return;
  }
  if (decoder != nullptr && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
    cuvidDestroyDecoder(decoder);
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
  }
  if (lock != nullptr) {
    cuvidCtxLockDestroy(lock);
  }
  if (context_retained) {
    cuDevicePrimaryCtxRelease(device);
  }
}

namespace {

DecoderCache* cache = nullptr;

}  // namespace

DecoderCache& DecoderCache::instance() {
  static std::once_flag created;
  std::call_once(created, [] {
    cache = new DecoderCache();
    // fork 出的子进程换一个空缓存：继承来的互斥锁可能正被父进程别的线程持有（子进程里没人能解开），
    // 继承来的解码器属于父进程的 CUDA。旧的不碰、不析构。子进程刚 fork 时只有一个线程
    pthread_atfork(nullptr, nullptr, [] { cache = new DecoderCache(); });
  });
  return *cache;
}

std::unique_ptr<HardwareDecoder> DecoderCache::take(CUdevice device, const CUVIDDECODECREATEINFO& create) {
  std::lock_guard<std::mutex> guard(mutex_);
  for (auto it = idle_.begin(); it != idle_.end(); ++it) {
    if ((*it)->process == getpid() && (*it)->device == device && same_create((*it)->create, create)) {
      std::unique_ptr<HardwareDecoder> found = std::move(*it);
      idle_.erase(it);
      ++stats_.hits;
      return found;
    }
  }
  ++stats_.misses;
  return nullptr;
}

// 每个设备只留最前面（最近放回）的 capacity_ 个，其余用 splice 挪进 evict：不分配内存，不会中途失败
void DecoderCache::trim(std::list<std::unique_ptr<HardwareDecoder>>& evict) {
  for (auto it = idle_.begin(); it != idle_.end();) {
    const auto next = std::next(it);
    size_t same = 0;
    for (auto earlier = idle_.begin(); earlier != it; ++earlier) {
      same += (*earlier)->device == (*it)->device;
    }
    if (same >= capacity_) {
      evict.splice(evict.end(), idle_, it);
      ++stats_.evicted;
    }
    it = next;
  }
}

void DecoderCache::give(std::unique_ptr<HardwareDecoder> decoder) noexcept {
  std::list<std::unique_ptr<HardwareDecoder>> evict;
  try {
    std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.returned;
    if (capacity_ == 0) {
      ++stats_.evicted;
    } else {
      idle_.push_front(std::move(decoder));  // 分配失败时 decoder 仍归参数所有，下面随参数销毁
      trim(evict);
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch) - 放不回就销毁，不让异常逃出会话的析构
  }
  // 没放进缓存的 decoder 和 evict 在函数返回时析构，销毁不占着锁
}

void DecoderCache::set_capacity(size_t capacity) {
  std::list<std::unique_ptr<HardwareDecoder>> evict;
  std::lock_guard<std::mutex> guard(mutex_);
  capacity_ = capacity;
  trim(evict);
  // guard 先于 evict 析构（后声明先析构）：销毁不占着锁
}

size_t DecoderCache::capacity() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return capacity_;
}

size_t DecoderCache::idle() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return idle_.size();
}

void DecoderCache::clear() {
  std::list<std::unique_ptr<HardwareDecoder>> evict;
  std::lock_guard<std::mutex> guard(mutex_);
  stats_.evicted += static_cast<int64_t>(idle_.size());
  evict.splice(evict.end(), idle_);
  // guard 先于 evict 析构：销毁不占着锁
}

DecoderCacheStats DecoderCache::stats() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return stats_;
}

}  // namespace vidlo
