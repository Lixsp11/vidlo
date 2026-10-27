#pragma once

#include <cuda.h>
#include <cuviddec.h>
#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>

namespace vidlo {

// 一个硬件解码器，连同它用的 context 锁和一份 primary context 引用：解码会话结束后它可以单独存活，放进缓存给下一次请求用。
// 析构时在自己的 context 里销毁；在 fork 出的子进程里只丢掉句柄（子进程不能调用父进程的 CUDA）。
struct HardwareDecoder {
  HardwareDecoder(CUcontext context, CUdevice device, const CUVIDDECODECREATEINFO& create, const CUVIDDECODECAPS& caps);
  ~HardwareDecoder();
  HardwareDecoder(const HardwareDecoder&) = delete;
  HardwareDecoder& operator=(const HardwareDecoder&) = delete;

  CUcontext context = nullptr;
  CUdevice device = 0;
  CUVIDDECODECREATEINFO create{};  // vidLock 不记录；用作缓存键
  CUVIDDECODECAPS caps{};          // 建它时查到的能力，命中缓存时不必再查
  CUvideoctxlock lock = nullptr;
  CUvideodecoder decoder = nullptr;
  pid_t process;
  bool context_retained = false;
};

struct DecoderCacheStats {
  int64_t hits = 0;      // 取到了缓存里的解码器
  int64_t misses = 0;    // 没有可用的，调用方新建
  int64_t returned = 0;  // 请求成功后放回
  int64_t evicted = 0;   // 超过容量被挤出、销毁
};

// 进程级的空闲硬件解码器缓存，按设备分开计容量。只放请求成功、已排空、GPU 工作都结束的解码器；取出的解码器由一个会话独占。
// 建解码器（约 40 ms）、销毁（实测两百多毫秒）比解一个短样本还慢，这里省掉。torchcodec 也按设备缓存（默认 20 个），
// 本库默认每设备 1 个：b1 录像格式都一样，一个进程通常只有一个解码会话。
// 销毁放在锁外做。单例不析构：进程退出时由驱动回收，避免静态析构时 CUDA 已经卸载。fork 出的子进程用一个新的空缓存。
class DecoderCache {
 public:
  static DecoderCache& instance();

  // 取一个设备与建立参数都相同的空闲解码器；没有返回空
  std::unique_ptr<HardwareDecoder> take(CUdevice device, const CUVIDDECODECREATEINFO& create);
  // 放回；容量为 0 或放不进去时就地销毁。在会话的析构里调用，所以不抛异常
  void give(std::unique_ptr<HardwareDecoder> decoder) noexcept;
  void set_capacity(size_t capacity);  // 0 关闭缓存，并销毁现有的空闲解码器
  size_t capacity() const;
  size_t idle() const;
  void clear();
  DecoderCacheStats stats() const;

 private:
  DecoderCache() = default;
  void trim(std::list<std::unique_ptr<HardwareDecoder>>& evict);  // 持锁调用

  mutable std::mutex mutex_;
  std::list<std::unique_ptr<HardwareDecoder>> idle_;  // 前面是最近放回的
  size_t capacity_ = 1;
  DecoderCacheStats stats_;
};

}  // namespace vidlo
