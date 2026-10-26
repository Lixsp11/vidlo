#pragma once

#include <sys/types.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vidlo {

// 录像字节的来源，libavformat 的读回调经它取数据
class Input {
 public:
  virtual ~Input() = default;
  virtual int64_t size() const = 0;
  // 从 position 起读至多 size 个字节；返回读到的字节数，0 表示已到末尾；出错抛异常
  virtual size_t read(int64_t position, uint8_t* out, size_t size) = 0;
  // 读完全部字节后确认来源也成功结束（例如下载的响应校验通过）；没有成功结束就抛异常
  virtual void confirm_end() {}
};

// 本地文件，任意位置都能读
class FileInput : public Input {
 public:
  explicit FileInput(const std::string& path);
  ~FileInput() override;
  FileInput(const FileInput&) = delete;
  FileInput& operator=(const FileInput&) = delete;

  int64_t size() const override { return size_; }
  size_t read(int64_t position, uint8_t* out, size_t size) override;

 private:
  int fd_ = -1;
  int64_t size_ = 0;
};

// 由另一个线程按顺序推入的字节流（对象存储的流式下载），只能从头按顺序读。
// 队列里的字节有上限：满了 push 就等，读取方取走后再继续，下载不会远远跑在解码前面。
// 结束只有两种终态：finish（字节齐全且响应校验通过）或 fail；读到末尾时要等到终态才算读完。
class StreamInput : public Input {
 public:
  StreamInput(int64_t size, size_t capacity);
  ~StreamInput() override;
  StreamInput(const StreamInput&) = delete;
  StreamInput& operator=(const StreamInput&) = delete;

  int64_t size() const override { return size_; }
  size_t read(int64_t position, uint8_t* out, size_t size) override;
  void confirm_end() override;

  // 下载方调用。push 在队列满时等待；读取方已关闭时返回 false，下载方应停止；已到终态后再 push 抛异常
  bool push(std::vector<uint8_t> chunk);
  void finish();                         // 全部字节都已推入且校验通过；字节不齐时按失败处理
  void fail(const std::string& reason);  // 下载失败，读取方读到这里时报错
  // 读取方调用：不再读取，唤醒等待中的下载方
  void close();

  int64_t pushed() const;

 private:
  // 锁和条件变量放在堆上：fork 出的子进程继承了父进程下载线程正在等的条件变量，而 glibc 的 pthread_cond_destroy
  // 在还记着等待者时会一直等（glibc 2.35 nptl/pthread_cond_destroy.c），那个线程在子进程里并不存在，
  // 所以子进程里析构时直接丢掉它们、不销毁
  struct Sync {
    std::mutex mutex;
    std::condition_variable changed;
  };

  const int64_t size_;
  const size_t capacity_;
  const pid_t process_;  // 建这个对象的进程
  std::unique_ptr<Sync> sync_;
  std::mutex& mutex_;
  std::condition_variable& changed_;
  std::deque<std::vector<uint8_t>> chunks_;
  size_t front_offset_ = 0;  // chunks_.front() 里已读走的字节数
  size_t queued_ = 0;        // 队列里还没读走的字节数
  int64_t pushed_ = 0;       // 已推入的总字节数
  int64_t position_ = 0;     // 下一个要读的位置
  bool finished_ = false;
  bool closed_ = false;
  std::string failure_;
};

}  // namespace vidlo
