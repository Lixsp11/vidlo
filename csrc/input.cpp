#include "input.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

#include "error.h"

namespace vidlo {

FileInput::FileInput(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0) {
    throw Error("cannot open " + path + ": " + std::strerror(errno));
  }
  struct stat st {};
  if (::fstat(fd_, &st) != 0) {
    const int err = errno;
    ::close(fd_);
    throw Error("cannot stat " + path + ": " + std::strerror(err));
  }
  size_ = static_cast<int64_t>(st.st_size);
}

FileInput::~FileInput() {
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

size_t FileInput::read(int64_t position, uint8_t* out, size_t size) {
  ssize_t n;
  do {
    n = ::pread(fd_, out, size, position);
  } while (n < 0 && errno == EINTR);
  if (n < 0) {
    throw Error("read at byte " + std::to_string(position) + " failed: " + std::strerror(errno));
  }
  return static_cast<size_t>(n);
}

StreamInput::StreamInput(int64_t size, size_t capacity)
    : size_(size), capacity_(capacity), process_(getpid()), sync_(std::make_unique<Sync>()), mutex_(sync_->mutex),
      changed_(sync_->changed) {
  if (size < 0 || capacity == 0) {
    throw Error("invalid stream size or capacity");
  }
}

StreamInput::~StreamInput() {
  if (process_ != getpid()) {
    (void)sync_.release();
  }
}

size_t StreamInput::read(int64_t position, uint8_t* out, size_t size) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (position != position_) {
    throw UnsupportedError("streamed input can only be read in order: the demuxer asked for byte " + std::to_string(position) +
                           " while the stream is at byte " + std::to_string(position_) + "; seeking is not supported yet");
  }
  if (size == 0) {
    return 0;
  }
  // 到了末尾也要等终态：字节齐全不等于下载成功（响应可能在最后才报校验错误）
  changed_.wait(lock, [&] { return queued_ > 0 || finished_ || !failure_.empty() || closed_; });
  if (queued_ == 0) {
    if (!failure_.empty()) {
      throw Error("download failed at byte " + std::to_string(position_) + " of " + std::to_string(size_) + ": " + failure_);
    }
    if (closed_) {
      throw Error("input was closed");
    }
    return 0;  // finish() 只在字节齐全时成立，这里 position_ 等于 size_
  }
  size_t done = 0;
  while (done < size && !chunks_.empty()) {
    std::vector<uint8_t>& front = chunks_.front();
    const size_t n = std::min(size - done, front.size() - front_offset_);
    std::memcpy(out + done, front.data() + front_offset_, n);
    done += n;
    front_offset_ += n;
    if (front_offset_ == front.size()) {
      chunks_.pop_front();
      front_offset_ = 0;
    }
  }
  queued_ -= done;
  position_ += static_cast<int64_t>(done);
  changed_.notify_all();
  return done;
}

void StreamInput::confirm_end() {
  std::unique_lock<std::mutex> lock(mutex_);
  changed_.wait(lock, [&] { return finished_ || !failure_.empty() || closed_; });
  if (!failure_.empty()) {
    throw Error("download failed: " + failure_);
  }
  if (closed_) {
    throw Error("input was closed");
  }
  if (position_ != size_) {
    throw Error("the demuxer stopped at byte " + std::to_string(position_) + " of " + std::to_string(size_));
  }
}

bool StreamInput::push(std::vector<uint8_t> chunk) {
  std::unique_lock<std::mutex> lock(mutex_);
  // 一块比上限还大时，等队列空了再放进去
  changed_.wait(lock, [&] {
    return closed_ || finished_ || !failure_.empty() || queued_ == 0 || queued_ + chunk.size() <= capacity_;
  });
  if (closed_) {
    return false;
  }
  if (finished_ || !failure_.empty()) {
    throw Error("data pushed after the stream was finished or failed");
  }
  if (pushed_ + static_cast<int64_t>(chunk.size()) > size_) {
    throw Error("download delivered more bytes than the object size " + std::to_string(size_));
  }
  if (chunk.empty()) {
    return true;
  }
  pushed_ += static_cast<int64_t>(chunk.size());
  queued_ += chunk.size();
  chunks_.push_back(std::move(chunk));
  changed_.notify_all();
  return true;
}

void StreamInput::finish() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_ || !failure_.empty()) {
    return;
  }
  if (pushed_ != size_) {
    failure_ = "download finished after " + std::to_string(pushed_) + " of " + std::to_string(size_) + " bytes";
  } else {
    finished_ = true;
  }
  changed_.notify_all();
}

void StreamInput::fail(const std::string& reason) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_ || !failure_.empty()) {
    return;  // 只记第一个终态
  }
  failure_ = reason.empty() ? "unknown error" : reason;
  changed_.notify_all();
}

void StreamInput::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  closed_ = true;
  chunks_.clear();
  queued_ = 0;
  changed_.notify_all();
}

int64_t StreamInput::pushed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pushed_;
}

}  // namespace vidlo
