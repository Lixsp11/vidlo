#include "mkv_scanner.h"

#include <algorithm>
#include <string>

#include "error.h"

namespace vidlo {
namespace {

constexpr uint32_t kSegment = 0x18538067;
constexpr uint32_t kCluster = 0x1F43B675;
constexpr uint32_t kBlockGroup = 0xA0;
constexpr uint32_t kBlock = 0xA1;
constexpr uint32_t kSimpleBlock = 0xA3;

// Segment 下一级的元素：长度未知的 Cluster 遇到它们就结束
bool is_top_level(uint32_t id) {
  switch (id) {
    case kCluster:
    case 0x114D9B74:  // SeekHead
    case 0x1549A966:  // Info
    case 0x1654AE6B:  // Tracks
    case 0x1C53BB6B:  // Cues
    case 0x1254C367:  // Tags
    case 0x1043A770:  // Chapters
    case 0x1941A469:  // Attachments
      return true;
    default:
      return false;
  }
}

// EBML 变长整数的字节数，由首字节前导零个数决定；0 表示无效
int vint_length(uint8_t first) {
  for (int length = 1; length <= 8; ++length) {
    if (first & (0x80 >> (length - 1))) {
      return length;
    }
  }
  return 0;
}

std::string at(int64_t position) { return " (byte " + std::to_string(position) + ")"; }

}  // namespace

void MkvScanner::advance(const ReadBytes& read, int64_t available_end) {
  uint8_t head[12];
  while (!stopped_) {
    while (!masters_.empty() && masters_.back().end >= 0 && position_ >= masters_.back().end) {
      if (position_ > masters_.back().end) {
        throw Error("Matroska element overruns its parent" + at(position_));
      }
      masters_.pop_back();
    }
    if (position_ >= file_size_) {
      return;
    }
    // 元素头：ID（1～4 字节，保留长度标记位）+ 长度（1～8 字节）。字节还没读到就等下一次
    if (position_ + 1 > available_end || !read(position_, head, 1)) {
      return;
    }
    const int id_length = vint_length(head[0]);
    if (id_length == 0 || id_length > 4) {
      throw Error("invalid Matroska element ID" + at(position_));
    }
    if (position_ + id_length + 1 > available_end || !read(position_, head, static_cast<size_t>(id_length) + 1)) {
      return;
    }
    const int size_length = vint_length(head[id_length]);
    if (size_length == 0) {
      throw Error("invalid Matroska element size" + at(position_));
    }
    const int header_length = id_length + size_length;
    if (position_ + header_length > available_end || !read(position_, head, static_cast<size_t>(header_length))) {
      return;
    }
    uint32_t id = 0;
    for (int i = 0; i < id_length; ++i) {
      id = (id << 8) | head[i];
    }
    uint64_t size = head[id_length] & (0xFF >> size_length);
    bool unknown_size = size == static_cast<uint64_t>(0xFF >> size_length);
    for (int i = 1; i < size_length; ++i) {
      size = (size << 8) | head[id_length + i];
      unknown_size = unknown_size && head[id_length + i] == 0xFF;
    }
    const int64_t content_start = position_ + header_length;

    if (!masters_.empty() && masters_.back().end < 0 && masters_.back().id == kCluster && is_top_level(id)) {
      masters_.pop_back();
    }
    int64_t end = -1;
    if (!unknown_size) {
      if (size > static_cast<uint64_t>(file_size_ - content_start)) {
        // 声明的长度超出文件：文件被截断。外层元素按文件末尾结束继续扫；不完整的块等元素到此为止
        if (id != kSegment && id != kCluster && id != kBlockGroup) {
          truncated_at_ = position_;
          stopped_ = true;
          return;
        }
        outer_incomplete_ = true;
        size = static_cast<uint64_t>(file_size_ - content_start);
      }
      end = content_start + static_cast<int64_t>(size);
      if (!masters_.empty() && masters_.back().end >= 0 && end > masters_.back().end) {
        throw Error("Matroska element overruns its parent" + at(position_));
      }
    } else if (id != kSegment && id != kCluster) {
      throw UnsupportedError("Matroska element with unknown size is not supported yet" + at(position_));
    }

    if (id == kSegment || id == kCluster || id == kBlockGroup) {
      masters_.push_back({id, end});
      position_ = content_start;
      continue;
    }
    if (id == kSimpleBlock || id == kBlock) {
      const uint32_t parent = masters_.empty() ? 0 : masters_.back().id;
      if ((id == kSimpleBlock && parent != kCluster) || (id == kBlock && parent != kBlockGroup)) {
        throw Error("Matroska block in an unexpected place" + at(position_));
      }
      // 块头：轨道号 VINT + 2 字节相对时间码 + 1 字节 flags
      uint8_t block_head[11];
      if (content_start + 1 > available_end || !read(content_start, block_head, 1)) {
        return;
      }
      const int track_length = vint_length(block_head[0]);
      if (track_length == 0) {
        throw Error("invalid track number in Matroska block" + at(content_start));
      }
      const int block_header_length = track_length + 3;
      if (end - content_start < block_header_length) {
        throw Error("Matroska block shorter than its header" + at(content_start));
      }
      if (content_start + block_header_length > available_end ||
          !read(content_start, block_head, static_cast<size_t>(block_header_length))) {
        return;
      }
      Block block;
      block.content_start = content_start;
      block.content_end = end;
      block.track = block_head[0] & (0xFF >> track_length);
      for (int i = 1; i < track_length; ++i) {
        block.track = (block.track << 8) | block_head[i];
      }
      block.flags = block_head[track_length + 2];
      block.header_length = block_header_length;
      block.in_block_group = id == kBlock;
      blocks_.push_back(block);
      position_ = end;
      continue;
    }
    // 其余元素（EBML 头、SeekHead、Info、Tracks、Cues、Tags、Void、Cluster 时间码等）整个跳过
    position_ = end;
  }
}

void MkvScanner::finish(int64_t available_end) {
  if (stopped_) {
    return;
  }
  if (position_ == file_size_) {
    if (outer_incomplete_) {
      truncated_at_ = file_size_;  // 里面的元素都完整，但外层声明的内容还没写完
    }
    return;
  }
  if (available_end == file_size_) {
    // 整个文件都读过，扫描却停在一个元素（头或块头）中间：文件结束在这个元素里，即被截断
    truncated_at_ = position_;
    stopped_ = true;
    return;
  }
  throw Error("Matroska structure was scanned up to byte " + std::to_string(position_) + " but only " +
              std::to_string(available_end) + " of " + std::to_string(file_size_) + " bytes were read");
}

}  // namespace vidlo
