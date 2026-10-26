#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <vector>

namespace vidlo {

// 顺序扫描 Matroska 的 EBML 结构，得到每个 Block 的真实边界，用来逐个核对 libavformat 交出的包，
// 并发现文件截断和被跳过的块。与 libavformat 读同一条字节流，不另读文件。
// 截断（元素声明的长度超出文件，或文件结束在元素头中间）不当作错误，只记下位置：不完整的块不交出，
// 之前完整的块照常核对，由调用方在交出全部完整的帧之后报告。
class MkvScanner {
 public:
  struct Block {
    int64_t content_start = 0;  // Block 内容（块头）起点，等于 libavformat 给出的 pos
    int64_t content_end = 0;    // 元素声明的结束位置
    int64_t track = 0;
    uint8_t flags = 0;
    int header_length = 0;  // 轨道号 VINT + 2 字节时间码 + 1 字节 flags
    bool in_block_group = false;
  };

  // 从 position 起取 size 个字节；拿不到（还没读到）时返回 false
  using ReadBytes = std::function<bool(int64_t position, uint8_t* out, size_t size)>;

  explicit MkvScanner(int64_t file_size) : file_size_(file_size) {}

  // 用 [0, available_end) 中已经读到的字节尽量往前扫
  void advance(const ReadBytes& read, int64_t available_end);
  // 文件读完：结构必须正好在文件末尾结束，或者停在截断处。available_end 是实际读到的位置
  void finish(int64_t available_end);

  int64_t position() const { return position_; }
  // 从这个位置起数据不完整或缺失：不完整的块或元素头的起点；里面的元素都完整、只是外层声明得更长时为文件末尾。
  // 没有截断时为 -1
  int64_t truncated_at() const { return truncated_at_; }
  std::deque<Block>& blocks() { return blocks_; }

 private:
  struct Master {
    uint32_t id;
    int64_t end;  // -1 表示长度未知
  };

  int64_t file_size_;
  int64_t position_ = 0;
  int64_t truncated_at_ = -1;
  bool outer_incomplete_ = false;  // 有外层元素（Segment、Cluster、BlockGroup）声明的长度超出文件
  bool stopped_ = false;           // 停在不完整的元素处，不再往后扫
  std::vector<Master> masters_;
  std::deque<Block> blocks_;  // 扫到、还没与包对上的 Block（所有轨道）
};

}  // namespace vidlo
