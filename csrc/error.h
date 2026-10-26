#pragma once

#include <cuda.h>

#include <stdexcept>
#include <string>

namespace vidlo {

// 数据不对、API 调用失败等一般错误
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// 架构里有、但当前版本还没实现的格式或配置
class UnsupportedError : public Error {
 public:
  using Error::Error;
};

// 录像被截断（如采集异常退出，最后一段没写完）。所有完整写入的帧都已交出之后才抛出
class TruncatedError : public Error {
 public:
  using Error::Error;
};

inline std::string cu_error_text(CUresult result) {
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &text);
  return std::string(name ? name : "CUDA_ERROR_UNKNOWN") + " (" + (text ? text : "no description") + ")";
}

inline void check_cu(CUresult result, const char* what) {
  if (result != CUDA_SUCCESS) {
    throw Error(std::string(what) + " failed: " + cu_error_text(result));
  }
}

}  // namespace vidlo
