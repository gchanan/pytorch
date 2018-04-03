#pragma once

#include <stdint.h>

namespace torch {
namespace utils {

struct Device {
  const bool is_cuda;
  const int64_t device_index;
  Device(bool is_cuda, int64_t device_index): is_cuda(is_cuda), device_index(device_index) {}  
};

}
}
