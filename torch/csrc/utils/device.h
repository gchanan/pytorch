#pragma once

#include "stdint.h"

namespace torch {

enum class DeviceType {CPU=0, CUDA=1};

struct Device {
  DeviceType device_type;
  int device_index{-1};
  bool is_default;
  Device(DeviceType device_type, int64_t device_index, bool is_default);
};

}
