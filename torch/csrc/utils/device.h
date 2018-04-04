#pragma once

#include <cstdint>

namespace torch {

enum class DeviceType {CPU=0, CUDA=1};

struct Device {
  DeviceType device_type;
  int device_index;
  bool is_default;  // is default device for type.
  Device(DeviceType device_type, int64_t device_index, bool is_default);
};

}
