#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include <iostream>

namespace at {
namespace native {

// FixMe: remove once scalars are fully supported in PyTorch.
Tensor _scalar_sum(const Tensor& self) {
  std::cerr << "called _scalar_sum" << std::endl;
  return self.sum();
}

}
}
