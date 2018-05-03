#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at { namespace native {

static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

Tensor relu(const Tensor & self) {
  return self.clamp_min(0.0);
}

Tensor & relu_(Tensor & self) {
  return self.clamp_min_(0.0);
}

Tensor selu(const Tensor & self) {
  return at::relu(self);
}

Tensor & selu_(Tensor & self) {
  return at::relu_(self);
}

Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::relu(self);
}

Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::relu_(self);
}

}}  // namespace at::native
