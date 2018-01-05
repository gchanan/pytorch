#pragma once

#include "ATen/TensorImpl.h"
#include <sstream>

namespace at {

static inline int64_t maybe_wrap_dim(int64_t dim, int64_t dim_post_expr, bool wrap_scalar=true) {
  std::cerr << "Dim " << dim << " " << dim_post_expr << std::endl;
  if (dim_post_expr <= 0) {
    if (!wrap_scalar || (dim < -1 || dim > 0)) {
      std::ostringstream oss;
      oss << "dimension out of range (expected to be in range of [-1, 0], but got " << dim << ")";
      throw std::runtime_error(oss.str());
    }
    return 0;
  } else if (dim < -(dim_post_expr) || dim >= (dim_post_expr)) {
    std::ostringstream oss;
    oss << "dimension2 out of range (expected to be in range of [" << -(dim_post_expr)
        << ", " << (dim_post_expr)-1 << "], but got " << dim << ")",
    throw std::runtime_error(oss.str());
  }
  if (dim < 0) dim += dim_post_expr;
  return dim;
}

static inline int64_t maybe_wrap_dim(int64_t dim, TensorImpl *tensor) {
  return maybe_wrap_dim(dim, tensor->dim());
}

static inline int64_t maybe_wrap_dim(int64_t dim, TensorList tensors) {
  if (tensors.size() == 0) {
    // can't wrap empty TensorList; rely on underlying implementation to throw error if necessary.
    return dim;
  }
  return maybe_wrap_dim(dim, tensors[0].dim());
}

}
