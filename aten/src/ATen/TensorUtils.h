#pragma once

#include "ATen/Tensor.h"
#include "ATen/TensorInfo.h"

namespace at {
namespace tensorutils {

bool overlappingIndices(const at::Tensor& t);
bool canUse32BitIndexMath(const at::Tensor &t, ptrdiff_t max_elem=UINT32_MAX);

template <typename ScalarType, typename IndexType>
TensorInfo<ScalarType, IndexType>
getTensorInfo(const at::Tensor& t) {
  IndexType sz[MAX_TENSORINFO_DIMS];
  IndexType st[MAX_TENSORINFO_DIMS];

  int dims = t.dim();
  for (int i = 0; i < dims; ++i) {
    sz[i] = t.size(i);
    st[i] = t.stride(i);
  }

  return TensorInfo<ScalarType, IndexType>(
    t.data<ScalarType>(), dims, sz, st);
}

} // tensorutils
} // at
