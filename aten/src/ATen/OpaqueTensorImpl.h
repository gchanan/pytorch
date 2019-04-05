#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {
struct CAFFE2_API OpaqueTensorImpl : public TensorImpl {
  // An "Opaque" TensorImpl -- there are no strides, no pointer-arithmetic, etc.
  // For now, even data() is not supported, because code in PyTorch calls that and
  // then does pointer arithmetic.

public:
  // Public for now...
  explicit OpaqueTensorImpl(at::TensorTypeId, const caffe2::TypeMeta&, c10::Device);

  IntArrayRef strides() const override;
  bool is_contiguous() const override;
  int64_t stride(int64_t d) const override;
  void resize_dim(int64_t ndim) override;
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;

  TensorImpl* maybe_zero_dim(bool condition_when_zero_dim) override;
  bool has_storage() const override;
  const Storage& storage() const override;
  int64_t storage_offset() const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach() const override;
};

} // namespace at
