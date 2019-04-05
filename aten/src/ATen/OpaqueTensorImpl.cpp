#include <ATen/ATen.h>
#include <ATen/OpaqueTensorImpl.h>

namespace at {

OpaqueTensorImpl::OpaqueTensorImpl(at::TensorTypeId type_id, const caffe2::TypeMeta& data_type,
                                   c10::Device device, c10::intrusive_ptr<c10::intrusive_ptr_target> opaque_handle,
                                   IntArrayRef sizes)
  :   TensorImpl(type_id, data_type, device, false),
      opaque_handle_(std::move(opaque_handle))
{
  sizes_ = sizes.vec();
  refresh_numel();
}

IntArrayRef OpaqueTensorImpl::strides() const {
  AT_ERROR("opaque tensors do not have strides");
}
bool OpaqueTensorImpl::is_contiguous() const {
  AT_ERROR("opaque tensors do not have is_contiguous");
}
int64_t OpaqueTensorImpl::stride(int64_t d) const {
  AT_ERROR("opaque tensors do not have strides");
}
void OpaqueTensorImpl::resize_dim(int64_t ndim) {
  AT_ERROR("opaque tensors do not have resize_dim");
}
void OpaqueTensorImpl::set_size(int64_t dim, int64_t new_size) {
  AT_ERROR("opaque tensors do not have set_size");
}
void OpaqueTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  AT_ERROR("opaque tensors do not have set_stride");
}
void OpaqueTensorImpl::set_storage_offset(int64_t storage_offset) {
  AT_ERROR("opaque tensors do not have set_storage_offset");
}

TensorImpl* OpaqueTensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
    AT_ERROR("opaque tensors do not support maybe_zero_dim");
}

bool OpaqueTensorImpl::has_storage() const {
  return false;
}
const Storage& OpaqueTensorImpl::storage() const {
  AT_ERROR("opaque tensors do not have storage");
}
int64_t OpaqueTensorImpl::storage_offset() const {
  AT_ERROR("opaque tensors do not have storage");
}

// NOTE: `shallow_copy_and_detach()` does not copy the AutogradMeta pointer
// because it is unique for each Variable.
// NOTE: We don't set `allow_tensor_metadata_change_` to false here, because there are call sites
// to this function that need to change the shallow copy's size or storage afterwards, and setting
// `allow_tensor_metadata_change_` to false would prevent those changes from happening and is
// undesirable.
c10::intrusive_ptr<TensorImpl> OpaqueTensorImpl::shallow_copy_and_detach() const {
  auto impl = c10::make_intrusive<OpaqueTensorImpl>(type_id(), dtype(), device(), opaque_handle_, sizes_);
  // TensorImpl general fields
  // Note that some of these fields are not used in opaque tensor code,
  // and we copy them here only for completeness.
  impl->sizes_ = sizes_;
  impl->strides_ = strides_;
  impl->storage_offset_ = storage_offset_;
  impl->is_contiguous_ = is_contiguous_;
  impl->is_wrapped_number_ = is_wrapped_number_;
  impl->reserved_ = reserved_;

  // OpaqueTensorImpl-specific fields (none currently).
  return impl;
}

} // namespace at
