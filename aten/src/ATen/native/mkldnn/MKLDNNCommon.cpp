#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#if AT_MKLDNN_ENABLED()

#include <ideep.hpp>

namespace at { namespace native {

// Custom allocator using c10 CPU allocator for `ideep::tensor`
struct AllocForMKLDNN {
  template<class computation_t = void>
  static char* malloc(size_t size) {
    auto allocator = c10::GetAllocator(c10::DeviceType::CPU);
    return (char*)allocator->raw_allocate(size);
  }

  template<class computation_t = void>
  static void free(void* p) {
    auto allocator = c10::GetAllocator(c10::DeviceType::CPU);
    allocator->raw_deallocate(p);
  }
};

Tensor new_with_itensor_mkldnn(ideep::tensor&& it, const TensorOptions& options) {
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  auto dims = it.get_dims();
  c10::intrusive_ptr<OpaqueHandle<ideep::tensor>> handle =
    c10::make_intrusive<OpaqueHandle<ideep::tensor> >(std::move(it));
  return detail::make_tensor<OpaqueTensorImpl<c10::intrusive_ptr<OpaqueHandle<ideep::tensor>>>>(
    MkldnnCPUTensorId(), options.dtype(), options.device(), handle, std::vector<int64_t>(dims.begin(), dims.end()));
}

Tensor new_with_sizes_mkldnn(IntArrayRef sizes, const TensorOptions& options) {
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  ideep::tensor::dims dst_dims (sizes.begin(), sizes.end());
  ideep::tensor it;
  it.resize<AllocForMKLDNN>(dst_dims, ideep::tensor::data_type::f32);
  return new_with_itensor_mkldnn(std::move(it), options);
}

using MKLDNNTensor = Tensor;

ideep::tensor& itensor_from_mkldnn(const MKLDNNTensor& mkldnn_tensor) {
  AT_ASSERTM(mkldnn_tensor.type_id() == MkldnnCPUTensorId(),
             "mkldnn_to_dense expects MKL-DNN tensor input");
  AT_ASSERTM(!mkldnn_tensor.is_variable(), "_internal_get_OpaqueTensorImpl: should not be a variable");
  OpaqueTensorImpl<c10::intrusive_ptr<OpaqueHandle<ideep::tensor>>> *oti =
    static_cast<OpaqueTensorImpl<c10::intrusive_ptr<OpaqueHandle<ideep::tensor>>> *>(mkldnn_tensor.unsafeGetTensorImpl());
  return oti->unsafe_opaque_handle()->get_handle();
}

}}

#endif // AT_MKLDNN_ENABLED()
