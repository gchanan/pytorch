#include <c10/core/OpaqueHandle.h>
#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Allocator.h>
#include "MKLDNNCommon.h"

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

// Helper function to construct a `Storage` given allocated `ideep::tensor`.
// The storage does not own the buffer. The assumption is that there would be
// no reallocation from `ideep::tensor` later.
c10::Storage new_with_itensor_storage(const ideep::tensor& it, const TensorOptions& options) {
  c10::DataPtr data_ptr(it.get_data_handle(), c10::DeviceType::CPU);
  return c10::Storage(
    options.dtype(), it.get_size() / options.dtype().itemsize(),
    std::move(data_ptr), /*allocator=*/nullptr, /*resizeable=*/false);
}

Tensor new_with_sizes_mkldnn(IntArrayRef sizes, const TensorOptions& options) {
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  ideep::tensor::dims dst_dims (sizes.begin(), sizes.end());
  ideep::tensor it;
  it.resize<AllocForMKLDNN>(dst_dims, ideep::tensor::data_type::f32);
  return new_with_itensor_mkldnn(std::move(it), options);
}

Tensor new_with_itensor_mkldnn(ideep::tensor&& it, const TensorOptions& options) {
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  auto dims = it.get_dims();
  //c10::Storage storage(new_with_itensor_storage(it, options));
  //at::TensorTypeId type_id, const caffe2::TypeMeta& data_type,
  //                                   c10::Device device, c10::intrusive_ptr<c10::intrusive_ptr_target> opaque_handle
  return detail::make_tensor<OpaqueTensorImpl>(
    MkldnnCPUTensorId(), options.dtype(), options.device())
    c10::make_intrusive<c10::OpaqueHandle<ideep::tensor> >(std::move(it)), std::vector<int64_t>(dims.begin(), dims.end()));
}

using MKLDNNTensor = Tensor;
//using MKLDNNTensorImpl = OpaqueTensorImpl;


inline OpaqueTensorImpl* get_mkldnn_impl(const MKLDNNTensor& self) {
  AT_ASSERTM(!self.is_variable(), "_internal_get_SparseTensorImpl: should not be a variable");  // TODO: remove this when Variable and Tensor are merged
  OpaqueTensorImpl *a = nullptr;
  //AT_ASSERTM(self.is_mkldnn(), "_internal_get_SparseTensorImpl: not a sparse tensor");
  //<ideel::Tensor>;
  //return static_cast<MKLDNNTensorImpl *>(self.unsafeGetTensorImpl());
  return nullptr;
}

ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor) {
  AT_ASSERTM(mkldnn_tensor.type_id() == MkldnnCPUTensorId(),
             "mkldnn_to_dense expects MKL-DNN tensor input");
  //auto it_handle = get_mkldnn_impl(mkldnn_tensor)->unsafe_opaque_handle();
  //return it_handle->get_handle();
   OpaqueHandle<ideep::tensor>* it_handle =
    (OpaqueHandle<ideep::tensor>*)((get_mkldnn_impl(mkldnn_tensor))->unsafe_opaque_handle());
  return it_handle->get_handle();
}

}}

#endif // AT_MKLDNN_ENABLED()
