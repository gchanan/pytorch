#include "torch/csrc/cuda/THCP.h"
#include "torch/csrc/expand_utils.h"

#include "torch/csrc/expand_utils-inl.h"

template <>
THCudaTensor *newForExpand(THCState *s) {
  return THCudaTensor_new(s);
}

template <>
THCudaDoubleTensor *newForExpand(THCState *s) {
  return THCudaDoubleTensor_new(s);
}

#ifdef CUDA_HALF_TENSOR
template <>
THCudaHalfTensor *newForExpand(THCState *s) {
  return THCudaHalfTensor_new(s);
}
#endif // CUDA_HALF_TENSOR

template <>
THCudaByteTensor *newForExpand(THCState *s) {
  return THCudaByteTensor_new(s);
}

template <>
THCudaCharTensor *newForExpand(THCState *s) {
  return THCudaCharTensor_new(s);
}

template <>
THCudaShortTensor *newForExpand(THCState *s) {
  return THCudaShortTensor_new(s);
}

template <>
THCudaIntTensor *newForExpand(THCState *s) {
  return THCudaIntTensor_new(s);
}

template <>
THCudaLongTensor *newForExpand(THCState *s) {
  return THCudaLongTensor_new(s);
}

template<>
int expand(THCState *s, THCudaTensor *r, THCudaTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template<>
int expand(THCState *s, THCudaDoubleTensor *r, THCudaDoubleTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaDoubleTensor_expand(s, r, tensor, sizes, raiseErrors);
}

#ifdef CUDA_HALF_TENSOR
template<>
int expand(THCState *s, THCudaHalfTensor *r, THCudaHalfTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaHalfTensor_expand(s, r, tensor, sizes, raiseErrors);
}
#endif // CUDA_HALF_TENSOR

template<>
int expand(THCState *s, THCudaByteTensor *r, THCudaByteTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaByteTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template<>
int expand(THCState *s, THCudaCharTensor *r, THCudaCharTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaCharTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template<>
int expand(THCState *s, THCudaShortTensor *r, THCudaShortTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaShortTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template<>
int expand(THCState *s, THCudaIntTensor *r, THCudaIntTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaIntTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template<>
int expand(THCState *s, THCudaLongTensor *r, THCudaLongTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaLongTensor_expand(s, r, tensor, sizes, raiseErrors);
}
