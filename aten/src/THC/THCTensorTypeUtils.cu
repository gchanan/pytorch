#include "THCTensorTypeUtils.cuh"
#include "THCTensor.h"
#include "THCTensorCopy.h"
#include "THCHalf.h"
#include <stdlib.h>

namespace {

struct SizeAndStride {
  int64_t size;
  int64_t stride;
};

/* 
 A comparator that will sort SizeAndStride structs by stride,
 in ascending order.
 */
int compareSizeAndStride(const void* a, const void* b) {
  const SizeAndStride* aS = (const SizeAndStride*) a;
  const SizeAndStride* bS = (const SizeAndStride*) b;
  
  if (aS->stride < bS->stride) return -1;
  if (aS->stride == bS->stride) return 0;
  return 1;
}

}

#define IMPL_TENSOR_UTILS(TENSOR_TYPE, DATA_TYPE)                       \
                                                                        \
TENSOR_TYPE*                                                            \
TensorUtils<TENSOR_TYPE>::newTensor(THCState* state) {                  \
  return TENSOR_TYPE##_new(state);                                      \
}                                                                       \
                                                                        \
TENSOR_TYPE*                                                            \
TensorUtils<TENSOR_TYPE>::newContiguous(THCState* state,                \
                                        TENSOR_TYPE* t) {               \
  return TENSOR_TYPE##_newContiguous(state, t);                         \
}                                                                       \
                                                                        \
void                                                                    \
TensorUtils<TENSOR_TYPE>::freeCopyTo(THCState* state,                   \
                                     TENSOR_TYPE* src,                  \
                                     TENSOR_TYPE* dst) {                \
  TENSOR_TYPE##_freeCopyTo(state, src, dst);                            \
}                                                                       \
                                                                        \
DATA_TYPE*                                                              \
TensorUtils<TENSOR_TYPE>::getData(THCState* state,                      \
                                  TENSOR_TYPE* t) {                     \
  /* FIXME: no cast is required except for THCudaHalfTensor */          \
  return (DATA_TYPE*) TENSOR_TYPE##_data(state, t);                     \
}                                                                       \
                                                                        \
/* Due to the resize semantics of ops with `out=` keywords, if       */ \
/* the output `tensor` has the same shape as the output of the       */ \
/* reduction operation, then any noncontiguities in the output       */ \
/* `tensor` should be preserved. This needs to be special cased b/c  */ \
/* otherwise, when keepdim=False, the implementations of reduction   */ \
/* ops resize `tensor` to the reduced size with keepdim=True, and    */ \
/* then later squeeze `tensor` to the correct output size, breaking  */ \
/* the contiguity guarantees of the resize semantics.                */ \
void                                                                    \
TensorUtils<TENSOR_TYPE>::preserveReduceDimSemantics(                   \
                          THCState *state, TENSOR_TYPE *tensor,         \
                          int in_dims, int64_t dimension, int keepdim) {\
  int out_dims = THCTensor_nDimension(state, tensor);                   \
  if (out_dims > 0 && !keepdim && out_dims == in_dims - 1) {            \
    THCTensor_unsqueeze1d(state, tensor, tensor, dimension);\
  }                                                                     \
}                                                                       \
                                                                        \
void                                                                    \
TensorUtils<TENSOR_TYPE>::copyIgnoringOverlaps(THCState* state,         \
                                               TENSOR_TYPE* dst,        \
                                               TENSOR_TYPE* src) {      \
  return TENSOR_TYPE##_copyIgnoringOverlaps(state, dst, src);           \
}                                                                       \

IMPL_TENSOR_UTILS(THCudaByteTensor, uint8_t)
IMPL_TENSOR_UTILS(THCudaCharTensor, int8_t)
IMPL_TENSOR_UTILS(THCudaShortTensor, int16_t)
IMPL_TENSOR_UTILS(THCudaIntTensor, int32_t)
IMPL_TENSOR_UTILS(THCudaLongTensor, int64_t)
IMPL_TENSOR_UTILS(THCudaTensor, float)
IMPL_TENSOR_UTILS(THCudaDoubleTensor, double)

#ifdef CUDA_HALF_TENSOR
IMPL_TENSOR_UTILS(THCudaHalfTensor, half);
#endif

#undef IMPL_TENSOR_UTILS
