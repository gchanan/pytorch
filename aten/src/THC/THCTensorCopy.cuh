#pragma once

#include "THCNumerics.cuh"

// Copy operator for the pointwise apply kernel
template <typename TypeDst, typename TypeSrc>
struct CopyOp {
  __device__ __forceinline__ void operator()(TypeDst* dst, TypeSrc* src) {
#if __CUDA_ARCH__ >= 350
    *dst = ScalarConvert<TypeSrc, TypeDst>::to(__ldg(src));
#else
    *dst = ScalarConvert<TypeSrc, TypeDst>::to(*src);
#endif
  }
};

template <typename ScalarTypeDst, typename ScalarTypeSrc>
void THC_copyTensor(THCState* state, _THCTensor* dst, _THCTensor* src);

template <typename ScalarType>
_THCTensor *THCTensor_newClone(THCState *state, _THCTensor *self);

template <typename ScalarType>
_THCTensor *THCTensor_newContiguous(THCState *state, _THCTensor *self);

template <typename ScalarType>
void THCTensor_freeCopyTo(THCState *state, _THCTensor *self, _THCTensor *dst);

template <typename ScalarType>
void THCTensor_copyIgnoringOverlaps(THCState* state, _THCTensor* dst, _THCTensor* src);
