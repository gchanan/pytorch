#pragma once

#include "ATen/TensorUtils.h"
#include "ATen/TensorInfo.h"

//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to MAX_CUTORCH_DIMS) dimensioned arguments without
// copying or temporary storage.
//

namespace at {

// Threads per block for our apply kernel
// FIXME: use occupancy calculator instead
#define THC_APPLY_THREADS_PER_BLOCK 32 * 16
#define THC_APPLY_BLOCKS_PER_SM 4

template <typename Op,
          typename ScalarType,
          typename IndexType,
          int ADims, int BDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(THC_APPLY_THREADS_PER_BLOCK, THC_APPLY_BLOCKS_PER_SM)
#endif
__global__ void
kernelPointwiseApply2(TensorInfo<ScalarType, IndexType> a,
                      TensorInfo<ScalarType, IndexType> b,
                      IndexType totalElements,
                      Op& op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<ScalarType, IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<ScalarType, IndexType, BDims>::get(linearIndex, b);

    bool earlyExit = false;
    op(a.data[aOffset], b.data[bOffset], earlyExit);
  }
}

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

inline bool getApplyGrid(uint64_t totalElements, dim3& grid) {
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  if (curDevice == -1) return false;

  uint64_t numBlocks = ATenCeilDiv(totalElements, static_cast<uint64_t>(THC_APPLY_THREADS_PER_BLOCK));
  uint64_t maxGridX = at::globalContext().getCurrentDeviceProperties()->maxGridSize[0];
  if (numBlocks > maxGridX)
      numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

enum class TensorArgType { ReadWrite, ReadOnly };

inline dim3 getApplyBlock() {
  return dim3(THC_APPLY_THREADS_PER_BLOCK);
}

template <typename ScalarType, typename Op>
bool pointwiseApply2(at::Tensor a,
                     at::Tensor b,
                     Op& op,
                     at::TensorArgType aType = at::TensorArgType::ReadWrite,
                     at::TensorArgType bType = at::TensorArgType::ReadOnly) {  
  ptrdiff_t totalElements = a.numel();

  if (totalElements != b.numel()) {
    return false;
  }

  if (a.dim() > MAX_TENSORINFO_DIMS ||
      b.dim() > MAX_TENSORINFO_DIMS) {
    return false;
  }

  if (a.dim() == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }
  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  Tensor oldA;
  Tensor oldB;

  std::cerr << "aType " << (aType == at::TensorArgType::ReadWrite) << " " << (bType == at::TensorArgType::ReadOnly) << std::endl;

  if (aType == at::TensorArgType::ReadWrite && at::tensorutils::overlappingIndices(a)) {
    // Must perform in contiguous space
    oldA = a;
    a = a.contiguous();
  }
  if (bType == at::TensorArgType::ReadWrite && at::tensorutils::overlappingIndices(b)) {
    // Must perform in contiguous space
    oldB = b;
    b = b.contiguous();
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.

#define HANDLE_CASE(TYPE, A, B)                                         \
  std::cerr << "kernelpointwise apply 0" << std::endl;                  \
  kernelPointwiseApply2<Op,                                             \
                        ScalarType,                                     \
                        TYPE, A, B>                                   \
   <<<grid, block, 0, at::globalContext().getCurrentCUDAStream()>>>(    \
       aInfo, bInfo, (TYPE) totalElements, op);

#define HANDLE_B_CASE(TYPE, A, B)               \
  {                                             \
    if (bInfo.isContiguous()) {                 \
      HANDLE_CASE(TYPE, A, -2);                 \
    } else {                                    \
      switch (B) {                              \
        case 1:                                 \
        HANDLE_CASE(TYPE, A, 1);                \
        break;                                  \
        case 2:                                 \
        HANDLE_CASE(TYPE, A, 2);                \
        break;                                  \
        default:                                \
        HANDLE_CASE(TYPE, A, -1);               \
        break;                                  \
      }                                         \
    }                                           \
  }

#define HANDLE_A_CASE(TYPE, A, B)               \
  {                                             \
    if (aInfo.isContiguous()) {                 \
      HANDLE_B_CASE(TYPE, -2, B);               \
    } else {                                    \
      switch (A) {                              \
        case 1:                                 \
        HANDLE_B_CASE(TYPE, 1, B);              \
        break;                                  \
        case 2:                                 \
        HANDLE_B_CASE(TYPE, 2, B);              \
        break;                                  \
        default:                                \
        HANDLE_B_CASE(TYPE, -1, B);             \
        break;                                  \
      }                                         \
    }                                           \
  }

  if (at::tensorutils::canUse32BitIndexMath(a) &&
      at::tensorutils::canUse32BitIndexMath(b)) {
    TensorInfo<ScalarType, unsigned int> aInfo =
      at::tensorutils::getTensorInfo<ScalarType, unsigned int>(a);
    aInfo.collapseDims();

    TensorInfo<ScalarType, unsigned int> bInfo =
      at::tensorutils::getTensorInfo<ScalarType, unsigned int>(b);
    bInfo.collapseDims();
#if CUDA_VERSION < 9000
    if (!(aInfo.isContiguous() && bInfo.isContiguous()))
        grid.x = std::min((unsigned int)at::globalContext().getCurrentDeviceProperties()->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    TensorInfo<ScalarType, uint64_t> aInfo =
      at::tensorutils::getTensorInfo<ScalarType, uint64_t>(a);
    aInfo.collapseDims();

    TensorInfo<ScalarType, uint64_t> bInfo =
      at::tensorutils::getTensorInfo<ScalarType, uint64_t>(b);
    bInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous()) {
      std::cerr << "kernelpointwise apply 1" << std::endl;
      kernelPointwiseApply2<Op,
                            ScalarType,
                          uint64_t, -2, -2>
        <<<grid, block, 0, at::globalContext().getCurrentCUDAStream()>>>(
           aInfo, bInfo, (uint64_t) totalElements, op);
    } else {
#if CUDA_VERSION < 9000
      grid.x = std::min((unsigned int)at::globalContext().getCurrentDeviceProperties()->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif
      std::cerr << "kernelpointwise apply 2" << std::endl;
      kernelPointwiseApply2<Op,
                            ScalarType,
                            uint64_t, -1, -1>
        <<<grid, block, 0, at::globalContext().getCurrentCUDAStream()>>>(
           aInfo, bInfo, (uint64_t) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA.defined()) {
    // Ignore overlaps when copying back; if we use copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    oldA._copyIgnoringOverlaps(a);
    a = oldA;
  }

  if (oldB.defined()) {
    // Ignore overlaps when copying back; if we use copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    oldB._copyIgnoringOverlaps(b);
    b = oldB;
  }

  return true;
}

}
