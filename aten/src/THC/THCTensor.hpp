#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THCTensor.h"
#include "THTensor.hpp"
#include "THCStorage.hpp"

#include <atomic>

typedef struct _THCTensor
{
    int64_t *size;
    int64_t *stride;
    int nDimension;

    THCStorage *storage;
    ptrdiff_t storageOffset;
    std::atomic<int> refcount;

    char flag;

} _THCTensor;

#include "generic/THCTensor.hpp"
#include "THCGenerateAllTypes.h"

THC_API int THCTensor_nDimension(THCState *state, const _THCTensor *self);
THC_API int64_t THCTensor_size(THCState *state, const _THCTensor *self, int dim);
THC_API int64_t THCTensor_stride(THCState *state, const _THCTensor *self, int dim);
THC_API THLongStorage *THCTensor_newSizeOf(THCState *state, _THCTensor *self);

THC_API bool THCTensor_isContiguous(THCState *state, const _THCTensor *self);
THC_API bool THCTensor_allContiguous(THCState *state, const _THCTensor **inputs, int numInputs);
THC_API ptrdiff_t THCTensor_nElement(THCState *state, const _THCTensor *self);

THC_API void THCTensor_retain(THCState *state, _THCTensor *self);
THC_API void THCTensor_free(THCState *state, _THCTensor *self);

THC_API int THCTensor_getDevice(THCState* state, const _THCTensor* tensor);
THC_API bool THCTensor_allSameDevice(THCState* state, const _THCTensor ** inputs, int numInputs);
