#include "THCGeneral.h"
#include "THCTensor.hpp"
#include "THCTensorCopy.h"

#include <new>

#include "generic/THCTensor.cpp"
#include "THCGenerateAllTypes.h"

int THCTensor_nDimension(THCState *state, const _THCTensor *self) {
  return self->nDimension;
}

int64_t THCTensor_size(THCState *state, const _THCTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->size[dim];
}

int64_t THCTensor_stride(THCState *state, const _THCTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->stride[dim];
}
THLongStorage *THCTensor_newSizeOf(THCState *state, _THCTensor *self) {
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

bool THCTensor_isContiguous(THCState *state, const _THCTensor *self) {
  int64_t z = 1;
  int d;
  for(d = self->nDimension-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return false;
    }
  }
  return true;
}

bool THCTensor_allContiguous(THCState *state, const _THCTensor **inputs, int numInputs) {
  THAssert(numInputs > 0);
  for (int i = 0; i < numInputs; ++i) {
    if (!THCTensor_isContiguous(state, inputs[i])) {
      return false;
    }
  }
  return true;
}

ptrdiff_t THCTensor_nElement(THCState *state, const _THCTensor *self) {
  if(self->nDimension == 0)
    return 0;
  else
  {
    ptrdiff_t nElement = 1;
    int d;
    for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THCTensor_retain(THCState *state, _THCTensor *self) {
  if(self->flag & TH_TENSOR_REFCOUNTED)
    self->refcount++;
}


void THCTensor_free(THCState *state, _THCTensor *self) {
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(--self->refcount == 0)
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THCStorage_free(state, self->storage);
      self->refcount.~atomic<int>();
      THFree(self);
    }
  }
}

int THCTensor_getDevice(THCState* state, const _THCTensor* tensor) {
  if (!tensor->storage) return -1;
  return THCStorage_getDevice(state, tensor->storage);
}

bool THCTensor_allSameDevice(THCState* state, const _THCTensor ** inputs, int numInputs) {
  THAssert(numInputs > 0);
  int device = THCTensor_getDevice(state, inputs[0]);
  for (int i = 1; i < numInputs; ++i) {
    if (THCTensor_getDevice(state, inputs[i]) != device) {
      return false;
    }
  }
  return true;
}
