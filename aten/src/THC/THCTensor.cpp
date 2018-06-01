#include "THCGeneral.h"
#include "THCTensor.hpp"
#include "THCTensorCopy.h"

#include <new>

#include "generic/THCTensor.cpp"
#include "THCGenerateAllTypes.h"

void THCTensor_retain(THCState *state, THCTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    self->refcount++;
}

void THCTensor_free(THCState *state, THCTensor *self)
{
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

int THCTensor_(nDimension)(THCState *state, const THCTensor *self)
{
  return self->nDimension;
}

int64_t THCTensor_size(THCState *state, const THCTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->size[dim];
}

int64_t THCTensor_stride(THCState *state, const THCTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->stride[dim];
}

ptrdiff_t THCTensor_(nElement)(THCState *state, const THCTensor *self)
{
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

int THCTensor_(getDevice)(THCState* state, const THCTensor* tensor) {
  if (!tensor->storage) return -1;
  return THCStorage_getDevice(state, tensor->storage);
}

THLongStorage *THCTensor_newSizeOf(THCState *state, THCTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

int THCTensor_(isContiguous)(THCState *state, const THCTensor *self)
{
  int64_t z = 1;
  int d;
  for(d = self->nDimension-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return 0;
    }
  }
  return 1;
}

int THCTensor_allContiguous(THCState *state, const THCTensor **inputs, int numInputs) {
  THAssert(numInputs > 0);
  for (int i = 0; i < numInputs; ++i) {
    if (!THCTensor_isContiguous(state, inputs[i])) {
      return 0;
    }
  }
  return 1;
}
