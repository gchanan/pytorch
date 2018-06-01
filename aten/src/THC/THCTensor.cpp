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

int THCTensor_nDimension(THCState *state, const THCTensor *self)
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

ptrdiff_t THCTensor_nElement(THCState *state, const THCTensor *self)
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

int THCTensor_getDevice(THCState* state, const THCTensor* tensor) {
  if (!tensor->storage) return -1;
  return THCStorage_getDevice(state, tensor->storage);
}

THLongStorage *THCTensor_newSizeOf(THCState *state, THCTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

int THCTensor_isContiguous(THCState *state, const THCTensor *self)
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

void THCTensor_squeeze1d(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->nDimension, 3, "dimension out of range");

  THCTensor_(set)(state, self, src);

  if(src->size[dimension] == 1 && src->nDimension > 1)
  {
    for(d = dimension; d < self->nDimension-1; d++)
    {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->nDimension--;
  }
}


void THCTensor_unsqueeze1d(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->nDimension), 3, "dimension out of range");
  THArgCheck(src->nDimension > 0, 3, "cannot unsqueeze empty tensor");

  THCTensor_(set)(state, self, src);

  self->size = (int64_t*)THRealloc(self->size, sizeof(int64_t)*(self->nDimension+1));
  self->stride = (int64_t*)THRealloc(self->stride, sizeof(int64_t)*(self->nDimension+1));
  self->nDimension++;
  for (d = self->nDimension-1; d > dimension; d--) {
    self->size[d] = self->size[d-1];
    self->stride[d] = self->stride[d-1];
  }
  if (dimension+1 < self->nDimension) {
    self->stride[dimension] = self->size[dimension+1] * self->stride[dimension+1];
  } else {
    self->stride[dimension] = 1;
  }
  self->size[dimension] = 1;
}

int THCTensor_allSameDevice(THCState* state, const THCTensor ** nputs, int numInputs) {
  THAssert(numInputs > 0);
  int device = THCTensor_getDevice(state, inputs[0]);
  for (int i = 1; i < numInputs; ++i) {
    if (THCTensor_getDevice(state, inputs[i]) != device) {
      return 0;
    }
  }
  return 1;
}

void THCTensor_resize(THCState *state, THCTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THCTensor_resizeNd(state, self, size->size, THLongStorage_data(size), (stride ? THLongStorage_data(stride) : NULL));
}

void THCTensor_resizeAs(THCState *state, THCTensor *self, THCTensor *src)
{
  int isSame = 0;
  int d;
  if(self->nDimension == src->nDimension)
  {
    isSame = 1;
    for(d = 0; d < self->nDimension; d++)
    {
      if(self->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THCTensor_resizeNd(state, self, src->nDimension, src->size, NULL);
}

int THCTensor_canUse32BitIndexMath(THCState* state, const THCTensor* t, ptrdiff_t max_elem) {
  ptrdiff_t elements = THCTensor_nElement(state, t);
  if (elements >= max_elem) {
    return 0;
  }

  ptrdiff_t offset = 0;
  ptrdiff_t linearId = elements - 1;

  for (int i = THCTensor_nDimension(state, t) - 1; i >= 0; --i) {
    ptrdiff_t curDimIndex =
      linearId % THCTensor_size(state, t, i);
    ptrdiff_t curDimOffset = curDimIndex *
      THCTensor_stride(state, t, i);
    offset += curDimOffset;
    linearId /= THCTensor_size(state, t, i);
  }

  if (offset >= max_elem) {
    return 0;
  }

  return 1;
}

int THCTensor_all32BitIndexable(THCState* state, const THCTensor** inputs, int numInputs) {
  for (int i = 0; i < numInputs; ++i) {
    if (!THCTensor_canUse32BitIndexMath(state, inputs[i])) {
      return 0;
    }
  }
  return 1;
}
