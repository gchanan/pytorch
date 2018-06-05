#include "THCGeneral.h"
#include "THCTensor.hpp"
#include "THCTensorCopy.h"

#include <new>

#include "generic/THCTensor.cpp"
#include "THCGenerateAllTypes.h"

int THCTensor_nDimension(THCState *state, const _THCTensor *self) {
  return self->nDimension;
}
