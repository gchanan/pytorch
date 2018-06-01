#include "THSTensor.hpp"

#include <new>

#include "generic/THSTensor.cpp"
#include "THSGenerateAllTypes.h"

#include "generic/THSTensorMath.c"
#include "THSGenerateAllTypes.h"

void THSTensor_free(THSTensor *self)
{
  if(!self)
    return;
  if(--self->refcount == 0)
  {
    THFree(self->size);
    THLongTensor_free(self->indices);
    THTensor_free(self->values);
    self->refcount.~atomic<int>();
    THFree(self);
  }
}
