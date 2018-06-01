#ifndef THS_TENSOR_INC
#define THS_TENSOR_INC

#include "TH.h"
#include <stdint.h>

#define THSTensor_(NAME)   TH_CONCAT_4(THS,Real,Tensor_,NAME)

#include "generic/THSTensor.h"
#include "THSGenerateAllTypes.h"

#include "generic/THSTensorMath.h"
#include "THSGenerateAllTypes.h"

// This exists to have a data-type independent way of freeing (necessary for THPPointer).
TH_API void THSTensor_free(THSTensor *storage);

#endif
