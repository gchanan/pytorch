#ifndef THC_TENSOR_INC
#define THC_TENSOR_INC

#include "THTensor.h"
#include "THCStorage.h"
#include "THCGeneral.h"

#define THCTensor_(NAME)   TH_CONCAT_4(TH,CReal,Tensor_,NAME)

#define THC_DESC_BUFF_LEN 64

typedef struct THC_CLASS THCDescBuff
{
    char str[THC_DESC_BUFF_LEN];
} THCDescBuff;

#include "generic/THCTensor.h"
#include "THCGenerateAllTypes.h"

THC_API void THCTensor_retain(THCState *state, THCTensor *self);
THC_API void THCTensor_free(THCState *state, THCTensor *self);
THC_API int THCTensor_nDimension(THCState *state, const THCTensor *self);
THC_API int64_t THCTensor_size(THCState *state, const THCTensor *self, int dim);
THC_API int64_t THCTensor_stride(THCState *state, const THCTensor *self, int dim);
THC_API ptrdiff_t THCTensor_nElement(THCState *state, const THCTensor *self);
THC_API int THCTensor_getDevice(THCState* state, const THCTensor* tensor);
THC_API THLongStorage *THCTensor_newSizeOf(THCState *state, THCTensor *self);
THC_API int THCTensor_isContiguous(THCState *state, const THCTensor *self);
THC_API int THCTensor_allContiguous(THCState *state, const THCTensor **inputs, int numInputs);
THC_API void THCTensor_squeeze1d(THCState *state, THCTensor *self, THCTensor *src, int dimension_);
THC_API void THCTensor_unsqueeze1d(THCState *state, THCTensor *self, THCTensor *src, int dimension_);
THC_API int THCTensor_allSameDevice(THCState* state, const THCTensor ** nputs, int numInputs);
THC_API void THCTensor_resize(THCState *state, THCTensor *tensor, THLongStorage *size, THLongStorage *stride);
THC_API void THCTensor_resizeAs(THCState *state, THCTensor *tensor, THCTensor *src);
/* Can we use 32 bit math for indexing? */    
THC_API int THCTensor_canUse32BitIndexMath(THCState* state, const THCTensor* t, ptrdiff_t max_elem);

#endif
