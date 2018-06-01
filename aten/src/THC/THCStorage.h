#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include "THStorage.h"
#include "THCGeneral.h"

#define THCStorage_(NAME) TH_CONCAT_4(TH,CReal,Storage_,NAME)

#include "generic/THCStorage.h"
#include "THCGenerateAllTypes.h"

THC_API void THCStorage_free(THCState *state, THCStorage *self);
THC_API int THCStorage_getDevice(THCState* state, const THCStorage* storage);

#endif
