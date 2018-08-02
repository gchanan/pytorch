#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.hpp"
#else

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

// NOTE: functions exist here only to support dispatch via Declarations.cwrap.  You probably don't want to put
// new functions in here, they should probably be un-genericized.

TH_CPP_API void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, at::IntList size_, at::IntList stride_);
TH_CPP_API THTensor *THTensor_(newView)(THTensor *tensor, at::IntList size);

#endif
