// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/${Dispatcher}.h"

// ${generated_comment}

namespace at {

${Dispatcher}::${Dispatcher}()
  : LegacyTHDispatcher(${Backend}TensorId()) {}

}
