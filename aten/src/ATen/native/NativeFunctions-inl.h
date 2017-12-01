#include <type_traits>
#include "ATen/ApplyUtils.h"
#include "ATen/DIspatchUtil.h"

namespace at {
namespace native {

template <typename scalartype> bool is_signed(const Tensor& self) {
  return std::is_signed<scalartype>::value;
}

template <typename scalartype>
struct AllCloseOp {
  bool equal = true;
  void operator()(const scalartype& x, const scalartype& y, bool& early_exit)
  {
    if (x != y) {
      equal = false;
      early_exit = true;
    }
  }
};

template <typename scalartype>
bool allclose3_cpu(const Tensor& self, const Tensor& other, double rtol, double atol) {
  AllCloseOp<scalartype> op;
  tensor_apply2<scalartype, AllCloseOp<scalartype>>(self, other, op);
  return op.equal;
}

}
}
