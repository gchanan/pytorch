#include <type_traits>
#include "ATen/ApplyUtils.h"

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
bool allclose2(const Tensor& self, const Tensor& other, double rtol, double atol) {
  int equal = 1;
  ATH_TENSOR_APPLY2(scalartype, self, other,
                    if (equal && *self_data != *other_data) {
                    equal = 0;
                    TH_TENSOR_APPLY_hasFinished = 1; break;
                    });
  return equal == 1;
}

template <typename scalartype>
bool allclose3_cpu(const Tensor& self, const Tensor& other, double rtol, double atol) {
  AllCloseOp<scalartype> op;
  tensor_apply2_op<scalartype, AllCloseOp<scalartype>>(self, other, op);
  return op.equal;
}

}
}
