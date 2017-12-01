#include <type_traits>

namespace at {
namespace native {

template <typename scalartype> bool is_signed(const Tensor& self) {
  return std::is_signed<scalartype>::value;
}

template <typename scalartype>
bool allclose2(const Tensor& self, const Tensor& other, double rtol, double atol) {
  // FixMe: something different for each backend?
  int equal = 1;
  ATH_TENSOR_APPLY2(scalartype, scalartype, self, other,
                    if (equal && *self_data != *other_data) {
                    equal = 0;
                    TH_TENSOR_APPLY_hasFinished = 1; break;
                    });
  return equal == 1;
}

}
}
