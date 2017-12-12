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
bool allclose3_cpu(const Tensor& self, const Tensor& other, double rtol, double atol) {
  AllCloseOp<scalartype> op;
  tensor_apply2<scalartype, AllCloseOp<scalartype>>(self, other, op);
  return op.equal;
}

// TODO Replace this with more accurate digamma().
template <typename ScalarType>
static inline ScalarType digamma_one(ScalarType x) {
  const ScalarType eps = x * 1e-2;
  return (std::lgamma(x + eps) - std::lgamma(x - eps)) / (eps + eps);
}

/** Computes the reparameterized gradient -(d/dalpha cdf(x;alpha)) / pdf(x;alpha)
    for random number x drawn from a standard Gamma distribution Gamma(alpha).
*/
template <typename ScalarType>
static inline ScalarType standard_gamma_grad_one(ScalarType x, ScalarType alpha) {
  // Use an asymptotic approximation for small x.
  if (x < 0.2f) {
    const ScalarType a0 = 1 / alpha;
    const ScalarType a1 = 1 / (alpha + 1);
    const ScalarType a2 = 1 / (alpha + 2);
    const ScalarType pow_x_alpha = std::pow(x, alpha);
    const ScalarType gamma_pdf = std::pow(x, alpha - 1) * std::exp(-x);
    const ScalarType gamma_cdf = pow_x_alpha * (a0 - x*a1 + 0.5f*x*x*a2);
    const ScalarType gamma_cdf_alpha = (std::log(x) - digamma_one(alpha)) * gamma_cdf
        - pow_x_alpha * (a0*a0 - x*a1*a1 + 0.5f*x*x*a2*a2);
    const ScalarType result = -gamma_cdf_alpha / gamma_pdf;
    return isnan(result) ? 0 : result;
  }

  // Use an asymptotic approximation for large alpha.
  if (alpha > 50.0f) {
    return (x / alpha);
  }

  // Use a bivariate rational approximation to the reparameterized gradient.
  const ScalarType u = std::log(x / alpha);
  const ScalarType v = std::log(alpha);
  static const ScalarType coef_uv[3][8] = {
    {0.16028008, -0.088064309, 0.019630876, -0.0016920282,
     1.0, 0.36659853, 0.10843863, 0.0066895454},
    {0.521894, 0.16095838, 0.06237597, 0.0023884253,
     0.083457714, 0.0073297628, -0.0059299053, -0.00093720389},
    {-0.0031143957, -0.012143877, -0.0057656484, -0.00064847254,
     0.0087262576, -0.00022820524, 1.8871047e-05, 9.6307964e-06},
  };
  ScalarType coef_v[8];
  for (int i = 0; i < 8; ++ i) {
    coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
  }
  const ScalarType p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
  const ScalarType q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
  return std::exp(p / q);
}

template <typename ScalarType>
struct StandardGammaGradOp {
  void operator()(ScalarType& ret_val, const ScalarType& self_val, ScalarType &alpha_val, bool& early_exit)
  {
    ret_val = standard_gamma_grad_one(self_val , alpha_val);
  }
};

template <typename ScalarType>
Tensor standard_gamma_grad_cpu(const Tensor& self, const Tensor& alpha) {
  StandardGammaGradOp<ScalarType> op;
  Tensor ret = self.type().tensor(self.sizes());
  tensor_apply3<ScalarType, StandardGammaGradOp<ScalarType>>(ret, self, alpha, op);
  return ret;
}

}
}
