#pragma once

// generated from tools/autograd/templates/python_torch_functions_dispatch.h

#include <ATen/ATen.h>
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/tensor/python_tensor.h"

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using at::Tensor;
using at::Scalar;
using at::TensorList;
using at::IntList;
using at::Generator;
using at::SparseTensor;
using at::Storage;

static at::Type& default_type() {
  return torch::tensor::get_default_tensor_type();
}

static void maybe_initialize_cuda(const at::Type &type) {
  if (type.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

inline Tensor & dispatch___and__(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::__and___out(result, self, other);
}
inline Tensor dispatch___and__(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__and__(other);
}
inline Tensor & dispatch___and__(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::__and___out(result, self, other);
}
inline Tensor dispatch___and__(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__and__(other);
}
inline Tensor & dispatch___lshift__(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::__lshift___out(result, self, other);
}
inline Tensor dispatch___lshift__(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__lshift__(other);
}
inline Tensor & dispatch___lshift__(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::__lshift___out(result, self, other);
}
inline Tensor dispatch___lshift__(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__lshift__(other);
}
inline Tensor & dispatch___or__(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::__or___out(result, self, other);
}
inline Tensor dispatch___or__(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__or__(other);
}
inline Tensor & dispatch___or__(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::__or___out(result, self, other);
}
inline Tensor dispatch___or__(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__or__(other);
}
inline Tensor & dispatch___rshift__(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::__rshift___out(result, self, other);
}
inline Tensor dispatch___rshift__(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__rshift__(other);
}
inline Tensor & dispatch___rshift__(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::__rshift___out(result, self, other);
}
inline Tensor dispatch___rshift__(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__rshift__(other);
}
inline Tensor & dispatch___xor__(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::__xor___out(result, self, other);
}
inline Tensor dispatch___xor__(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__xor__(other);
}
inline Tensor & dispatch___xor__(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::__xor___out(result, self, other);
}
inline Tensor dispatch___xor__(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__xor__(other);
}
inline Tensor & dispatch__abs(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_abs_out(result, self);
}
inline Tensor dispatch__abs(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._abs();
}
inline Tensor & dispatch__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_addmv_out(result, self, mat, vec, beta, alpha);
}
inline Tensor dispatch__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._addmv(mat, vec, beta, alpha);
}
inline Tensor & dispatch__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_addr_out(result, self, vec1, vec2, beta, alpha);
}
inline Tensor dispatch__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._addr(vec1, vec2, beta, alpha);
}
inline Tensor dispatch__argmax(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._argmax(dim, keepdim);
}
inline Tensor dispatch__argmin(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._argmin(dim, keepdim);
}
inline Tensor dispatch__cast_Half(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._cast_Half(non_blocking);
}
inline Tensor dispatch__cast_double(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._cast_double(non_blocking);
}
inline Tensor dispatch__cast_float(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._cast_float(non_blocking);
}
inline Tensor dispatch__cast_int(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._cast_int(non_blocking);
}
inline Tensor dispatch__cast_int16_t(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._cast_int16_t(non_blocking);
}
inline Tensor dispatch__cast_int64_t(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._cast_int64_t(non_blocking);
}
inline Tensor dispatch__cast_int8_t(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._cast_int8_t(non_blocking);
}
inline Tensor dispatch__cast_uint8_t(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._cast_uint8_t(non_blocking);
}
inline Tensor & dispatch__cat(TensorList tensors, int64_t dim, Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::_cat_out(self, tensors, dim);
}
inline Tensor dispatch__cat(TensorList tensors, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(tensors);
  return at::_cat(tensors, dim);
}
inline Tensor & dispatch__ceil(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_ceil_out(result, self);
}
inline Tensor dispatch__ceil(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._ceil();
}
inline Tensor dispatch__convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}
inline Tensor dispatch__convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
}
inline Tensor & dispatch__cos(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_cos_out(result, self);
}
inline Tensor dispatch__cos(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._cos();
}
inline Tensor dispatch__cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const Type & ty, int64_t device) {
  maybe_initialize_cuda(ty);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::_cudnn_init_dropout_state(ty, dropout, train, dropout_seed);
}
inline std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> dispatch__cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::_cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}
inline Tensor dispatch__cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(weight_arr);
  return at::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
}
inline Tensor & dispatch__dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::_dirichlet_grad_out(output, x, alpha, total);
}
inline Tensor dispatch__dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(x);
  return at::_dirichlet_grad(x, alpha, total);
}
inline Tensor dispatch__dot(const Tensor & self, const Tensor & tensor) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._dot(tensor);
}
inline Tensor & dispatch__exp(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_exp_out(result, self);
}
inline Tensor dispatch__exp(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._exp();
}
inline Tensor dispatch__fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._fft_with_size(signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
inline Tensor & dispatch__floor(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_floor_out(result, self);
}
inline Tensor dispatch__floor(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._floor();
}
inline Tensor & dispatch__ger(const Tensor & self, const Tensor & vec2, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_ger_out(result, self, vec2);
}
inline Tensor dispatch__ger(const Tensor & self, const Tensor & vec2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._ger(vec2);
}
inline Tensor & dispatch__log(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_log_out(result, self);
}
inline Tensor dispatch__log(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._log();
}
inline Tensor & dispatch__mm(const Tensor & self, const Tensor & mat2, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_mm_out(result, self, mat2);
}
inline Tensor dispatch__mm(const Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._mm(mat2);
}
inline Tensor & dispatch__mv(const Tensor & self, const Tensor & vec, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_mv_out(result, self, vec);
}
inline Tensor dispatch__mv(const Tensor & self, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._mv(vec);
}
inline Tensor & dispatch__prod(const Tensor & self, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_prod_out(result, self, dim, keepdim);
}
inline Tensor dispatch__prod(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._prod(dim, keepdim);
}
inline Tensor dispatch__prodall(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._prodall();
}
inline Tensor & dispatch__round(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_round_out(result, self);
}
inline Tensor dispatch__round(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._round();
}
inline Tensor dispatch__s_where(const Tensor & condition, const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(condition);
  return self._s_where(condition, other);
}
inline Tensor & dispatch__sin(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_sin_out(result, self);
}
inline Tensor dispatch__sin(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._sin();
}
inline Tensor & dispatch__sqrt(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_sqrt_out(result, self);
}
inline Tensor dispatch__sqrt(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._sqrt();
}
inline Tensor & dispatch__standard_gamma(const Tensor & self, Generator * generator, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::_standard_gamma_out(output, self, generator);
}
inline Tensor dispatch__standard_gamma(const Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._standard_gamma(generator);
}
inline Tensor dispatch__standard_gamma_grad(const Tensor & self, const Tensor & output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._standard_gamma_grad(output);
}
inline Tensor & dispatch__sum(const Tensor & self, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_sum_out(result, self, dim, keepdim);
}
inline Tensor dispatch__sum(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._sum(dim, keepdim);
}
inline Tensor dispatch__sumall(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._sumall();
}
inline Tensor & dispatch__trunc(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::_trunc_out(result, self);
}
inline Tensor dispatch__trunc(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._trunc();
}
inline std::tuple<Tensor,Tensor> dispatch__unique(const Tensor & self, bool sorted, bool return_inverse) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._unique(sorted, return_inverse);
}
inline Tensor & dispatch_abs(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::abs_out(result, self);
}
inline Tensor dispatch_abs(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.abs();
}
inline Tensor & dispatch_abs_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.abs_();
}
inline Tensor & dispatch_acos(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::acos_out(result, self);
}
inline Tensor dispatch_acos(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.acos();
}
inline Tensor dispatch_adaptive_avg_pool1d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::adaptive_avg_pool1d(self, output_size);
}
inline std::tuple<Tensor,Tensor> dispatch_adaptive_max_pool1d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::adaptive_max_pool1d(self, output_size);
}
inline Tensor & dispatch_add(const Tensor & self, Scalar alpha, const Tensor & other, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::add_out(out, self, other, alpha);
}
inline Tensor dispatch_add(const Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.add(other, alpha);
}
inline Tensor & dispatch_add(const Tensor & self, Scalar other, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::add_out(result, self, other, alpha);
}
inline Tensor dispatch_add(const Tensor & self, Scalar other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.add(other, alpha);
}
inline Tensor & dispatch_add(const Tensor & self, const Tensor & other, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::add_out(result, self, other, alpha);
}
inline Tensor dispatch_add(const Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.add(other, alpha);
}
inline Tensor & dispatch_addbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::addbmm_out(out, self, batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addbmm(batch1, batch2, beta, alpha);
}
inline Tensor & dispatch_addbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::addbmm_out(out, self, batch1, batch2, beta, 1);
}
inline Tensor dispatch_addbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addbmm(batch1, batch2, beta, 1);
}
inline Tensor & dispatch_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::addbmm_out(result, self, batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addbmm(batch1, batch2, beta, alpha);
}
inline Tensor & dispatch_addcdiv(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::addcdiv_out(out, self, tensor1, tensor2, value);
}
inline Tensor dispatch_addcdiv(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcdiv(tensor1, tensor2, value);
}
inline Tensor & dispatch_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::addcdiv_out(result, self, tensor1, tensor2, value);
}
inline Tensor dispatch_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcdiv(tensor1, tensor2, value);
}
inline Tensor & dispatch_addcmul(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::addcmul_out(out, self, tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcmul(tensor1, tensor2, value);
}
inline Tensor & dispatch_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::addcmul_out(result, self, tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcmul(tensor1, tensor2, value);
}
inline Tensor & dispatch_addmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::addmm_out(out, self, mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmm(mat1, mat2, beta, alpha);
}
inline Tensor & dispatch_addmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::addmm_out(out, self, mat1, mat2, beta, 1);
}
inline Tensor dispatch_addmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmm(mat1, mat2, beta, 1);
}
inline Tensor & dispatch_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::addmm_out(result, self, mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmm(mat1, mat2, beta, alpha);
}
inline Tensor & dispatch_addmv(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::addmv_out(out, self, mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv(mat, vec, beta, alpha);
}
inline Tensor & dispatch_addmv(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::addmv_out(out, self, mat, vec, beta, 1);
}
inline Tensor dispatch_addmv(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv(mat, vec, beta, 1);
}
inline Tensor & dispatch_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::addmv_out(result, self, mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv(mat, vec, beta, alpha);
}
inline Tensor & dispatch_addmv_(Scalar beta, Tensor self, Scalar alpha, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv_(mat, vec, beta, alpha);
}
inline Tensor & dispatch_addmv_(Scalar beta, Tensor self, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv_(mat, vec, beta, 1);
}
inline Tensor & dispatch_addmv_(Tensor self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv_(mat, vec, beta, alpha);
}
inline Tensor & dispatch_addr(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::addr_out(out, self, vec1, vec2, beta, alpha);
}
inline Tensor dispatch_addr(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addr(vec1, vec2, beta, alpha);
}
inline Tensor & dispatch_addr(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::addr_out(out, self, vec1, vec2, beta, 1);
}
inline Tensor dispatch_addr(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addr(vec1, vec2, beta, 1);
}
inline Tensor & dispatch_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::addr_out(result, self, vec1, vec2, beta, alpha);
}
inline Tensor dispatch_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addr(vec1, vec2, beta, alpha);
}
inline bool dispatch_allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.allclose(other, rtol, atol, equal_nan);
}
inline Tensor & dispatch_arange(Scalar end, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::arange_out(result, end);
}
inline Tensor dispatch_arange(Scalar end, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::arange(dtype, end);
}
inline Tensor & dispatch_arange(Scalar start, Scalar end, Scalar step, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::arange_out(result, start, end, step);
}
inline Tensor dispatch_arange(Scalar start, Scalar end, Scalar step, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::arange(dtype, start, end, step);
}
inline Tensor & dispatch_as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::as_strided_out(result, self, size, stride, storage_offset);
}
inline Tensor dispatch_as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.as_strided(size, stride, storage_offset);
}
inline Tensor & dispatch_asin(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::asin_out(result, self);
}
inline Tensor dispatch_asin(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.asin();
}
inline Tensor & dispatch_atan(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::atan_out(result, self);
}
inline Tensor dispatch_atan(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.atan();
}
inline Tensor & dispatch_atan2(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::atan2_out(result, self, other);
}
inline Tensor dispatch_atan2(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.atan2(other);
}
inline Tensor & dispatch_baddbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::baddbmm_out(out, self, batch1, batch2, beta, alpha);
}
inline Tensor dispatch_baddbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.baddbmm(batch1, batch2, beta, alpha);
}
inline Tensor & dispatch_baddbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::baddbmm_out(out, self, batch1, batch2, beta, 1);
}
inline Tensor dispatch_baddbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.baddbmm(batch1, batch2, beta, 1);
}
inline Tensor & dispatch_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::baddbmm_out(result, self, batch1, batch2, beta, alpha);
}
inline Tensor dispatch_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.baddbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
inline Tensor & dispatch_bernoulli(const Tensor & self, Generator * generator, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::bernoulli_out(output, self, generator);
}
inline Tensor dispatch_bernoulli(const Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.bernoulli(generator);
}
inline Tensor & dispatch_bernoulli_(Tensor self, const Tensor & p, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.bernoulli_(p, generator);
}
inline Tensor & dispatch_bernoulli_(Tensor self, double p, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.bernoulli_(p, generator);
}
inline Tensor dispatch_bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input1);
  return at::bilinear(input1, input2, weight, bias);
}
inline Tensor & dispatch_bmm(const Tensor & self, const Tensor & mat2, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::bmm_out(result, self, mat2);
}
inline Tensor dispatch_bmm(const Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.bmm(mat2);
}
inline std::tuple<Tensor &,Tensor &> dispatch_btrifact(const Tensor & self, bool pivot, Tensor & result, Tensor & pivots) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::btrifact_out(result, pivots, self, pivot);
}
inline std::tuple<Tensor,Tensor> dispatch_btrifact(const Tensor & self, bool pivot) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.btrifact(pivot);
}
inline std::tuple<Tensor &,Tensor &,Tensor &> dispatch_btrifact_with_info(const Tensor & self, bool pivot, Tensor & result, Tensor & pivots, Tensor & info) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::btrifact_with_info_out(result, pivots, info, self, pivot);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_btrifact_with_info(const Tensor & self, bool pivot) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.btrifact_with_info(pivot);
}
inline Tensor & dispatch_btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::btrisolve_out(result, self, LU_data, LU_pivots);
}
inline Tensor dispatch_btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.btrisolve(LU_data, LU_pivots);
}
inline Tensor & dispatch_cat(TensorList tensors, int64_t dim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::cat_out(result, tensors, dim);
}
inline Tensor dispatch_cat(TensorList tensors, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(tensors);
  return at::cat(tensors, dim);
}
inline Tensor & dispatch_ceil(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::ceil_out(result, self);
}
inline Tensor dispatch_ceil(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ceil();
}
inline Tensor & dispatch_ceil_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ceil_();
}
inline std::vector<Tensor> dispatch_chunk(const Tensor & self, int64_t chunks, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.chunk(chunks, dim);
}
inline Tensor dispatch_conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::conv1d(input, weight, bias, stride, padding, dilation, groups);
}
inline Tensor dispatch_conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::conv2d(input, weight, bias, stride, padding, dilation, groups);
}
inline Tensor dispatch_conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::conv3d(input, weight, bias, stride, padding, dilation, groups);
}
inline Tensor dispatch_conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.conv_tbc(weight, bias, pad);
}
inline Tensor dispatch_conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
inline Tensor dispatch_conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
inline Tensor dispatch_conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
inline Tensor dispatch_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}
inline Tensor & dispatch_cos(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::cos_out(result, self);
}
inline Tensor dispatch_cos(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cos();
}
inline Tensor & dispatch_cos_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cos_();
}
inline Tensor & dispatch_cosh(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::cosh_out(result, self);
}
inline Tensor dispatch_cosh(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cosh();
}
inline Tensor dispatch_cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input1);
  return at::cosine_embedding_loss(input1, input2, target, margin, size_average, reduce);
}
inline Tensor & dispatch_cross(const Tensor & self, const Tensor & other, int64_t dim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::cross_out(result, self, other, dim);
}
inline Tensor dispatch_cross(const Tensor & self, const Tensor & other, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cross(other, dim);
}
inline Tensor dispatch_cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(theta);
  return at::cudnn_affine_grid_generator(theta, N, C, H, W);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}
inline Tensor dispatch_cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::cudnn_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}
inline Tensor dispatch_cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::cudnn_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}
inline Tensor dispatch_cudnn_grid_sampler(const Tensor & self, const Tensor & grid) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::cudnn_grid_sampler(self, grid);
}
inline bool dispatch_cudnn_is_acceptable(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::cudnn_is_acceptable(self);
}
inline Tensor & dispatch_cumprod(const Tensor & self, int64_t dim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::cumprod_out(result, self, dim);
}
inline Tensor dispatch_cumprod(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cumprod(dim);
}
inline Tensor & dispatch_cumsum(const Tensor & self, int64_t dim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::cumsum_out(result, self, dim);
}
inline Tensor dispatch_cumsum(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cumsum(dim);
}
inline Tensor dispatch_det(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.det();
}
inline Tensor & dispatch_diag(const Tensor & self, int64_t diagonal, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::diag_out(result, self, diagonal);
}
inline Tensor dispatch_diag(const Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.diag(diagonal);
}
inline Tensor dispatch_diagflat(const Tensor & self, int64_t offset) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::diagflat(self, offset);
}
inline Tensor dispatch_diagonal(const Tensor & self, int64_t offset) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::diagonal(self, offset);
}
inline Tensor & dispatch_digamma(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::digamma_out(result, self);
}
inline Tensor dispatch_digamma(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.digamma();
}
inline Tensor dispatch_dist(const Tensor & self, const Tensor & other, Scalar p) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.dist(other, p);
}
inline Tensor & dispatch_div(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::div_out(result, self, other);
}
inline Tensor dispatch_div(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.div(other);
}
inline Tensor & dispatch_div(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::div_out(result, self, other);
}
inline Tensor dispatch_div(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.div(other);
}
inline Tensor dispatch_dot(const Tensor & self, const Tensor & tensor) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.dot(tensor);
}
inline std::tuple<Tensor &,Tensor &> dispatch_eig(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(res1);
  return at::eig_out(res1, res2, self, eigenvectors);
}
inline std::tuple<Tensor,Tensor> dispatch_eig(const Tensor & self, bool eigenvectors) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.eig(eigenvectors);
}
inline Tensor dispatch_embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(weight);
  return at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(weight);
  return at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
inline Tensor & dispatch_embedding_renorm_(Tensor self, const Tensor & indices, double max_norm, double norm_type) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::embedding_renorm_(self, indices, max_norm, norm_type);
}
inline Tensor & dispatch_empty(IntList size, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::empty_out(result, size);
}
inline Tensor dispatch_empty(IntList size, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::empty(dtype, size);
}
inline Tensor dispatch_empty_like(const Tensor & self, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device == -1 ? (self.type().is_cuda() ? self.get_device() : -1) : device);
  return at::empty_like(self, dtype);
}
inline Tensor dispatch_empty_like(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::empty_like(self);
}
inline Tensor & dispatch_eq(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::eq_out(result, self, other);
}
inline Tensor dispatch_eq(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.eq(other);
}
inline Tensor & dispatch_eq(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::eq_out(result, self, other);
}
inline Tensor dispatch_eq(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.eq(other);
}
inline bool dispatch_equal(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.equal(other);
}
inline Tensor & dispatch_erf(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::erf_out(result, self);
}
inline Tensor dispatch_erf(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.erf();
}
inline Tensor & dispatch_erfinv(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::erfinv_out(result, self);
}
inline Tensor dispatch_erfinv(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.erfinv();
}
inline Tensor & dispatch_exp(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::exp_out(result, self);
}
inline Tensor dispatch_exp(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.exp();
}
inline Tensor & dispatch_exp_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.exp_();
}
inline Tensor & dispatch_expm1(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::expm1_out(result, self);
}
inline Tensor dispatch_expm1(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.expm1();
}
inline Tensor & dispatch_eye(int64_t n, int64_t m, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::eye_out(result, n, m);
}
inline Tensor dispatch_eye(int64_t n, int64_t m, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::eye(dtype, n, m);
}
inline Tensor dispatch_fft(const Tensor & self, int64_t signal_ndim, bool normalized) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.fft(signal_ndim, normalized);
}
inline Tensor & dispatch_floor(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::floor_out(result, self);
}
inline Tensor dispatch_floor(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.floor();
}
inline Tensor & dispatch_floor_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.floor_();
}
inline Tensor & dispatch_fmod(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::fmod_out(result, self, other);
}
inline Tensor dispatch_fmod(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.fmod(other);
}
inline Tensor & dispatch_fmod(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::fmod_out(result, self, other);
}
inline Tensor dispatch_fmod(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.fmod(other);
}
inline Tensor & dispatch_frac(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::frac_out(result, self);
}
inline Tensor dispatch_frac(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.frac();
}
inline Tensor & dispatch_full(IntList size, Scalar fill_value, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::full_out(result, size, fill_value);
}
inline Tensor dispatch_full(IntList size, Scalar fill_value, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::full(dtype, size, fill_value);
}
inline Tensor dispatch_full_like(const Tensor & self, Scalar fill_value, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device == -1 ? (self.type().is_cuda() ? self.get_device() : -1) : device);
  return at::full_like(self, fill_value, dtype);
}
inline Tensor dispatch_full_like(const Tensor & self, Scalar fill_value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::full_like(self, fill_value);
}
inline Tensor & dispatch_gather(const Tensor & self, int64_t dim, const Tensor & index, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::gather_out(result, self, dim, index);
}
inline Tensor dispatch_gather(const Tensor & self, int64_t dim, const Tensor & index) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gather(dim, index);
}
inline Tensor & dispatch_ge(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::ge_out(result, self, other);
}
inline Tensor dispatch_ge(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ge(other);
}
inline Tensor & dispatch_ge(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::ge_out(result, self, other);
}
inline Tensor dispatch_ge(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ge(other);
}
inline std::tuple<Tensor &,Tensor &> dispatch_gels(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(res1);
  return at::gels_out(res1, res2, self, A);
}
inline std::tuple<Tensor,Tensor> dispatch_gels(const Tensor & self, const Tensor & A) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gels(A);
}
inline std::tuple<Tensor &,Tensor &> dispatch_geqrf(const Tensor & self, Tensor & res1, Tensor & res2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(res1);
  return at::geqrf_out(res1, res2, self);
}
inline std::tuple<Tensor,Tensor> dispatch_geqrf(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.geqrf();
}
inline Tensor & dispatch_ger(const Tensor & self, const Tensor & vec2, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::ger_out(result, self, vec2);
}
inline Tensor dispatch_ger(const Tensor & self, const Tensor & vec2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ger(vec2);
}
inline std::tuple<Tensor &,Tensor &> dispatch_gesv(const Tensor & self, const Tensor & A, Tensor & solution, Tensor & lu) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(solution);
  return at::gesv_out(solution, lu, self, A);
}
inline std::tuple<Tensor,Tensor> dispatch_gesv(const Tensor & self, const Tensor & A) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gesv(A);
}
inline Tensor dispatch_group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::group_norm(input, num_groups, weight, bias, eps, cudnn_enabled);
}
inline Tensor & dispatch_gt(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::gt_out(result, self, other);
}
inline Tensor dispatch_gt(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gt(other);
}
inline Tensor & dispatch_gt(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::gt_out(result, self, other);
}
inline Tensor dispatch_gt(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gt(other);
}
inline Tensor dispatch_hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::hinge_embedding_loss(self, target, margin, size_average, reduce);
}
inline Tensor & dispatch_histc(const Tensor & self, int64_t bins, Scalar min, Scalar max, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::histc_out(result, self, bins, min, max);
}
inline Tensor dispatch_histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.histc(bins, min, max);
}
inline Tensor & dispatch_hspmm(const Tensor & mat1, const Tensor & mat2, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::hspmm_out(result, mat1, mat2);
}
inline Tensor dispatch_hspmm(const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(mat1);
  return at::hspmm(mat1, mat2);
}
inline Tensor dispatch_ifft(const Tensor & self, int64_t signal_ndim, bool normalized) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ifft(signal_ndim, normalized);
}
inline Tensor dispatch_index(const Tensor & self, TensorList indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index(indices);
}
inline Tensor & dispatch_index_put_(Tensor self, TensorList indices, const Tensor & values) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index_put_(indices, values);
}
inline Tensor & dispatch_index_select(const Tensor & self, int64_t dim, const Tensor & index, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::index_select_out(result, self, dim, index);
}
inline Tensor dispatch_index_select(const Tensor & self, int64_t dim, const Tensor & index) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index_select(dim, index);
}
inline Tensor & dispatch_inverse(const Tensor & self, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::inverse_out(output, self);
}
inline Tensor dispatch_inverse(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.inverse();
}
inline Tensor dispatch_irfft(const Tensor & self, int64_t signal_ndim, IntList signal_sizes, bool normalized, bool onesided) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.irfft(signal_ndim, signal_sizes, normalized, onesided);
}
inline bool dispatch_is_distributed(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_distributed();
}
inline bool dispatch_is_floating_point(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_floating_point();
}
inline bool dispatch_is_nonzero(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_nonzero();
}
inline bool dispatch_is_same_size(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_same_size(other);
}
inline bool dispatch_is_signed(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_signed();
}
inline Tensor dispatch_isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.isclose(other, rtol, atol, equal_nan);
}
inline std::tuple<Tensor &,Tensor &> dispatch_kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(values);
  return at::kthvalue_out(values, indices, self, k, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.kthvalue(k, dim, keepdim);
}
inline Tensor dispatch_layer_norm(const Tensor & input, IntList normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input);
  return at::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
}
inline Tensor & dispatch_le(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::le_out(result, self, other);
}
inline Tensor dispatch_le(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.le(other);
}
inline Tensor & dispatch_le(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::le_out(result, self, other);
}
inline Tensor dispatch_le(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.le(other);
}
inline Tensor & dispatch_lerp(const Tensor & self, const Tensor & end, Scalar weight, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::lerp_out(result, self, end, weight);
}
inline Tensor dispatch_lerp(const Tensor & self, const Tensor & end, Scalar weight) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lerp(end, weight);
}
inline Tensor & dispatch_lgamma(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::lgamma_out(result, self);
}
inline Tensor dispatch_lgamma(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lgamma();
}
inline Tensor & dispatch_linspace(Scalar start, Scalar end, int64_t steps, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::linspace_out(result, start, end, steps);
}
inline Tensor dispatch_linspace(Scalar start, Scalar end, int64_t steps, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::linspace(dtype, start, end, steps);
}
inline Tensor & dispatch_log(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::log_out(result, self);
}
inline Tensor dispatch_log(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.log();
}
inline Tensor & dispatch_log10(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::log10_out(result, self);
}
inline Tensor dispatch_log10(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.log10();
}
inline Tensor & dispatch_log1p(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::log1p_out(result, self);
}
inline Tensor dispatch_log1p(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.log1p();
}
inline Tensor & dispatch_log2(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::log2_out(result, self);
}
inline Tensor dispatch_log2(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.log2();
}
inline Tensor & dispatch_log_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.log_();
}
inline Tensor dispatch_logdet(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.logdet();
}
inline Tensor & dispatch_logspace(Scalar start, Scalar end, int64_t steps, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::logspace_out(result, start, end, steps);
}
inline Tensor dispatch_logspace(Scalar start, Scalar end, int64_t steps, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::logspace(dtype, start, end, steps);
}
inline Tensor & dispatch_lt(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::lt_out(result, self, other);
}
inline Tensor dispatch_lt(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lt(other);
}
inline Tensor & dispatch_lt(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::lt_out(result, self, other);
}
inline Tensor dispatch_lt(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lt(other);
}
inline Tensor dispatch_margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(input1);
  return at::margin_ranking_loss(input1, input2, target, margin, size_average, reduce);
}
inline Tensor & dispatch_masked_select(const Tensor & self, const Tensor & mask, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::masked_select_out(result, self, mask);
}
inline Tensor dispatch_masked_select(const Tensor & self, const Tensor & mask) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.masked_select(mask);
}
inline Tensor dispatch_matmul(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.matmul(other);
}
inline Tensor dispatch_max(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.max();
}
inline Tensor & dispatch_max(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::max_out(result, self, other);
}
inline Tensor dispatch_max(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.max(other);
}
inline std::tuple<Tensor &,Tensor &> dispatch_max(const Tensor & self, int64_t dim, bool keepdim, Tensor & max, Tensor & max_indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(max);
  return at::max_out(max, max_indices, self, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_max(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.max(dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_max_pool1d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
}
inline Tensor dispatch_mean(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mean();
}
inline Tensor & dispatch_mean(const Tensor & self, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::mean_out(result, self, dim, keepdim);
}
inline Tensor dispatch_mean(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mean(dim, keepdim);
}
inline Tensor dispatch_median(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.median();
}
inline std::tuple<Tensor &,Tensor &> dispatch_median(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(values);
  return at::median_out(values, indices, self, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_median(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.median(dim, keepdim);
}
inline Tensor dispatch_min(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.min();
}
inline Tensor & dispatch_min(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::min_out(result, self, other);
}
inline Tensor dispatch_min(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.min(other);
}
inline std::tuple<Tensor &,Tensor &> dispatch_min(const Tensor & self, int64_t dim, bool keepdim, Tensor & min, Tensor & min_indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(min);
  return at::min_out(min, min_indices, self, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_min(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.min(dim, keepdim);
}
inline Tensor dispatch_mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::mkldnn_convolution(self, weight, bias, padding, stride, dilation);
}
inline std::tuple<Tensor,Tensor> dispatch_mkldnn_convolution_backward_weights(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, bool bias_defined) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(grad_output);
  return at::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, bias_defined);
}
inline Tensor & dispatch_mm(const Tensor & self, const Tensor & mat2, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::mm_out(result, self, mat2);
}
inline Tensor dispatch_mm(const Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mm(mat2);
}
inline std::tuple<Tensor &,Tensor &> dispatch_mode(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(values);
  return at::mode_out(values, indices, self, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_mode(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mode(dim, keepdim);
}
inline Tensor & dispatch_mul(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::mul_out(result, self, other);
}
inline Tensor dispatch_mul(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mul(other);
}
inline Tensor & dispatch_mul(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::mul_out(result, self, other);
}
inline Tensor dispatch_mul(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mul(other);
}
inline Tensor & dispatch_multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::multinomial_out(result, self, num_samples, replacement, generator);
}
inline Tensor dispatch_multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.multinomial(num_samples, replacement, generator);
}
inline Tensor & dispatch_mv(const Tensor & self, const Tensor & vec, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::mv_out(result, self, vec);
}
inline Tensor dispatch_mv(const Tensor & self, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mv(vec);
}
inline Tensor dispatch_narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.narrow(dim, start, length);
}
inline Tensor & dispatch_ne(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::ne_out(result, self, other);
}
inline Tensor dispatch_ne(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ne(other);
}
inline Tensor & dispatch_ne(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::ne_out(result, self, other);
}
inline Tensor dispatch_ne(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ne(other);
}
inline Tensor & dispatch_neg(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::neg_out(result, self);
}
inline Tensor dispatch_neg(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.neg();
}
inline Tensor & dispatch_nonzero(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::nonzero_out(result, self);
}
inline Tensor dispatch_nonzero(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.nonzero();
}
inline Tensor dispatch_norm(const Tensor & self, Scalar p) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.norm(p);
}
inline Tensor & dispatch_norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::norm_out(result, self, p, dim, keepdim);
}
inline Tensor dispatch_norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.norm(p, dim, keepdim);
}
inline Tensor & dispatch_normal(const Tensor & mean, const Tensor & std, Generator * generator, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::normal_out(output, mean, std, generator);
}
inline Tensor dispatch_normal(const Tensor & mean, const Tensor & std, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(mean);
  return at::normal(mean, std, generator);
}
inline Tensor & dispatch_normal(const Tensor & mean, double std, Generator * generator, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::normal_out(output, mean, std, generator);
}
inline Tensor dispatch_normal(const Tensor & mean, double std, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(mean);
  return at::normal(mean, std, generator);
}
inline Tensor & dispatch_normal(double mean, const Tensor & std, Generator * generator, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::normal_out(output, mean, std, generator);
}
inline Tensor dispatch_normal(double mean, const Tensor & std, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(std);
  return at::normal(mean, std, generator);
}
inline int64_t dispatch_numel(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.numel();
}
inline Tensor & dispatch_ones(IntList size, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::ones_out(result, size);
}
inline Tensor dispatch_ones(IntList size, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::ones(dtype, size);
}
inline Tensor dispatch_ones_like(const Tensor & self, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device == -1 ? (self.type().is_cuda() ? self.get_device() : -1) : device);
  return at::ones_like(self, dtype);
}
inline Tensor dispatch_ones_like(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::ones_like(self);
}
inline Tensor & dispatch_orgqr(const Tensor & self, const Tensor & input2, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::orgqr_out(result, self, input2);
}
inline Tensor dispatch_orgqr(const Tensor & self, const Tensor & input2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.orgqr(input2);
}
inline Tensor & dispatch_ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::ormqr_out(result, self, input2, input3, left, transpose);
}
inline Tensor dispatch_ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ormqr(input2, input3, left, transpose);
}
inline Tensor dispatch_pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(x1);
  return at::pairwise_distance(x1, x2, p, eps, keepdim);
}
inline Tensor dispatch_pin_memory(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.pin_memory();
}
inline Tensor dispatch_poisson(const Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::poisson(self, generator);
}
inline Tensor & dispatch_polygamma(int64_t n, const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::polygamma_out(result, n, self);
}
inline Tensor dispatch_polygamma(int64_t n, const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.polygamma(n);
}
inline Tensor & dispatch_potrf(const Tensor & self, bool upper, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::potrf_out(output, self, upper);
}
inline Tensor dispatch_potrf(const Tensor & self, bool upper) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.potrf(upper);
}
inline Tensor & dispatch_potri(const Tensor & self, bool upper, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::potri_out(output, self, upper);
}
inline Tensor dispatch_potri(const Tensor & self, bool upper) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.potri(upper);
}
inline Tensor & dispatch_potrs(const Tensor & self, const Tensor & input2, bool upper, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::potrs_out(result, self, input2, upper);
}
inline Tensor dispatch_potrs(const Tensor & self, const Tensor & input2, bool upper) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.potrs(input2, upper);
}
inline Tensor & dispatch_pow(Scalar base, const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::pow_out(result, base, self);
}
inline Tensor dispatch_pow(Scalar base, const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::pow(base, self);
}
inline Tensor & dispatch_pow(const Tensor & self, Scalar exponent, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::pow_out(result, self, exponent);
}
inline Tensor dispatch_pow(const Tensor & self, Scalar exponent) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.pow(exponent);
}
inline Tensor & dispatch_pow(const Tensor & self, const Tensor & exponent, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::pow_out(result, self, exponent);
}
inline Tensor dispatch_pow(const Tensor & self, const Tensor & exponent) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.pow(exponent);
}
inline Tensor dispatch_prod(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.prod();
}
inline Tensor & dispatch_prod(const Tensor & self, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::prod_out(result, self, dim, keepdim);
}
inline Tensor dispatch_prod(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.prod(dim, keepdim);
}
inline std::tuple<Tensor &,Tensor &> dispatch_pstrf(const Tensor & self, bool upper, Scalar tol, Tensor & res1, Tensor & res2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(res1);
  return at::pstrf_out(res1, res2, self, upper, tol);
}
inline std::tuple<Tensor,Tensor> dispatch_pstrf(const Tensor & self, bool upper, Scalar tol) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.pstrf(upper, tol);
}
inline std::tuple<Tensor &,Tensor &> dispatch_qr(const Tensor & self, Tensor & res1, Tensor & res2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(res1);
  return at::qr_out(res1, res2, self);
}
inline std::tuple<Tensor,Tensor> dispatch_qr(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.qr();
}
inline Tensor & dispatch_rand(IntList size, Generator * generator, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::rand_out(result, size, generator);
}
inline Tensor dispatch_rand(IntList size, Generator * generator, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::rand(dtype, size, generator);
}
inline Tensor dispatch_rand_like(const Tensor & self, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device == -1 ? (self.type().is_cuda() ? self.get_device() : -1) : device);
  return at::rand_like(self, dtype);
}
inline Tensor dispatch_rand_like(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::rand_like(self);
}
inline Tensor & dispatch_randn(IntList size, Generator * generator, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::randn_out(result, size, generator);
}
inline Tensor dispatch_randn(IntList size, Generator * generator, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::randn(dtype, size, generator);
}
inline Tensor dispatch_randn_like(const Tensor & self, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device == -1 ? (self.type().is_cuda() ? self.get_device() : -1) : device);
  return at::randn_like(self, dtype);
}
inline Tensor dispatch_randn_like(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::randn_like(self);
}
inline Tensor & dispatch_randperm(int64_t n, Generator * generator, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::randperm_out(result, n, generator);
}
inline Tensor dispatch_randperm(int64_t n, Generator * generator, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::randperm(dtype, n, generator);
}
inline Tensor & dispatch_range(Scalar start, Scalar end, Scalar step, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::range_out(result, start, end, step);
}
inline Tensor dispatch_range(Scalar start, Scalar end, Scalar step, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::range(dtype, start, end, step);
}
inline Tensor & dispatch_reciprocal(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::reciprocal_out(result, self);
}
inline Tensor dispatch_reciprocal(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.reciprocal();
}
inline Tensor dispatch_relu(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.relu();
}
inline Tensor & dispatch_relu_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.relu_();
}
inline Tensor & dispatch_remainder(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::remainder_out(result, self, other);
}
inline Tensor dispatch_remainder(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.remainder(other);
}
inline Tensor & dispatch_remainder(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::remainder_out(result, self, other);
}
inline Tensor dispatch_remainder(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.remainder(other);
}
inline Tensor & dispatch_renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::renorm_out(result, self, p, dim, maxnorm);
}
inline Tensor dispatch_renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.renorm(p, dim, maxnorm);
}
inline Tensor dispatch_reshape(const Tensor & self, IntList shape) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.reshape(shape);
}
inline Tensor dispatch_rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.rfft(signal_ndim, normalized, onesided);
}
inline Tensor & dispatch_round(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::round_out(result, self);
}
inline Tensor dispatch_round(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.round();
}
inline Tensor & dispatch_round_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.round_();
}
inline Tensor dispatch_rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::rrelu(self, lower, upper, training, generator);
}
inline Tensor & dispatch_rrelu_(Tensor self, Scalar lower, Scalar upper, bool training, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::rrelu_(self, lower, upper, training, generator);
}
inline Tensor & dispatch_rsqrt(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::rsqrt_out(result, self);
}
inline Tensor dispatch_rsqrt(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.rsqrt();
}
inline Tensor dispatch_select(const Tensor & self, int64_t dim, int64_t index) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.select(dim, index);
}
inline Tensor dispatch_selu(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::selu(self);
}
inline Tensor & dispatch_selu_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::selu_(self);
}
inline Tensor & dispatch_sigmoid(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::sigmoid_out(result, self);
}
inline Tensor dispatch_sigmoid(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sigmoid();
}
inline Tensor & dispatch_sign(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::sign_out(result, self);
}
inline Tensor dispatch_sign(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sign();
}
inline Tensor & dispatch_sin(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::sin_out(result, self);
}
inline Tensor dispatch_sin(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sin();
}
inline Tensor & dispatch_sin_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sin_();
}
inline Tensor & dispatch_sinh(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::sinh_out(result, self);
}
inline Tensor dispatch_sinh(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sinh();
}
inline Tensor dispatch_slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.slice(dim, start, end, step);
}
inline std::tuple<Tensor,Tensor> dispatch_slogdet(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.slogdet();
}
inline Tensor dispatch_smm(const Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.smm(mat2);
}
inline std::tuple<Tensor &,Tensor &> dispatch_sort(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(values);
  return at::sort_out(values, indices, self, dim, descending);
}
inline std::tuple<Tensor,Tensor> dispatch_sort(const Tensor & self, int64_t dim, bool descending) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sort(dim, descending);
}
inline std::vector<Tensor> dispatch_split(const Tensor & self, int64_t split_size, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.split(split_size, dim);
}
inline std::vector<Tensor> dispatch_split_with_sizes(const Tensor & self, IntList split_sizes, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.split_with_sizes(split_sizes, dim);
}
inline Tensor & dispatch_sqrt(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::sqrt_out(result, self);
}
inline Tensor dispatch_sqrt(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sqrt();
}
inline Tensor & dispatch_sqrt_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sqrt_();
}
inline Tensor dispatch_squeeze(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.squeeze();
}
inline Tensor dispatch_squeeze(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.squeeze(dim);
}
inline Tensor dispatch_sspaddmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sspaddmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_sspaddmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sspaddmm(mat1, mat2, beta, 1);
}
inline Tensor & dispatch_sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::sspaddmm_out(result, self, mat1, mat2, beta, alpha);
}
inline Tensor dispatch_sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sspaddmm(mat1, mat2, beta, alpha);
}
inline Tensor & dispatch_stack(TensorList tensors, int64_t dim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::stack_out(result, tensors, dim);
}
inline Tensor dispatch_stack(TensorList tensors, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(tensors);
  return at::stack(tensors, dim);
}
inline Tensor dispatch_std(const Tensor & self, bool unbiased) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.std(unbiased);
}
inline Tensor & dispatch_std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::std_out(result, self, dim, unbiased, keepdim);
}
inline Tensor dispatch_std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.std(dim, unbiased, keepdim);
}
inline Tensor dispatch_stft(const Tensor & self, int64_t frame_length, int64_t hop, int64_t fft_size, bool normalized, bool onesided, const Tensor & window, int64_t pad_end) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.stft(frame_length, hop, fft_size, normalized, onesided, window, pad_end);
}
inline Tensor & dispatch_sub(const Tensor & self, Scalar alpha, const Tensor & other, Tensor out) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::sub_out(out, self, other, alpha);
}
inline Tensor dispatch_sub(const Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sub(other, alpha);
}
inline Tensor & dispatch_sub(const Tensor & self, Scalar other, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::sub_out(result, self, other, alpha);
}
inline Tensor dispatch_sub(const Tensor & self, Scalar other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sub(other, alpha);
}
inline Tensor & dispatch_sub(const Tensor & self, const Tensor & other, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::sub_out(result, self, other, alpha);
}
inline Tensor dispatch_sub(const Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sub(other, alpha);
}
inline Tensor dispatch_sum(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sum();
}
inline Tensor & dispatch_sum(const Tensor & self, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::sum_out(result, self, dim, keepdim);
}
inline Tensor dispatch_sum(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sum(dim, keepdim);
}
inline std::tuple<Tensor &,Tensor &,Tensor &> dispatch_svd(const Tensor & self, bool some, Tensor & res1, Tensor & res2, Tensor & res3) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(res1);
  return at::svd_out(res1, res2, res3, self, some);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_svd(const Tensor & self, bool some) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.svd(some);
}
inline std::tuple<Tensor &,Tensor &> dispatch_symeig(const Tensor & self, bool eigenvectors, bool upper, Tensor & res1, Tensor & res2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(res1);
  return at::symeig_out(res1, res2, self, eigenvectors, upper);
}
inline std::tuple<Tensor,Tensor> dispatch_symeig(const Tensor & self, bool eigenvectors, bool upper) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.symeig(eigenvectors, upper);
}
inline Tensor dispatch_t(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.t();
}
inline Tensor & dispatch_take(const Tensor & self, const Tensor & index, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::take_out(result, self, index);
}
inline Tensor dispatch_take(const Tensor & self, const Tensor & index) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.take(index);
}
inline Tensor & dispatch_tan(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::tan_out(result, self);
}
inline Tensor dispatch_tan(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.tan();
}
inline Tensor & dispatch_tanh(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::tanh_out(result, self);
}
inline Tensor dispatch_tanh(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.tanh();
}
inline std::tuple<Tensor &,Tensor &> dispatch_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(values);
  return at::topk_out(values, indices, self, k, dim, largest, sorted);
}
inline std::tuple<Tensor,Tensor> dispatch_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.topk(k, dim, largest, sorted);
}
inline Tensor dispatch_trace(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.trace();
}
inline Tensor dispatch_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.transpose(dim0, dim1);
}
inline Tensor & dispatch_tril(const Tensor & self, int64_t diagonal, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::tril_out(result, self, diagonal);
}
inline Tensor dispatch_tril(const Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.tril(diagonal);
}
inline Tensor dispatch_triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(anchor);
  return at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, size_average, reduce);
}
inline Tensor & dispatch_triu(const Tensor & self, int64_t diagonal, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::triu_out(result, self, diagonal);
}
inline Tensor dispatch_triu(const Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.triu(diagonal);
}
inline std::tuple<Tensor &,Tensor &> dispatch_trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular, Tensor & res1, Tensor & res2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(res1);
  return at::trtrs_out(res1, res2, self, A, upper, transpose, unitriangular);
}
inline std::tuple<Tensor,Tensor> dispatch_trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.trtrs(A, upper, transpose, unitriangular);
}
inline Tensor & dispatch_trunc(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::trunc_out(result, self);
}
inline Tensor dispatch_trunc(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.trunc();
}
inline Tensor & dispatch_trunc_(Tensor self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.trunc_();
}
inline Tensor dispatch_unsqueeze(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.unsqueeze(dim);
}
inline Tensor dispatch_var(const Tensor & self, bool unbiased) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.var(unbiased);
}
inline Tensor & dispatch_var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::var_out(result, self, dim, unbiased, keepdim);
}
inline Tensor dispatch_var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.var(dim, unbiased, keepdim);
}
inline Tensor dispatch_where(const Tensor & condition, const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(condition);
  return self.where(condition, other);
}
inline Tensor & dispatch_zeros(IntList size, Tensor result) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::zeros_out(result, size);
}
inline Tensor dispatch_zeros(IntList size, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return at::zeros(dtype, size);
}
inline Tensor dispatch_zeros_like(const Tensor & self, const Type & dtype, int64_t device) {
  maybe_initialize_cuda(dtype);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device == -1 ? (self.type().is_cuda() ? self.get_device() : -1) : device);
  return at::zeros_like(self, dtype);
}
inline Tensor dispatch_zeros_like(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::zeros_like(self);
}

}} // namespace torch::autograd
