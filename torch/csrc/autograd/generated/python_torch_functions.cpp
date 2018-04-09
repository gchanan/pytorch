// generated from tools/autograd/templates/python_torch_functions.cpp

// Python bindings for torch.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// torch._C._VariableFunctions which is also aliased as Variable._torch
// and also copied into 'torch' module.

#include <Python.h>

#include "torch/csrc/Dtype.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/utils/tensor_devices.h"
#include "torch/csrc/utils/tensor_layouts.h"

#include "python_torch_functions_dispatch.h"

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static Tensor set_requires_grad(Tensor self, bool requires_grad) {
  as_variable_ref(self).set_requires_grad(requires_grad);
  return self;
}

static void check_out_type_matches(Tensor result, ScalarType scalarType, const THPLayout& layout,
                                   const Device& device, bool device_is_none) {
  auto result_device_type = result.type().is_cuda() ? DeviceType::CUDA : DeviceType::CPU;
  auto device_type = device_is_none ? result_device_type : device.type;
  const auto& type = torch::getType(scalarType, layout, device_type);
  if (result.type() != type) {
    AT_ERROR(
        "type corresponding to %s does not match type of out parameter (%s)",
        type.toString(),
        result.type().toString());
  }
}

static Tensor dispatch_clamp(const Tensor & self, Scalar min, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp(min, max);
}
static Tensor dispatch_clamp_min(const Tensor & self, Scalar min) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_min(min);
}
static Tensor dispatch_clamp_max(const Tensor & self, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_max(max);
}

// The Python clamp() syntax has to be mapped to one of three C++ functions
static PyObject * THPVariable_clamp(PyObject* module, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp(Tensor input, Scalar min=None, Scalar max=None)",
  });
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (!r.isNone(1) && !r.isNone(2)) {
    return THPVariable_Wrap(dispatch_clamp(r.tensor(0), r.scalar(1), r.scalar(2)));
  } else if (!r.isNone(1)) {
    return THPVariable_Wrap(dispatch_clamp_min(r.tensor(0), r.scalar(1)));
  } else if (!r.isNone(2)) {
    return THPVariable_Wrap(dispatch_clamp_max(r.tensor(0), r.scalar(2)));
  } else {
    throw std::runtime_error("At least one of 'min' or 'max' must not be None");
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_from_numpy(PyObject* module, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto data = torch::utils::tensor_from_numpy(arg);
  return THPVariable_Wrap(make_variable(std::move(data), /*requires_grad=*/false));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable__promote_types(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_promote_types(ScalarType type1, ScalarType type2)",
  });
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    ScalarType promoted = at::promoteTypes(r.scalartype(0), r.scalartype(1));
    return torch::autograd::utils::wrap(torch::getDtype(promoted));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_sparse_coo_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  return THPVariable_Wrap(torch::utils::sparse_coo_tensor_ctor(default_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  return THPVariable_Wrap(torch::utils::tensor_ctor(default_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

// generated methods start here

static PyObject * THPVariable___and__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__and__(Tensor input, Scalar other, *, Tensor out=None)",
    "__and__(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch___and__(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch___and__(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch___and__(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch___and__(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___lshift__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__lshift__(Tensor input, Scalar other, *, Tensor out=None)",
    "__lshift__(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch___lshift__(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch___lshift__(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch___lshift__(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch___lshift__(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___or__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__or__(Tensor input, Scalar other, *, Tensor out=None)",
    "__or__(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch___or__(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch___or__(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch___or__(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch___or__(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___rshift__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__rshift__(Tensor input, Scalar other, *, Tensor out=None)",
    "__rshift__(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch___rshift__(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch___rshift__(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch___rshift__(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch___rshift__(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___xor__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__xor__(Tensor input, Scalar other, *, Tensor out=None)",
    "__xor__(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch___xor__(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch___xor__(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch___xor__(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch___xor__(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__abs(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_abs(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__abs(r.tensor(0)));
    } else {
      return wrap(dispatch__abs(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__addmv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_addmv(Tensor input, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch__addmv(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch__addmv(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__addr(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_addr(Tensor input, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch__addr(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch__addr(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__argmax(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_argmax(Tensor input, int64_t dim, bool keepdim=False)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__argmax(r.tensor(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__argmin(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_argmin(Tensor input, int64_t dim, bool keepdim=False)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__argmin(r.tensor(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_Half(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Half(Tensor input, bool non_blocking=False)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_Half(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_double(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_double(Tensor input, bool non_blocking=False)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_double(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_float(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_float(Tensor input, bool non_blocking=False)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_float(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_int(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_int(Tensor input, bool non_blocking=False)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_int(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_int16_t(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_int16_t(Tensor input, bool non_blocking=False)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_int16_t(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_int64_t(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_int64_t(Tensor input, bool non_blocking=False)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_int64_t(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_int8_t(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_int8_t(Tensor input, bool non_blocking=False)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_int8_t(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_uint8_t(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_uint8_t(Tensor input, bool non_blocking=False)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_uint8_t(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cat(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cat(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch__cat(r.tensorlist(0), r.toInt64(1)));
    } else {
      return wrap(dispatch__cat(r.tensorlist(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__ceil(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_ceil(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__ceil(r.tensor(0)));
    } else {
      return wrap(dispatch__ceil(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__convolution(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convolution(Tensor input, Tensor weight, Tensor? bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled)",
  });

  ParsedArgs<12> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__convolution(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toBool(6), r.intlist(7), r.toInt64(8), r.toBool(9), r.toBool(10), r.toBool(11)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__convolution_nogroup(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__convolution_nogroup(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toBool(6), r.intlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cos(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cos(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__cos(r.tensor(0)));
    } else {
      return wrap(dispatch__cos(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cudnn_init_dropout_state(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, *, ScalarType ty=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto device = r.device(5);
    return wrap(set_requires_grad(dispatch__cudnn_init_dropout_state(r.toDouble(0), r.toBool(1), r.toInt64(2), torch::getType(r.scalartype(3), r.layout(4), device.type), device.deviceInt64()), r.toBool(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cudnn_rnn(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_rnn(Tensor input, TensorList weight, int64_t weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, Tensor? dropout_state)",
  });

  ParsedArgs<15> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cudnn_rnn(r.tensor(0), r.tensorlist(1), r.toInt64(2), r.tensor(3), r.tensor(4), r.tensor(5), r.toInt64(6), r.toInt64(7), r.toInt64(8), r.toBool(9), r.toDouble(10), r.toBool(11), r.toBool(12), r.intlist(13), r.tensor(14)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cudnn_rnn_flatten_weight(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cudnn_rnn_flatten_weight(r.tensorlist(0), r.toInt64(1), r.toInt64(2), r.toInt64(3), r.toInt64(4), r.toInt64(5), r.toBool(6), r.toBool(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__dirichlet_grad(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dirichlet_grad(Tensor x, Tensor alpha, Tensor total, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch__dirichlet_grad(r.tensor(0), r.tensor(1), r.tensor(2)));
    } else {
      return wrap(dispatch__dirichlet_grad(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__dot(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dot(Tensor input, Tensor tensor)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__dot(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__exp(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_exp(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__exp(r.tensor(0)));
    } else {
      return wrap(dispatch__exp(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__fft_with_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fft_with_size(Tensor input, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes)",
  });

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__fft_with_size(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3), r.toBool(4), r.intlist(5), r.toBool(6), r.toBool(7), r.intlist(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__floor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_floor(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__floor(r.tensor(0)));
    } else {
      return wrap(dispatch__floor(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__ger(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_ger(Tensor input, Tensor vec2, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch__ger(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch__ger(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__log(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_log(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__log(r.tensor(0)));
    } else {
      return wrap(dispatch__log(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__mm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mm(Tensor input, Tensor mat2, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch__mm(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch__mm(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__mv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mv(Tensor input, Tensor vec, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch__mv(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch__mv(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__prod(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_prod(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch__prod(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      return wrap(dispatch__prod(r.tensor(0), r.toInt64(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__prodall(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_prodall(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__prodall(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__round(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_round(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__round(r.tensor(0)));
    } else {
      return wrap(dispatch__round(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__s_where(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_s_where(Tensor condition, Tensor input, Tensor other)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__s_where(r.tensor(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__sin(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sin(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__sin(r.tensor(0)));
    } else {
      return wrap(dispatch__sin(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__sqrt(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sqrt(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__sqrt(r.tensor(0)));
    } else {
      return wrap(dispatch__sqrt(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__standard_gamma(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_standard_gamma(Tensor input, *, Generator generator=None, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch__standard_gamma(r.tensor(0), r.generator(1)));
    } else {
      return wrap(dispatch__standard_gamma(r.tensor(0), r.generator(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__standard_gamma_grad(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_standard_gamma_grad(Tensor input, Tensor output)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__standard_gamma_grad(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__sum(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sum(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch__sum(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      return wrap(dispatch__sum(r.tensor(0), r.toInt64(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__sumall(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sumall(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__sumall(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__trunc(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_trunc(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__trunc(r.tensor(0)));
    } else {
      return wrap(dispatch__trunc(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__unique(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unique(Tensor input, bool sorted=False, bool return_inverse=False)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__unique(r.tensor(0), r.toBool(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_abs(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "abs(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_abs(r.tensor(0)));
    } else {
      return wrap(dispatch_abs(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_abs_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "abs_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_abs_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_acos(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "acos(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_acos(r.tensor(0)));
    } else {
      return wrap(dispatch_acos(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_adaptive_avg_pool1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_avg_pool1d(Tensor input, IntList[1] output_size)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_adaptive_avg_pool1d(r.tensor(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_adaptive_max_pool1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_max_pool1d(Tensor input, IntList[1] output_size)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_adaptive_max_pool1d(r.tensor(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_add(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "add(Tensor input, Scalar alpha, Tensor other, *, Tensor out=None)|deprecated",
    "add(Tensor input, Scalar other, *, Scalar alpha=1, Tensor out=None)",
    "add(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_add(r.tensor(0), r.scalar(1), r.tensor(2)));
    } else {
      return wrap(dispatch_add(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_add(r.tensor(0), r.scalar(1), r.scalar(2)));
    } else {
      return wrap(dispatch_add(r.tensor(0), r.scalar(1), r.scalar(2), r.tensor(3)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_add(r.tensor(0), r.tensor(1), r.scalar(2)));
    } else {
      return wrap(dispatch_add(r.tensor(0), r.tensor(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addbmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addbmm(Scalar beta, Tensor input, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "addbmm(Scalar beta, Tensor input, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "addbmm(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_addbmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
    } else {
      return wrap(dispatch_addbmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4), r.tensor(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addbmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addbmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_addbmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_addbmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcdiv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcdiv(Tensor input, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor out=None)|deprecated",
    "addcdiv(Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_addcdiv(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addcdiv(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addcdiv(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3)));
    } else {
      return wrap(dispatch_addcdiv(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcmul(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcmul(Tensor input, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor out=None)|deprecated",
    "addcmul(Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_addcmul(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addcmul(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addcmul(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3)));
    } else {
      return wrap(dispatch_addcmul(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmm(Scalar beta, Tensor input, Scalar alpha, Tensor mat1, Tensor mat2, *, Tensor out=None)|deprecated",
    "addmm(Scalar beta, Tensor input, Tensor mat1, Tensor mat2, *, Tensor out=None)|deprecated",
    "addmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_addmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
    } else {
      return wrap(dispatch_addmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4), r.tensor(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_addmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_addmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmv(Scalar beta, Tensor input, Scalar alpha, Tensor mat, Tensor vec, *, Tensor out=None)|deprecated",
    "addmv(Scalar beta, Tensor input, Tensor mat, Tensor vec, *, Tensor out=None)|deprecated",
    "addmv(Tensor input, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_addmv(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
    } else {
      return wrap(dispatch_addmv(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4), r.tensor(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addmv(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addmv(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_addmv(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_addmv(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmv_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmv_(Scalar beta, Tensor input, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Scalar beta, Tensor input, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Tensor input, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addmv_(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addmv_(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addmv_(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addr(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addr(Scalar beta, Tensor input, Scalar alpha, Tensor vec1, Tensor vec2, *, Tensor out=None)|deprecated",
    "addr(Scalar beta, Tensor input, Tensor vec1, Tensor vec2, *, Tensor out=None)|deprecated",
    "addr(Tensor input, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_addr(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
    } else {
      return wrap(dispatch_addr(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4), r.tensor(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addr(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addr(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_addr(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_addr(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_allclose(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "allclose(Tensor input, Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_allclose(r.tensor(0), r.tensor(1), r.toDouble(2), r.toDouble(3), r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_arange(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arange(Scalar end, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "arange(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto device = r.device(4);
      return wrap(set_requires_grad(dispatch_arange(r.scalar(0), torch::getType(r.scalartype(2), r.layout(3), device.type), device.deviceInt64()), r.toBool(5)));
    } else {
      if (!r.isNone(2)) {
        check_out_type_matches(r.tensor(1), r.scalartype(2), r.layout(3),
                               r.device(4), r.isNone(4));
      }
      return wrap(set_requires_grad(dispatch_arange(r.scalar(0), r.tensor(1)), r.toBool(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      auto device = r.device(6);
      return wrap(set_requires_grad(dispatch_arange(r.scalar(0), r.scalar(1), r.scalar(2), torch::getType(r.scalartype(4), r.layout(5), device.type), device.deviceInt64()), r.toBool(7)));
    } else {
      if (!r.isNone(4)) {
        check_out_type_matches(r.tensor(3), r.scalartype(4), r.layout(5),
                               r.device(6), r.isNone(6));
      }
      return wrap(set_requires_grad(dispatch_arange(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)), r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_as_strided(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided(Tensor input, IntList size, IntList stride, int64_t storage_offset=-1, *, Tensor out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_as_strided(r.tensor(0), r.intlist(1), r.intlist(2), r.toInt64(3)));
    } else {
      return wrap(dispatch_as_strided(r.tensor(0), r.intlist(1), r.intlist(2), r.toInt64(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_asin(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "asin(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_asin(r.tensor(0)));
    } else {
      return wrap(dispatch_asin(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_atan(r.tensor(0)));
    } else {
      return wrap(dispatch_atan(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan2(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan2(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_atan2(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_atan2(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_baddbmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "baddbmm(Scalar beta, Tensor input, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "baddbmm(Scalar beta, Tensor input, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "baddbmm(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_baddbmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
    } else {
      return wrap(dispatch_baddbmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4), r.tensor(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_baddbmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_baddbmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_baddbmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_baddbmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_batch_norm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double momentum, double eps, bool cudnn_enabled)",
  });

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_batch_norm(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.toBool(5), r.toDouble(6), r.toDouble(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bernoulli(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bernoulli(Tensor input, *, Generator generator=None, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_bernoulli(r.tensor(0), r.generator(1)));
    } else {
      return wrap(dispatch_bernoulli(r.tensor(0), r.generator(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bernoulli_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bernoulli_(Tensor input, Tensor p, Generator generator=None)",
    "bernoulli_(Tensor input, double p=0.5, Generator generator=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_bernoulli_(r.tensor(0), r.tensor(1), r.generator(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_bernoulli_(r.tensor(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bilinear(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_bilinear(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bmm(Tensor input, Tensor mat2, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_bmm(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_bmm(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_btrifact(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "btrifact(Tensor input, *, bool pivot=True, TensorList[2] out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_btrifact(r.tensor(0), r.toBool(1)));
    } else {
      auto results = r.tensorlist_n<2>(2);
      return wrap(dispatch_btrifact(r.tensor(0), r.toBool(1), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_btrifact_with_info(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "btrifact_with_info(Tensor input, *, bool pivot=True, TensorList[3] out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_btrifact_with_info(r.tensor(0), r.toBool(1)));
    } else {
      auto results = r.tensorlist_n<3>(2);
      return wrap(dispatch_btrifact_with_info(r.tensor(0), r.toBool(1), results[0], results[1], results[2]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_btrisolve(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "btrisolve(Tensor input, Tensor LU_data, Tensor LU_pivots, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_btrisolve(r.tensor(0), r.tensor(1), r.tensor(2)));
    } else {
      return wrap(dispatch_btrisolve(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cat(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cat(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_cat(r.tensorlist(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_cat(r.tensorlist(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ceil(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ceil(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_ceil(r.tensor(0)));
    } else {
      return wrap(dispatch_ceil(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ceil_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ceil_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ceil_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_chunk(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "chunk(Tensor input, int64_t chunks, int64_t dim=0)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_chunk(r.tensor(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv1d(Tensor input, Tensor weight, Tensor bias=None, IntList[1] stride=1, IntList[1] padding=0, IntList[1] dilation=1, int64_t groups=1)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv1d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv2d(Tensor input, Tensor weight, Tensor bias=None, IntList[2] stride=1, IntList[2] padding=0, IntList[2] dilation=1, int64_t groups=1)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv2d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv3d(Tensor input, Tensor weight, Tensor bias=None, IntList[3] stride=1, IntList[3] padding=0, IntList[3] dilation=1, int64_t groups=1)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv3d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv_tbc(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_tbc(Tensor input, Tensor weight, Tensor bias, int64_t pad)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv_tbc(r.tensor(0), r.tensor(1), r.tensor(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv_transpose1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose1d(Tensor input, Tensor weight, Tensor bias=None, IntList[1] stride=1, IntList[1] padding=0, IntList[1] output_padding=0, int64_t groups=1, IntList[1] dilation=1)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv_transpose1d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6), r.intlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv_transpose2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose2d(Tensor input, Tensor weight, Tensor bias=None, IntList[2] stride=1, IntList[2] padding=0, IntList[2] output_padding=0, int64_t groups=1, IntList[2] dilation=1)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv_transpose2d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6), r.intlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv_transpose3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose3d(Tensor input, Tensor weight, Tensor bias=None, IntList[3] stride=1, IntList[3] padding=0, IntList[3] output_padding=0, int64_t groups=1, IntList[3] dilation=1)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv_transpose3d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6), r.intlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_convolution(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "convolution(Tensor input, Tensor weight, Tensor? bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups)",
  });

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_convolution(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toBool(6), r.intlist(7), r.toInt64(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cos(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cos(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_cos(r.tensor(0)));
    } else {
      return wrap(dispatch_cos(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cos_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cos_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cos_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cosh(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosh(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_cosh(r.tensor(0)));
    } else {
      return wrap(dispatch_cosh(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cosine_embedding_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, double margin=0.0, bool size_average=True, bool reduce=True)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cosine_embedding_loss(r.tensor(0), r.tensor(1), r.tensor(2), r.toDouble(3), r.toBool(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cross(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cross(Tensor input, Tensor other, int64_t dim=-1, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_cross(r.tensor(0), r.tensor(1), r.toInt64(2)));
    } else {
      return wrap(dispatch_cross(r.tensor(0), r.tensor(1), r.toInt64(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_affine_grid_generator(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_affine_grid_generator(Tensor theta, int64_t N, int64_t C, int64_t H, int64_t W)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_affine_grid_generator(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toInt64(3), r.toInt64(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_batch_norm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double exponential_average_factor, double epsilon)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_batch_norm(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.toBool(5), r.toDouble(6), r.toDouble(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_convolution(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_convolution(Tensor input, Tensor weight, Tensor? bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic)",
  });

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_convolution(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6), r.toBool(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_convolution_transpose(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_convolution_transpose(Tensor input, Tensor weight, Tensor? bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic)",
  });

  ParsedArgs<10> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_convolution_transpose(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.intlist(6), r.toInt64(7), r.toBool(8), r.toBool(9)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_grid_sampler(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_grid_sampler(Tensor input, Tensor grid)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_grid_sampler(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_is_acceptable(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_is_acceptable(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_is_acceptable(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cumprod(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumprod(Tensor input, int64_t dim, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_cumprod(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_cumprod(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cumsum(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumsum(Tensor input, int64_t dim, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_cumsum(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_cumsum(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_det(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "det(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_det(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diag(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diag(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_diag(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_diag(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diagflat(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagflat(Tensor input, int64_t offset=0)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_diagflat(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diagonal(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagonal(Tensor input, int64_t offset=0)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_diagonal(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_digamma(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "digamma(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_digamma(r.tensor(0)));
    } else {
      return wrap(dispatch_digamma(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dist(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dist(Tensor input, Tensor other, Scalar p=2)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_dist(r.tensor(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_div(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "div(Tensor input, Scalar other, *, Tensor out=None)",
    "div(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_div(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_div(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_div(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_div(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dot(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dot(Tensor input, Tensor tensor)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_dot(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eig(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eig(Tensor input, bool eigenvectors=False, *, TensorList[2] out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_eig(r.tensor(0), r.toBool(1)));
    } else {
      auto results = r.tensorlist_n<2>(2);
      return wrap(dispatch_eig(r.tensor(0), r.toBool(1), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_embedding(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding(Tensor weight, Tensor indices, int64_t padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_embedding(r.tensor(0), r.tensor(1), r.toInt64(2), r.toBool(3), r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_embedding_bag(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int64_t mode=0, bool sparse=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_embedding_bag(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toInt64(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_embedding_renorm_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding_renorm_(Tensor input, Tensor indices, double max_norm, double norm_type)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_embedding_renorm_(r.tensor(0), r.tensor(1), r.toDouble(2), r.toDouble(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_empty(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty(IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto device = r.device(4);
      return wrap(set_requires_grad(dispatch_empty(r.intlist(0), torch::getType(r.scalartype(2), r.layout(3), device.type), device.deviceInt64()), r.toBool(5)));
    } else {
      if (!r.isNone(2)) {
        check_out_type_matches(r.tensor(1), r.scalartype(2), r.layout(3),
                               r.device(4), r.isNone(4));
      }
      return wrap(set_requires_grad(dispatch_empty(r.intlist(0), r.tensor(1)), r.toBool(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_empty_like(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty_like(Tensor input, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "empty_like(Tensor input, *, bool requires_grad=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto dtype = r.scalartypeWithDefault(1, self.type().scalarType());
    auto layout = r.layoutWithDefault(2, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(3, torch::utils::getDevice(self));
    return wrap(set_requires_grad(dispatch_empty_like(self, torch::getType(dtype, layout, device.type), device.deviceInt64()), r.toBool(4)));
  } else if (r.idx == 1) {
    return wrap(set_requires_grad(dispatch_empty_like(r.tensor(0)), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eq(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eq(Tensor input, Scalar other, *, Tensor out=None)",
    "eq(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_eq(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_eq(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_eq(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_eq(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_equal(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "equal(Tensor input, Tensor other)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_equal(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erf(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erf(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_erf(r.tensor(0)));
    } else {
      return wrap(dispatch_erf(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erfinv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erfinv(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_erfinv(r.tensor(0)));
    } else {
      return wrap(dispatch_erfinv(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_exp(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exp(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_exp(r.tensor(0)));
    } else {
      return wrap(dispatch_exp(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_exp_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exp_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_exp_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expm1(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expm1(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_expm1(r.tensor(0)));
    } else {
      return wrap(dispatch_expm1(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eye(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eye(int64_t n, int64_t m=-1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      auto device = r.device(5);
      return wrap(set_requires_grad(dispatch_eye(r.toInt64(0), r.toInt64(1), torch::getType(r.scalartype(3), r.layout(4), device.type), device.deviceInt64()), r.toBool(6)));
    } else {
      if (!r.isNone(3)) {
        check_out_type_matches(r.tensor(2), r.scalartype(3), r.layout(4),
                               r.device(5), r.isNone(5));
      }
      return wrap(set_requires_grad(dispatch_eye(r.toInt64(0), r.toInt64(1), r.tensor(2)), r.toBool(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fft(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft(Tensor input, int64_t signal_ndim, bool normalized=False)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_fft(r.tensor(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_floor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "floor(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_floor(r.tensor(0)));
    } else {
      return wrap(dispatch_floor(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_floor_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "floor_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_floor_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fmod(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fmod(Tensor input, Scalar other, *, Tensor out=None)",
    "fmod(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_fmod(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_fmod(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_fmod(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_fmod(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_frac(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "frac(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_frac(r.tensor(0)));
    } else {
      return wrap(dispatch_frac(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_full(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "full(IntList size, Scalar fill_value, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      auto device = r.device(5);
      return wrap(set_requires_grad(dispatch_full(r.intlist(0), r.scalar(1), torch::getType(r.scalartype(3), r.layout(4), device.type), device.deviceInt64()), r.toBool(6)));
    } else {
      if (!r.isNone(3)) {
        check_out_type_matches(r.tensor(2), r.scalartype(3), r.layout(4),
                               r.device(5), r.isNone(5));
      }
      return wrap(set_requires_grad(dispatch_full(r.intlist(0), r.scalar(1), r.tensor(2)), r.toBool(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_full_like(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "full_like(Tensor input, Scalar fill_value, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "full_like(Tensor input, Scalar fill_value, *, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto fill_value = r.scalar(1);
    auto dtype = r.scalartypeWithDefault(2, self.type().scalarType());
    auto layout = r.layoutWithDefault(3, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(4, torch::utils::getDevice(self));
    return wrap(set_requires_grad(dispatch_full_like(self, fill_value, torch::getType(dtype, layout, device.type), device.deviceInt64()), r.toBool(5)));
  } else if (r.idx == 1) {
    return wrap(set_requires_grad(dispatch_full_like(r.tensor(0), r.scalar(1)), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gather(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gather(Tensor input, int64_t dim, Tensor index, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_gather(r.tensor(0), r.toInt64(1), r.tensor(2)));
    } else {
      return wrap(dispatch_gather(r.tensor(0), r.toInt64(1), r.tensor(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ge(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ge(Tensor input, Scalar other, *, Tensor out=None)",
    "ge(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_ge(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_ge(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_ge(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_ge(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gels(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gels(Tensor input, Tensor A, *, TensorList[2] out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_gels(r.tensor(0), r.tensor(1)));
    } else {
      auto results = r.tensorlist_n<2>(2);
      return wrap(dispatch_gels(r.tensor(0), r.tensor(1), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_geqrf(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "geqrf(Tensor input, *, TensorList[2] out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_geqrf(r.tensor(0)));
    } else {
      auto results = r.tensorlist_n<2>(1);
      return wrap(dispatch_geqrf(r.tensor(0), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ger(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ger(Tensor input, Tensor vec2, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_ger(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_ger(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gesv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gesv(Tensor input, Tensor A, *, TensorList[2] out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_gesv(r.tensor(0), r.tensor(1)));
    } else {
      auto results = r.tensorlist_n<2>(2);
      return wrap(dispatch_gesv(r.tensor(0), r.tensor(1), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_group_norm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "group_norm(Tensor input, int64_t num_groups, Tensor? weight=None, Tensor? bias=None, double eps=1e-05, bool cudnn_enabled=True)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_group_norm(r.tensor(0), r.toInt64(1), r.tensor(2), r.tensor(3), r.toDouble(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gt(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gt(Tensor input, Scalar other, *, Tensor out=None)",
    "gt(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_gt(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_gt(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_gt(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_gt(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hinge_embedding_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hinge_embedding_loss(Tensor input, Tensor target, double margin=1.0, bool size_average=True, bool reduce=True)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_hinge_embedding_loss(r.tensor(0), r.tensor(1), r.toDouble(2), r.toBool(3), r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_histc(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "histc(Tensor input, int64_t bins=100, Scalar min=0, Scalar max=0, *, Tensor out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_histc(r.tensor(0), r.toInt64(1), r.scalar(2), r.scalar(3)));
    } else {
      return wrap(dispatch_histc(r.tensor(0), r.toInt64(1), r.scalar(2), r.scalar(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hspmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hspmm(Tensor mat1, Tensor mat2, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_hspmm(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_hspmm(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ifft(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ifft(Tensor input, int64_t signal_ndim, bool normalized=False)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ifft(r.tensor(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index(Tensor input, TensorList indices)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_index(r.tensor(0), r.tensorlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_put_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_put_(Tensor input, TensorList indices, Tensor values)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_index_put_(r.tensor(0), r.tensorlist(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_select(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_select(Tensor input, int64_t dim, Tensor index, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_index_select(r.tensor(0), r.toInt64(1), r.tensor(2)));
    } else {
      return wrap(dispatch_index_select(r.tensor(0), r.toInt64(1), r.tensor(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_inverse(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "inverse(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_inverse(r.tensor(0)));
    } else {
      return wrap(dispatch_inverse(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_irfft(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "irfft(Tensor input, int64_t signal_ndim, IntList signal_sizes=None, bool normalized=False, bool onesided=True)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_irfft(r.tensor(0), r.toInt64(1), r.intlist(2), r.toBool(3), r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_distributed(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_distributed(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_distributed(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_floating_point(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_floating_point(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_floating_point(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_nonzero(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_nonzero(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_nonzero(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_same_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_same_size(Tensor input, Tensor other)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_same_size(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_signed(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_signed(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_signed(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_isclose(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isclose(Tensor input, Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_isclose(r.tensor(0), r.tensor(1), r.toDouble(2), r.toDouble(3), r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_kthvalue(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "kthvalue(Tensor input, int64_t k, int64_t dim=-1, bool keepdim=False, *, TensorList[2] out=None)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_kthvalue(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toBool(3)));
    } else {
      auto results = r.tensorlist_n<2>(4);
      return wrap(dispatch_kthvalue(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toBool(3), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_layer_norm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "layer_norm(Tensor input, IntList normalized_shape, Tensor? weight=None, Tensor? bias=None, double eps=1e-05, bool cudnn_enable=True)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_layer_norm(r.tensor(0), r.intlist(1), r.tensor(2), r.tensor(3), r.toDouble(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_le(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "le(Tensor input, Scalar other, *, Tensor out=None)",
    "le(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_le(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_le(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_le(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_le(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lerp(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lerp(Tensor input, Tensor end, Scalar weight, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_lerp(r.tensor(0), r.tensor(1), r.scalar(2)));
    } else {
      return wrap(dispatch_lerp(r.tensor(0), r.tensor(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lgamma(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lgamma(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_lgamma(r.tensor(0)));
    } else {
      return wrap(dispatch_lgamma(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_linspace(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linspace(Scalar start, Scalar end, int64_t steps=100, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      auto device = r.device(6);
      return wrap(set_requires_grad(dispatch_linspace(r.scalar(0), r.scalar(1), r.toInt64(2), torch::getType(r.scalartype(4), r.layout(5), device.type), device.deviceInt64()), r.toBool(7)));
    } else {
      if (!r.isNone(4)) {
        check_out_type_matches(r.tensor(3), r.scalartype(4), r.layout(5),
                               r.device(6), r.isNone(6));
      }
      return wrap(set_requires_grad(dispatch_linspace(r.scalar(0), r.scalar(1), r.toInt64(2), r.tensor(3)), r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_log(r.tensor(0)));
    } else {
      return wrap(dispatch_log(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log10(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log10(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_log10(r.tensor(0)));
    } else {
      return wrap(dispatch_log10(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log1p(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log1p(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_log1p(r.tensor(0)));
    } else {
      return wrap(dispatch_log1p(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log2(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log2(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_log2(r.tensor(0)));
    } else {
      return wrap(dispatch_log2(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_log_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logdet(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logdet(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_logdet(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logspace(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logspace(Scalar start, Scalar end, int64_t steps=100, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      auto device = r.device(6);
      return wrap(set_requires_grad(dispatch_logspace(r.scalar(0), r.scalar(1), r.toInt64(2), torch::getType(r.scalartype(4), r.layout(5), device.type), device.deviceInt64()), r.toBool(7)));
    } else {
      if (!r.isNone(4)) {
        check_out_type_matches(r.tensor(3), r.scalartype(4), r.layout(5),
                               r.device(6), r.isNone(6));
      }
      return wrap(set_requires_grad(dispatch_logspace(r.scalar(0), r.scalar(1), r.toInt64(2), r.tensor(3)), r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lt(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lt(Tensor input, Scalar other, *, Tensor out=None)",
    "lt(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_lt(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_lt(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_lt(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_lt(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_margin_ranking_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, double margin=0.0, bool size_average=True, bool reduce=True)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_margin_ranking_loss(r.tensor(0), r.tensor(1), r.tensor(2), r.toDouble(3), r.toBool(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_masked_select(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_select(Tensor input, Tensor mask, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_masked_select(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_masked_select(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_matmul(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matmul(Tensor input, Tensor other)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_matmul(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_max(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max(Tensor input)",
    "max(Tensor input, Tensor other, *, Tensor out=None)",
    "max(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_max(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_max(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_max(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_max(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_max(r.tensor(0), r.toInt64(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_max_pool1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool1d(Tensor input, IntList[1] kernel_size, IntList[1] stride=None, IntList[1] padding=0, IntList[1] dilation=1, bool ceil_mode=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_max_pool1d(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.intlist(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mean(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mean(Tensor input)",
    "mean(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mean(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_mean(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      return wrap(dispatch_mean(r.tensor(0), r.toInt64(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_median(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "median(Tensor input)",
    "median(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_median(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_median(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_median(r.tensor(0), r.toInt64(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_min(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "min(Tensor input)",
    "min(Tensor input, Tensor other, *, Tensor out=None)",
    "min(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_min(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_min(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_min(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_min(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_min(r.tensor(0), r.toInt64(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mkldnn_convolution(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_convolution(Tensor input, Tensor weight, Tensor? bias, IntList padding, IntList stride, IntList dilation)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mkldnn_convolution(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mkldnn_convolution_backward_weights(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_convolution_backward_weights(IntList weight_size, Tensor grad_output, Tensor input, IntList padding, IntList stride, IntList dilation, bool bias_defined)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mkldnn_convolution_backward_weights(r.intlist(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toBool(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mm(Tensor input, Tensor mat2, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_mm(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_mm(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mode(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mode(Tensor input, int64_t dim=-1, bool keepdim=False, *, TensorList[2] out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_mode(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_mode(r.tensor(0), r.toInt64(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mul(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mul(Tensor input, Scalar other, *, Tensor out=None)",
    "mul(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_mul(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_mul(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_mul(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_mul(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_multinomial(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multinomial(Tensor input, int64_t num_samples, bool replacement=False, *, Generator generator=None, Tensor out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_multinomial(r.tensor(0), r.toInt64(1), r.toBool(2), r.generator(3)));
    } else {
      return wrap(dispatch_multinomial(r.tensor(0), r.toInt64(1), r.toBool(2), r.generator(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mv(Tensor input, Tensor vec, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_mv(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_mv(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_narrow(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "narrow(Tensor input, int64_t dim, int64_t start, int64_t length)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_narrow(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ne(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ne(Tensor input, Scalar other, *, Tensor out=None)",
    "ne(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_ne(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_ne(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_ne(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_ne(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_neg(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "neg(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_neg(r.tensor(0)));
    } else {
      return wrap(dispatch_neg(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nonzero(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_nonzero(r.tensor(0)));
    } else {
      return wrap(dispatch_nonzero(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_norm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "norm(Tensor input, Scalar p=2)",
    "norm(Tensor input, Scalar p=None, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_norm(r.tensor(0), r.scalar(1)));
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      auto self = r.tensor(0);
      auto p = r.scalarWithDefault(1, 2);
      auto dim = r.toInt64(2);
      auto keepdim = r.toBool(3);
      return wrap(dispatch_norm(self, p, dim, keepdim));
    } else {
      auto self = r.tensor(0);
      auto p = r.scalarWithDefault(1, 2);
      auto dim = r.toInt64(2);
      auto keepdim = r.toBool(3);
      return wrap(dispatch_norm(self, p, dim, keepdim, r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_normal(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "normal(Tensor mean, Tensor std, *, Generator generator=None, Tensor out=None)",
    "normal(Tensor mean, double std=1, *, Generator generator=None, Tensor out=None)",
    "normal(double mean, Tensor std, *, Generator generator=None, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_normal(r.tensor(0), r.tensor(1), r.generator(2)));
    } else {
      return wrap(dispatch_normal(r.tensor(0), r.tensor(1), r.generator(2), r.tensor(3)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_normal(r.tensor(0), r.toDouble(1), r.generator(2)));
    } else {
      return wrap(dispatch_normal(r.tensor(0), r.toDouble(1), r.generator(2), r.tensor(3)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_normal(r.toDouble(0), r.tensor(1), r.generator(2)));
    } else {
      return wrap(dispatch_normal(r.toDouble(0), r.tensor(1), r.generator(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_numel(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "numel(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_numel(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ones(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ones(IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto device = r.device(4);
      return wrap(set_requires_grad(dispatch_ones(r.intlist(0), torch::getType(r.scalartype(2), r.layout(3), device.type), device.deviceInt64()), r.toBool(5)));
    } else {
      if (!r.isNone(2)) {
        check_out_type_matches(r.tensor(1), r.scalartype(2), r.layout(3),
                               r.device(4), r.isNone(4));
      }
      return wrap(set_requires_grad(dispatch_ones(r.intlist(0), r.tensor(1)), r.toBool(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ones_like(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ones_like(Tensor input, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "ones_like(Tensor input, *, bool requires_grad=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto dtype = r.scalartypeWithDefault(1, self.type().scalarType());
    auto layout = r.layoutWithDefault(2, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(3, torch::utils::getDevice(self));
    return wrap(set_requires_grad(dispatch_ones_like(self, torch::getType(dtype, layout, device.type), device.deviceInt64()), r.toBool(4)));
  } else if (r.idx == 1) {
    return wrap(set_requires_grad(dispatch_ones_like(r.tensor(0)), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_orgqr(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "orgqr(Tensor input, Tensor input2, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_orgqr(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_orgqr(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ormqr(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ormqr(Tensor input, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor out=None)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_ormqr(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toBool(4)));
    } else {
      return wrap(dispatch_ormqr(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toBool(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pairwise_distance(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pairwise_distance(Tensor x1, Tensor x2, double p=2, double eps=1e-06, bool keepdim=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_pairwise_distance(r.tensor(0), r.tensor(1), r.toDouble(2), r.toDouble(3), r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pin_memory(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pin_memory(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_pin_memory(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_poisson(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "poisson(Tensor input, Generator generator=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_poisson(r.tensor(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_polygamma(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "polygamma(int64_t n, Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_polygamma(r.toInt64(0), r.tensor(1)));
    } else {
      return wrap(dispatch_polygamma(r.toInt64(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_potrf(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "potrf(Tensor input, bool upper=True, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_potrf(r.tensor(0), r.toBool(1)));
    } else {
      return wrap(dispatch_potrf(r.tensor(0), r.toBool(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_potri(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "potri(Tensor input, bool upper=True, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_potri(r.tensor(0), r.toBool(1)));
    } else {
      return wrap(dispatch_potri(r.tensor(0), r.toBool(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_potrs(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "potrs(Tensor input, Tensor input2, bool upper=True, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_potrs(r.tensor(0), r.tensor(1), r.toBool(2)));
    } else {
      return wrap(dispatch_potrs(r.tensor(0), r.tensor(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pow(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pow(Scalar base, Tensor input, *, Tensor out=None)",
    "pow(Tensor input, Scalar exponent, *, Tensor out=None)",
    "pow(Tensor input, Tensor exponent, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_pow(r.scalar(0), r.tensor(1)));
    } else {
      return wrap(dispatch_pow(r.scalar(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_pow(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_pow(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(2)) {
      return wrap(dispatch_pow(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_pow(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_prod(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prod(Tensor input)",
    "prod(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_prod(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_prod(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      return wrap(dispatch_prod(r.tensor(0), r.toInt64(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pstrf(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pstrf(Tensor input, bool upper=True, Scalar tol=-1, *, TensorList[2] out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_pstrf(r.tensor(0), r.toBool(1), r.scalar(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_pstrf(r.tensor(0), r.toBool(1), r.scalar(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_qr(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "qr(Tensor input, *, TensorList[2] out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_qr(r.tensor(0)));
    } else {
      auto results = r.tensorlist_n<2>(1);
      return wrap(dispatch_qr(r.tensor(0), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rand(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rand(IntList size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      auto device = r.device(5);
      return wrap(set_requires_grad(dispatch_rand(r.intlist(0), r.generator(1), torch::getType(r.scalartype(3), r.layout(4), device.type), device.deviceInt64()), r.toBool(6)));
    } else {
      if (!r.isNone(3)) {
        check_out_type_matches(r.tensor(2), r.scalartype(3), r.layout(4),
                               r.device(5), r.isNone(5));
      }
      return wrap(set_requires_grad(dispatch_rand(r.intlist(0), r.generator(1), r.tensor(2)), r.toBool(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rand_like(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rand_like(Tensor input, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "rand_like(Tensor input, *, bool requires_grad=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto dtype = r.scalartypeWithDefault(1, self.type().scalarType());
    auto layout = r.layoutWithDefault(2, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(3, torch::utils::getDevice(self));
    return wrap(set_requires_grad(dispatch_rand_like(self, torch::getType(dtype, layout, device.type), device.deviceInt64()), r.toBool(4)));
  } else if (r.idx == 1) {
    return wrap(set_requires_grad(dispatch_rand_like(r.tensor(0)), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_randn(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randn(IntList size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      auto device = r.device(5);
      return wrap(set_requires_grad(dispatch_randn(r.intlist(0), r.generator(1), torch::getType(r.scalartype(3), r.layout(4), device.type), device.deviceInt64()), r.toBool(6)));
    } else {
      if (!r.isNone(3)) {
        check_out_type_matches(r.tensor(2), r.scalartype(3), r.layout(4),
                               r.device(5), r.isNone(5));
      }
      return wrap(set_requires_grad(dispatch_randn(r.intlist(0), r.generator(1), r.tensor(2)), r.toBool(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_randn_like(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randn_like(Tensor input, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "randn_like(Tensor input, *, bool requires_grad=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto dtype = r.scalartypeWithDefault(1, self.type().scalarType());
    auto layout = r.layoutWithDefault(2, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(3, torch::utils::getDevice(self));
    return wrap(set_requires_grad(dispatch_randn_like(self, torch::getType(dtype, layout, device.type), device.deviceInt64()), r.toBool(4)));
  } else if (r.idx == 1) {
    return wrap(set_requires_grad(dispatch_randn_like(r.tensor(0)), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_randperm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randperm(int64_t n, *, Generator generator=None, Tensor out=None, ScalarType dtype=torch.int64, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      auto device = r.device(5);
      return wrap(set_requires_grad(dispatch_randperm(r.toInt64(0), r.generator(1), torch::getType(r.scalartype(3), r.layout(4), device.type), device.deviceInt64()), r.toBool(6)));
    } else {
      if (!r.isNone(3)) {
        check_out_type_matches(r.tensor(2), r.scalartype(3), r.layout(4),
                               r.device(5), r.isNone(5));
      }
      return wrap(set_requires_grad(dispatch_randperm(r.toInt64(0), r.generator(1), r.tensor(2)), r.toBool(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_range(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "range(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      auto device = r.device(6);
      return wrap(set_requires_grad(dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), torch::getType(r.scalartype(4), r.layout(5), device.type), device.deviceInt64()), r.toBool(7)));
    } else {
      if (!r.isNone(4)) {
        check_out_type_matches(r.tensor(3), r.scalartype(4), r.layout(5),
                               r.device(6), r.isNone(6));
      }
      return wrap(set_requires_grad(dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)), r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reciprocal(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reciprocal(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_reciprocal(r.tensor(0)));
    } else {
      return wrap(dispatch_reciprocal(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_relu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "relu(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_relu(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_relu_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "relu_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_relu_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_remainder(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "remainder(Tensor input, Scalar other, *, Tensor out=None)",
    "remainder(Tensor input, Tensor other, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_remainder(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_remainder(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_remainder(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_remainder(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_renorm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "renorm(Tensor input, Scalar p, int64_t dim, Scalar maxnorm, *, Tensor out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_renorm(r.tensor(0), r.scalar(1), r.toInt64(2), r.scalar(3)));
    } else {
      return wrap(dispatch_renorm(r.tensor(0), r.scalar(1), r.toInt64(2), r.scalar(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reshape(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reshape(Tensor input, IntList shape)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_reshape(r.tensor(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rfft(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rfft(Tensor input, int64_t signal_ndim, bool normalized=False, bool onesided=True)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rfft(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_round(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "round(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_round(r.tensor(0)));
    } else {
      return wrap(dispatch_round(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_round_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "round_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_round_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rrelu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu(Tensor input, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rrelu(r.tensor(0), r.scalar(1), r.scalar(2), r.toBool(3), r.generator(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rrelu_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_(Tensor input, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rrelu_(r.tensor(0), r.scalar(1), r.scalar(2), r.toBool(3), r.generator(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rsqrt(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rsqrt(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_rsqrt(r.tensor(0)));
    } else {
      return wrap(dispatch_rsqrt(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_select(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "select(Tensor input, int64_t dim, int64_t index)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_select(r.tensor(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_selu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "selu(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_selu(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_selu_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "selu_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_selu_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sigmoid(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sigmoid(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_sigmoid(r.tensor(0)));
    } else {
      return wrap(dispatch_sigmoid(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sign(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sign(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_sign(r.tensor(0)));
    } else {
      return wrap(dispatch_sign(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sin(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sin(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_sin(r.tensor(0)));
    } else {
      return wrap(dispatch_sin(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sin_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sin_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sin_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sinh(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sinh(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_sinh(r.tensor(0)));
    } else {
      return wrap(dispatch_sinh(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_slice(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slice(Tensor input, int64_t dim=0, int64_t start=0, int64_t end=9223372036854775807, int64_t step=1)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_slice(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toInt64(3), r.toInt64(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_slogdet(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slogdet(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_slogdet(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_smm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "smm(Tensor input, Tensor mat2)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_smm(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sort(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sort(Tensor input, int64_t dim=-1, bool descending=False, *, TensorList[2] out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_sort(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_sort(r.tensor(0), r.toInt64(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_split(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split(Tensor input, int64_t split_size, int64_t dim=0)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_split(r.tensor(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_split_with_sizes(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split_with_sizes(Tensor input, IntList split_sizes, int64_t dim=0)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_split_with_sizes(r.tensor(0), r.intlist(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sqrt(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sqrt(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_sqrt(r.tensor(0)));
    } else {
      return wrap(dispatch_sqrt(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sqrt_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sqrt_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sqrt_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_squeeze(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "squeeze(Tensor input)",
    "squeeze(Tensor input, int64_t dim)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_squeeze(r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_squeeze(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sspaddmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sspaddmm(Scalar beta, Tensor input, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Scalar beta, Tensor input, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sspaddmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
  } else if (r.idx == 1) {
    return wrap(dispatch_sspaddmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_sspaddmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_sspaddmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_stack(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stack(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_stack(r.tensorlist(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_stack(r.tensorlist(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_std(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "std(Tensor input, bool unbiased=True)",
    "std(Tensor input, int64_t dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_std(r.tensor(0), r.toBool(1)));
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_std(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3)));
    } else {
      return wrap(dispatch_std(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_stft(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stft(Tensor input, int64_t frame_length, int64_t hop, int64_t fft_size=None, bool normalized=False, bool onesided=True, Tensor? window=None, int64_t pad_end=0)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto frame_length = r.toInt64(1);
    auto hop = r.toInt64(2);
    auto fft_size = r.toInt64WithDefault(3, frame_length);
    auto normalized = r.toBool(4);
    auto onesided = r.toBool(5);
    auto window = r.tensor(6);
    auto pad_end = r.toInt64(7);
    return wrap(dispatch_stft(self, frame_length, hop, fft_size, normalized, onesided, window, pad_end));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sub(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sub(Tensor input, Scalar alpha, Tensor other, *, Tensor out=None)|deprecated",
    "sub(Tensor input, Scalar other, *, Scalar alpha=1, Tensor out=None)",
    "sub(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_sub(r.tensor(0), r.scalar(1), r.tensor(2)));
    } else {
      return wrap(dispatch_sub(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_sub(r.tensor(0), r.scalar(1), r.scalar(2)));
    } else {
      return wrap(dispatch_sub(r.tensor(0), r.scalar(1), r.scalar(2), r.tensor(3)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_sub(r.tensor(0), r.tensor(1), r.scalar(2)));
    } else {
      return wrap(dispatch_sub(r.tensor(0), r.tensor(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sum(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sum(Tensor input)",
    "sum(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sum(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_sum(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      return wrap(dispatch_sum(r.tensor(0), r.toInt64(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_svd(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "svd(Tensor input, bool some=True, *, TensorList[3] out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_svd(r.tensor(0), r.toBool(1)));
    } else {
      auto results = r.tensorlist_n<3>(2);
      return wrap(dispatch_svd(r.tensor(0), r.toBool(1), results[0], results[1], results[2]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_symeig(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "symeig(Tensor input, bool eigenvectors=False, bool upper=True, *, TensorList[2] out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_symeig(r.tensor(0), r.toBool(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_symeig(r.tensor(0), r.toBool(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_t(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "t(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_t(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_take(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "take(Tensor input, Tensor index, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_take(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_take(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tan(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tan(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_tan(r.tensor(0)));
    } else {
      return wrap(dispatch_tan(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tanh(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tanh(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_tanh(r.tensor(0)));
    } else {
      return wrap(dispatch_tanh(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_topk(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "topk(Tensor input, int64_t k, int64_t dim=-1, bool largest=True, bool sorted=True, *, TensorList[2] out=None)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_topk(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toBool(3), r.toBool(4)));
    } else {
      auto results = r.tensorlist_n<2>(5);
      return wrap(dispatch_topk(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toBool(3), r.toBool(4), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trace(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trace(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_trace(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_transpose(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "transpose(Tensor input, int64_t dim0, int64_t dim1)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_transpose(r.tensor(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tril(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_tril(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_tril(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_triplet_margin_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, double margin=1.0, double p=2, double eps=1e-06, bool swap=False, bool size_average=True, bool reduce=True)",
  });

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_triplet_margin_loss(r.tensor(0), r.tensor(1), r.tensor(2), r.toDouble(3), r.toDouble(4), r.toDouble(5), r.toBool(6), r.toBool(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_triu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_triu(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_triu(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trtrs(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trtrs(Tensor input, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, TensorList[2] out=None)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_trtrs(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3), r.toBool(4)));
    } else {
      auto results = r.tensorlist_n<2>(5);
      return wrap(dispatch_trtrs(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3), r.toBool(4), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trunc(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trunc(Tensor input, *, Tensor out=None)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_trunc(r.tensor(0)));
    } else {
      return wrap(dispatch_trunc(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trunc_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trunc_(Tensor input)",
  });

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_trunc_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unsqueeze(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsqueeze(Tensor input, int64_t dim)",
  });

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_unsqueeze(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_var(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "var(Tensor input, bool unbiased=True)",
    "var(Tensor input, int64_t dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_var(r.tensor(0), r.toBool(1)));
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_var(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3)));
    } else {
      return wrap(dispatch_var(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_where(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "where(Tensor condition, Tensor input, Tensor other)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_where(r.tensor(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_zeros(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zeros(IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto device = r.device(4);
      return wrap(set_requires_grad(dispatch_zeros(r.intlist(0), torch::getType(r.scalartype(2), r.layout(3), device.type), device.deviceInt64()), r.toBool(5)));
    } else {
      if (!r.isNone(2)) {
        check_out_type_matches(r.tensor(1), r.scalartype(2), r.layout(3),
                               r.device(4), r.isNone(4));
      }
      return wrap(set_requires_grad(dispatch_zeros(r.intlist(0), r.tensor(1)), r.toBool(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_zeros_like(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zeros_like(Tensor input, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "zeros_like(Tensor input, *, bool requires_grad=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto dtype = r.scalartypeWithDefault(1, self.type().scalarType());
    auto layout = r.layoutWithDefault(2, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(3, torch::utils::getDevice(self));
    return wrap(set_requires_grad(dispatch_zeros_like(self, torch::getType(dtype, layout, device.type), device.deviceInt64()), r.toBool(4)));
  } else if (r.idx == 1) {
    return wrap(set_requires_grad(dispatch_zeros_like(r.tensor(0)), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef torch_functions[] = {
  {"clamp", (PyCFunction)THPVariable_clamp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dsmm", (PyCFunction)THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"from_numpy", (PyCFunction)THPVariable_from_numpy, METH_STATIC | METH_O, NULL},
  {"hsmm", (PyCFunction)THPVariable_hspmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_promote_types", (PyCFunction)THPVariable__promote_types, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"saddmm", (PyCFunction)THPVariable_sspaddmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sparse_coo_tensor", (PyCFunction)THPVariable_sparse_coo_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"spmm", (PyCFunction)THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tensor", (PyCFunction)THPVariable_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__and__", (PyCFunction)THPVariable___and__, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__lshift__", (PyCFunction)THPVariable___lshift__, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__or__", (PyCFunction)THPVariable___or__, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__rshift__", (PyCFunction)THPVariable___rshift__, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__xor__", (PyCFunction)THPVariable___xor__, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_abs", (PyCFunction)THPVariable__abs, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_addmv", (PyCFunction)THPVariable__addmv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_addr", (PyCFunction)THPVariable__addr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_argmax", (PyCFunction)THPVariable__argmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_argmin", (PyCFunction)THPVariable__argmin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Half", (PyCFunction)THPVariable__cast_Half, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_double", (PyCFunction)THPVariable__cast_double, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_float", (PyCFunction)THPVariable__cast_float, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_int", (PyCFunction)THPVariable__cast_int, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_int16_t", (PyCFunction)THPVariable__cast_int16_t, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_int64_t", (PyCFunction)THPVariable__cast_int64_t, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_int8_t", (PyCFunction)THPVariable__cast_int8_t, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_uint8_t", (PyCFunction)THPVariable__cast_uint8_t, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cat", (PyCFunction)THPVariable__cat, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_ceil", (PyCFunction)THPVariable__ceil, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_convolution", (PyCFunction)THPVariable__convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_convolution_nogroup", (PyCFunction)THPVariable__convolution_nogroup, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cos", (PyCFunction)THPVariable__cos, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_init_dropout_state", (PyCFunction)THPVariable__cudnn_init_dropout_state, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_rnn", (PyCFunction)THPVariable__cudnn_rnn, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_rnn_flatten_weight", (PyCFunction)THPVariable__cudnn_rnn_flatten_weight, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_dirichlet_grad", (PyCFunction)THPVariable__dirichlet_grad, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_dot", (PyCFunction)THPVariable__dot, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_exp", (PyCFunction)THPVariable__exp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_fft_with_size", (PyCFunction)THPVariable__fft_with_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_floor", (PyCFunction)THPVariable__floor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_ger", (PyCFunction)THPVariable__ger, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_log", (PyCFunction)THPVariable__log, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_mm", (PyCFunction)THPVariable__mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_mv", (PyCFunction)THPVariable__mv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_prod", (PyCFunction)THPVariable__prod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_prodall", (PyCFunction)THPVariable__prodall, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_round", (PyCFunction)THPVariable__round, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_s_where", (PyCFunction)THPVariable__s_where, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sin", (PyCFunction)THPVariable__sin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sqrt", (PyCFunction)THPVariable__sqrt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_standard_gamma", (PyCFunction)THPVariable__standard_gamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_standard_gamma_grad", (PyCFunction)THPVariable__standard_gamma_grad, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sum", (PyCFunction)THPVariable__sum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sumall", (PyCFunction)THPVariable__sumall, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_trunc", (PyCFunction)THPVariable__trunc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_unique", (PyCFunction)THPVariable__unique, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"abs", (PyCFunction)THPVariable_abs, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"abs_", (PyCFunction)THPVariable_abs_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"acos", (PyCFunction)THPVariable_acos, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"adaptive_avg_pool1d", (PyCFunction)THPVariable_adaptive_avg_pool1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"adaptive_max_pool1d", (PyCFunction)THPVariable_adaptive_max_pool1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"add", (PyCFunction)THPVariable_add, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addbmm", (PyCFunction)THPVariable_addbmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addcdiv", (PyCFunction)THPVariable_addcdiv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addcmul", (PyCFunction)THPVariable_addcmul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addmm", (PyCFunction)THPVariable_addmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addmv", (PyCFunction)THPVariable_addmv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addmv_", (PyCFunction)THPVariable_addmv_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addr", (PyCFunction)THPVariable_addr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"allclose", (PyCFunction)THPVariable_allclose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"arange", (PyCFunction)THPVariable_arange, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"as_strided", (PyCFunction)THPVariable_as_strided, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"asin", (PyCFunction)THPVariable_asin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"atan", (PyCFunction)THPVariable_atan, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"atan2", (PyCFunction)THPVariable_atan2, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"baddbmm", (PyCFunction)THPVariable_baddbmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"batch_norm", (PyCFunction)THPVariable_batch_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bernoulli", (PyCFunction)THPVariable_bernoulli, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bernoulli_", (PyCFunction)THPVariable_bernoulli_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bilinear", (PyCFunction)THPVariable_bilinear, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bmm", (PyCFunction)THPVariable_bmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"btrifact", (PyCFunction)THPVariable_btrifact, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"btrifact_with_info", (PyCFunction)THPVariable_btrifact_with_info, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"btrisolve", (PyCFunction)THPVariable_btrisolve, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cat", (PyCFunction)THPVariable_cat, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ceil", (PyCFunction)THPVariable_ceil, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ceil_", (PyCFunction)THPVariable_ceil_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"chunk", (PyCFunction)THPVariable_chunk, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv1d", (PyCFunction)THPVariable_conv1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv2d", (PyCFunction)THPVariable_conv2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv3d", (PyCFunction)THPVariable_conv3d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_tbc", (PyCFunction)THPVariable_conv_tbc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_transpose1d", (PyCFunction)THPVariable_conv_transpose1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_transpose2d", (PyCFunction)THPVariable_conv_transpose2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_transpose3d", (PyCFunction)THPVariable_conv_transpose3d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"convolution", (PyCFunction)THPVariable_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cos", (PyCFunction)THPVariable_cos, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cos_", (PyCFunction)THPVariable_cos_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cosh", (PyCFunction)THPVariable_cosh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cosine_embedding_loss", (PyCFunction)THPVariable_cosine_embedding_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cross", (PyCFunction)THPVariable_cross, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_affine_grid_generator", (PyCFunction)THPVariable_cudnn_affine_grid_generator, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_batch_norm", (PyCFunction)THPVariable_cudnn_batch_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_convolution", (PyCFunction)THPVariable_cudnn_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_convolution_transpose", (PyCFunction)THPVariable_cudnn_convolution_transpose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_grid_sampler", (PyCFunction)THPVariable_cudnn_grid_sampler, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_is_acceptable", (PyCFunction)THPVariable_cudnn_is_acceptable, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cumprod", (PyCFunction)THPVariable_cumprod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cumsum", (PyCFunction)THPVariable_cumsum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"det", (PyCFunction)THPVariable_det, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diag", (PyCFunction)THPVariable_diag, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diagflat", (PyCFunction)THPVariable_diagflat, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diagonal", (PyCFunction)THPVariable_diagonal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"digamma", (PyCFunction)THPVariable_digamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dist", (PyCFunction)THPVariable_dist, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"div", (PyCFunction)THPVariable_div, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dot", (PyCFunction)THPVariable_dot, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"eig", (PyCFunction)THPVariable_eig, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"embedding", (PyCFunction)THPVariable_embedding, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"embedding_bag", (PyCFunction)THPVariable_embedding_bag, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"embedding_renorm_", (PyCFunction)THPVariable_embedding_renorm_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"empty", (PyCFunction)THPVariable_empty, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"empty_like", (PyCFunction)THPVariable_empty_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"eq", (PyCFunction)THPVariable_eq, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"equal", (PyCFunction)THPVariable_equal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erf", (PyCFunction)THPVariable_erf, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erfinv", (PyCFunction)THPVariable_erfinv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"exp", (PyCFunction)THPVariable_exp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"exp_", (PyCFunction)THPVariable_exp_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"expm1", (PyCFunction)THPVariable_expm1, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"eye", (PyCFunction)THPVariable_eye, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fft", (PyCFunction)THPVariable_fft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"floor", (PyCFunction)THPVariable_floor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"floor_", (PyCFunction)THPVariable_floor_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fmod", (PyCFunction)THPVariable_fmod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"frac", (PyCFunction)THPVariable_frac, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"full", (PyCFunction)THPVariable_full, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"full_like", (PyCFunction)THPVariable_full_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gather", (PyCFunction)THPVariable_gather, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ge", (PyCFunction)THPVariable_ge, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gels", (PyCFunction)THPVariable_gels, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"geqrf", (PyCFunction)THPVariable_geqrf, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ger", (PyCFunction)THPVariable_ger, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gesv", (PyCFunction)THPVariable_gesv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"group_norm", (PyCFunction)THPVariable_group_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gt", (PyCFunction)THPVariable_gt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hinge_embedding_loss", (PyCFunction)THPVariable_hinge_embedding_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"histc", (PyCFunction)THPVariable_histc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hspmm", (PyCFunction)THPVariable_hspmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ifft", (PyCFunction)THPVariable_ifft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index", (PyCFunction)THPVariable_index, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_put_", (PyCFunction)THPVariable_index_put_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_select", (PyCFunction)THPVariable_index_select, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"inverse", (PyCFunction)THPVariable_inverse, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"irfft", (PyCFunction)THPVariable_irfft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_distributed", (PyCFunction)THPVariable_is_distributed, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_floating_point", (PyCFunction)THPVariable_is_floating_point, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_nonzero", (PyCFunction)THPVariable_is_nonzero, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_same_size", (PyCFunction)THPVariable_is_same_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_signed", (PyCFunction)THPVariable_is_signed, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"isclose", (PyCFunction)THPVariable_isclose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"kthvalue", (PyCFunction)THPVariable_kthvalue, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"layer_norm", (PyCFunction)THPVariable_layer_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"le", (PyCFunction)THPVariable_le, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lerp", (PyCFunction)THPVariable_lerp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lgamma", (PyCFunction)THPVariable_lgamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"linspace", (PyCFunction)THPVariable_linspace, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log", (PyCFunction)THPVariable_log, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log10", (PyCFunction)THPVariable_log10, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log1p", (PyCFunction)THPVariable_log1p, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log2", (PyCFunction)THPVariable_log2, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log_", (PyCFunction)THPVariable_log_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logdet", (PyCFunction)THPVariable_logdet, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logspace", (PyCFunction)THPVariable_logspace, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lt", (PyCFunction)THPVariable_lt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"margin_ranking_loss", (PyCFunction)THPVariable_margin_ranking_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"masked_select", (PyCFunction)THPVariable_masked_select, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"matmul", (PyCFunction)THPVariable_matmul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"max", (PyCFunction)THPVariable_max, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"max_pool1d", (PyCFunction)THPVariable_max_pool1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mean", (PyCFunction)THPVariable_mean, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"median", (PyCFunction)THPVariable_median, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"min", (PyCFunction)THPVariable_min, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mkldnn_convolution", (PyCFunction)THPVariable_mkldnn_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mkldnn_convolution_backward_weights", (PyCFunction)THPVariable_mkldnn_convolution_backward_weights, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mm", (PyCFunction)THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mode", (PyCFunction)THPVariable_mode, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mul", (PyCFunction)THPVariable_mul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"multinomial", (PyCFunction)THPVariable_multinomial, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mv", (PyCFunction)THPVariable_mv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"narrow", (PyCFunction)THPVariable_narrow, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ne", (PyCFunction)THPVariable_ne, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"neg", (PyCFunction)THPVariable_neg, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"nonzero", (PyCFunction)THPVariable_nonzero, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"norm", (PyCFunction)THPVariable_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"normal", (PyCFunction)THPVariable_normal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"numel", (PyCFunction)THPVariable_numel, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ones", (PyCFunction)THPVariable_ones, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ones_like", (PyCFunction)THPVariable_ones_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"orgqr", (PyCFunction)THPVariable_orgqr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ormqr", (PyCFunction)THPVariable_ormqr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pairwise_distance", (PyCFunction)THPVariable_pairwise_distance, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pin_memory", (PyCFunction)THPVariable_pin_memory, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"poisson", (PyCFunction)THPVariable_poisson, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"polygamma", (PyCFunction)THPVariable_polygamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"potrf", (PyCFunction)THPVariable_potrf, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"potri", (PyCFunction)THPVariable_potri, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"potrs", (PyCFunction)THPVariable_potrs, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pow", (PyCFunction)THPVariable_pow, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"prod", (PyCFunction)THPVariable_prod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pstrf", (PyCFunction)THPVariable_pstrf, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"qr", (PyCFunction)THPVariable_qr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rand", (PyCFunction)THPVariable_rand, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rand_like", (PyCFunction)THPVariable_rand_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randn", (PyCFunction)THPVariable_randn, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randn_like", (PyCFunction)THPVariable_randn_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randperm", (PyCFunction)THPVariable_randperm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"range", (PyCFunction)THPVariable_range, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"reciprocal", (PyCFunction)THPVariable_reciprocal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"relu", (PyCFunction)THPVariable_relu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"relu_", (PyCFunction)THPVariable_relu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"remainder", (PyCFunction)THPVariable_remainder, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"renorm", (PyCFunction)THPVariable_renorm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"reshape", (PyCFunction)THPVariable_reshape, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rfft", (PyCFunction)THPVariable_rfft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"round", (PyCFunction)THPVariable_round, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"round_", (PyCFunction)THPVariable_round_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rrelu", (PyCFunction)THPVariable_rrelu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rrelu_", (PyCFunction)THPVariable_rrelu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rsqrt", (PyCFunction)THPVariable_rsqrt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"select", (PyCFunction)THPVariable_select, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"selu", (PyCFunction)THPVariable_selu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"selu_", (PyCFunction)THPVariable_selu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sigmoid", (PyCFunction)THPVariable_sigmoid, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sign", (PyCFunction)THPVariable_sign, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sin", (PyCFunction)THPVariable_sin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sin_", (PyCFunction)THPVariable_sin_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sinh", (PyCFunction)THPVariable_sinh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"slice", (PyCFunction)THPVariable_slice, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"slogdet", (PyCFunction)THPVariable_slogdet, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"smm", (PyCFunction)THPVariable_smm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sort", (PyCFunction)THPVariable_sort, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"split", (PyCFunction)THPVariable_split, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"split_with_sizes", (PyCFunction)THPVariable_split_with_sizes, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sqrt", (PyCFunction)THPVariable_sqrt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sqrt_", (PyCFunction)THPVariable_sqrt_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"squeeze", (PyCFunction)THPVariable_squeeze, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sspaddmm", (PyCFunction)THPVariable_sspaddmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"stack", (PyCFunction)THPVariable_stack, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"std", (PyCFunction)THPVariable_std, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"stft", (PyCFunction)THPVariable_stft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sub", (PyCFunction)THPVariable_sub, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sum", (PyCFunction)THPVariable_sum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"svd", (PyCFunction)THPVariable_svd, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"symeig", (PyCFunction)THPVariable_symeig, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"t", (PyCFunction)THPVariable_t, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"take", (PyCFunction)THPVariable_take, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tan", (PyCFunction)THPVariable_tan, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tanh", (PyCFunction)THPVariable_tanh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"topk", (PyCFunction)THPVariable_topk, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trace", (PyCFunction)THPVariable_trace, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"transpose", (PyCFunction)THPVariable_transpose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tril", (PyCFunction)THPVariable_tril, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"triplet_margin_loss", (PyCFunction)THPVariable_triplet_margin_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"triu", (PyCFunction)THPVariable_triu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trtrs", (PyCFunction)THPVariable_trtrs, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trunc", (PyCFunction)THPVariable_trunc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trunc_", (PyCFunction)THPVariable_trunc_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"unsqueeze", (PyCFunction)THPVariable_unsqueeze, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"var", (PyCFunction)THPVariable_var, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"where", (PyCFunction)THPVariable_where, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"zeros", (PyCFunction)THPVariable_zeros, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"zeros_like", (PyCFunction)THPVariable_zeros_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {NULL}
};

static PyTypeObject THPVariableFunctions = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._VariableFunctions",         /* tp_name */
  0,                                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  torch_functions,                       /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  0                                      /* tp_new */
};

void initTorchFunctions(PyObject* module) {
  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(module, "_VariableFunctions", (PyObject*)&THPVariableFunctions) < 0) {
    throw python_error();
  }
}

}} // namespace torch::autograd
