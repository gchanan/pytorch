#include <Python.h>
#include <ATen/ATen.h>
#include "tensor_layouts.h"
//#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Layout.h"

//#include "torch/csrc/autograd/generated/VariableType.h"
//#include "torch/csrc/utils/tensor_types.h"

namespace torch { namespace utils {

void initializeLayouts() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) python_error();

  PyObject *strided_layout = THPLayout_New(at::kCPU, "torch.strided");
  Py_INCREF(strided_layout);
  if (PyModule_AddObject(torch_module, "dense", strided_layout) != 0) {
    throw python_error();
  }

  PyObject *sparse_layout = THPLayout_New(at::kSparseCPU, "torch.sparse_coo");
  Py_INCREF(sparse_layout);
  if (PyModule_AddObject(torch_module, "sparse_coo", sparse_layout) != 0) {
    throw python_error();
  }
}

}} // namespace torch::utils
