#include <Python.h>
#include <ATen/ATen.h>
#include "tensor_layouts.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"

namespace torch { namespace utils {

void initializeLayouts() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) python_error();

  PyObject *strided_layout = THPLayout_New(at::kCPU, "torch.strided");
  Py_INCREF(strided_layout);
  if (PyModule_AddObject(torch_module, "strided", strided_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)strided_layout, at::Backend::CPU);
  registerLayoutObject((THPLayout*)strided_layout, at::Backend::CUDA);

  PyObject *sparse_coo_layout = THPLayout_New(at::kSparseCPU, "torch.sparse_coo");
  Py_INCREF(sparse_coo_layout);
  if (PyModule_AddObject(torch_module, "sparse_coo", sparse_coo_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)sparse_coo_layout, at::Backend::SparseCPU);
  registerLayoutObject((THPLayout*)sparse_coo_layout, at::Backend::SparseCUDA);
}

const at::Type& toLayout(const at::Type& type, const THPLayout& layout) {
  if (layout.is_strided) {
    return type.toBackend(type.is_cuda() ? at::Backend::CUDA : at::Backend::CPU);
  } else {
    return type.toBackend(type.is_cuda() ? at::Backend::SparseCUDA : at::Backend::SparseCPU);
  }
}

}} // namespace torch::utils
