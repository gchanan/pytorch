#pragma once

// Provides conversions between Python tensor objects and at::Tensor.

#include <Python.h>
#include <memory>
#include <unordered_map>
#include <ATen/ATen.h>
#include "torch/csrc/Dtype.h"
#include "torch/csrc/Layout.h"


namespace torch {

// Register a PyTypeObject* with the given attributes
void registerStoragePyTypeObject(
    PyTypeObject *pytype, const std::string& name,
    bool is_cuda, bool is_sparse);

void registerDtypeObject(THPDtype *dtype, at::Backend backend, at::ScalarType scalarType, const at::Type* type);
void registerLayoutObject(THPLayout *layout, at::Backend backend);

PyObject* createPyObject(const at::Storage& storage);
THPDtype* getDtype(bool is_cuda, at::ScalarType scalarType);
at::Type& getType(const THPDtype &dtype, const THPLayout& layout);
THPLayout* getLayout(at::Backend backend);
std::unique_ptr<at::Storage> createStorage(PyObject* obj);

bool isStorage(PyObject* obj);

}  // namespace torch
