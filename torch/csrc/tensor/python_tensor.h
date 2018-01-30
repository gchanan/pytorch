#pragma once

#include <Python.h>
#include <ATen/ATen.h>

namespace torch { namespace tensor {

// Initializes the Python tensor type objects: torch.Tensor, torch.FloatTensor,
// etc.
void initialize_types(PyObject* module);

// Sets the default tensor type
void set_default_tensor_type(PyObject* type_obj);

// Gets the ATen type object for the default tensor type. Note that the
// returned value will be a VariableType instance.
at::Type& get_default_tensor_type();

}} // namespace torch::tensor
