#pragma once

#include <Python.h>

namespace torch { namespace utils {

void initializeDtypes(PyObject *module);

}} // namespace torch::utils
