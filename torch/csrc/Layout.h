#pragma once

#include <Python.h>
#include "ATen/ATen.h"

const int LAYOUT_NAME_LEN = 64;

struct THPLayout {
  PyObject_HEAD
  char name[LAYOUT_NAME_LEN + 1];
};

extern PyTypeObject THPLayoutType;

inline bool THPLayout_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPLayoutType;
}

PyObject * THPLayout_New(at::Backend cdata, const std::string& name);

bool THPLayout_init(PyObject *module);
