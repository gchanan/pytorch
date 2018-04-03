#pragma once

#include <Python.h>
#include <string>

const int DEVICE_TYPE_LEN = 64;

struct THPDeviceSpec {
  PyObject_HEAD
  char device_type[DEVICE_TYPE_LEN + 1];
  int64_t device_index;
};

extern PyTypeObject THPDeviceSpecType;

inline bool THPDeviceSpec_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPDeviceSpecType;
}

PyObject * THPDeviceSpec_New(const std::string& device_type, int64_t device_index);

void THPDeviceSpec_init(PyObject *module);
