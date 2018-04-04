#pragma once

#include <Python.h>
#include <string>

enum class THPDeviceType {CPU=0, CUDA=1};

struct THPDeviceSpec {
  PyObject_HEAD
  THPDeviceType device_type;
  int64_t device_index;
  bool is_default;
};

extern PyTypeObject THPDeviceSpecType;

inline bool THPDeviceSpec_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPDeviceSpecType;
}

PyObject * THPDeviceSpec_New(THPDeviceType device_type, int64_t device_index, bool is_default);

void THPDeviceSpec_init(PyObject *module);
