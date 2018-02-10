#ifndef THP_DTYPE_INC
#define THP_DTYPE_INC

#include <Python.h>
#include "stdint.h"
#include "ATen/ATen.h"

struct THPDtype {
  PyObject_HEAD
  at::Type *cdata;
};

extern PyObject *THPDtypeClass;

#define THPDtype_Check(obj) ((PyObject*)Py_TYPE(obj) == THPDtypeClass)

PyObject * THPDtype_NewWithType(at::Type* type);

#ifdef _THP_CORE
bool THPDtype_init(PyObject *module);
#endif

#endif
