#include "DeviceSpec.h"

#include <cstring>
#include <structmember.h>
#include <sstream>
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_strings.h"

PyObject *THPDeviceSpec_New(const std::string& device_type, int64_t device_index)
{
  auto type = (PyTypeObject*)&THPDeviceSpecType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPDeviceSpec*>(self.get());
  std::strncpy (self_->device_type, device_type.c_str(), DEVICE_TYPE_LEN);
  self_->device_type[DEVICE_TYPE_LEN] = '\0';
  self_->device_index = device_index;
  return self.release();
}

PyObject *THPDeviceSpec_repr(THPDeviceSpec *self)
{
  std::ostringstream oss;
  oss << self->device_type << ":" << self->device_index;
  return THPUtils_packString(oss.str().c_str());
}

int THPDeviceSpec_pyinit(PyObject *self, PyObject *args, PyObject *kwds)
{
  return 0;
}

PyTypeObject THPDeviceSpecType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.DeviceSpec",                    /* tp_name */
  sizeof(THPDeviceSpec),                 /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  (reprfunc)THPDeviceSpec_repr,          /* tp_repr */
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
  nullptr,                               /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  THPDeviceSpec_pyinit,                  /* tp_init */
  0,                                     /* tp_alloc */
  0,                                     /* tp_new */
};

void THPDeviceSpec_init(PyObject *module)
{
  if (PyType_Ready(&THPDeviceSpecType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPDeviceSpecType);
  if (PyModule_AddObject(module, "DeviceSpec", (PyObject *)&THPDeviceSpecType) != 0) {
    throw python_error();
  }
}
