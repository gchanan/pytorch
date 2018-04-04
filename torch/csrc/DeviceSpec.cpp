#include "DeviceSpec.h"

#include <cstring>
#include <structmember.h>
#include <sstream>
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_strings.h"

PyObject *THPDeviceSpec_New(THPDeviceType device_type, int64_t device_index, bool is_default)
{
  auto type = (PyTypeObject*)&THPDeviceSpecType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPDeviceSpec*>(self.get());
  self_->device_type = device_type;
  self_->device_index = device_index;
  self_->is_default = is_default;
  return self.release();
}

static inline std::string deviceTypeString(THPDeviceType device_type) {
  switch (device_type) {
    case THPDeviceType::CUDA:
      return "cuda";
    case THPDeviceType::CPU:
      return "cpu";
    default:
      throw std::runtime_error("unexpected device type");
  }
}

PyObject *THPDeviceSpec_repr(THPDeviceSpec *self)
{
  std::ostringstream oss;
  if (!self->is_default) {
    oss << deviceTypeString(self->device_type) << ":" << self->device_index;
  } else {
    oss << deviceTypeString(self->device_type);
  }
  return THPUtils_packString(oss.str().c_str());
}

PyObject *THPDeviceSpec_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  /*static torch::PythonArgParser parser({
    "DeviceSpec(String str)",
    "DeviceSpec(String device_type, int64_t device_index)"
    "DeviceSpec(int64_t device_index)"
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto str = r.string(0);
    std::string cpu_prefix("cpu:");
    std::string cuda_prefix("cuda:");
    if (str == "cpu" || str == "cuda") {
      return THPDeviceSpec_New(str, -1, true);
    } else if (str.compare(0, cpu_prefix.length(), cpu_prefix) == 0) {
      auto device_index = std::stoi(str.substr(cuda_prefix.length()));
      if (device_index < 0) {
        throw std::runtime_error("device index must be positive");
      }
      return THPDeviceSpec_New("cuda", device_index, false);
    } else if (str.compare(0, cuda_prefix.length(), cuda_prefix) == 0) {
      auto device_index = std::stoi(str.substr(cuda_prefix.length()));
      if (device_index < 0) {
        throw std::runtime_error("device index must be positive");
      }
      return THPDeviceSpec_New("cuda", device_index, false);
    }
    throw torch::TypeError("only \"cuda\" and \"cpu\" are valid device types, got %s", str.c_str());
  } else if (r.idx == 1) {
    auto device_type = r.string(0);
    if (device_type != "cuda" && device_type != "cpu") {
      throw torch::TypeError("only \"cuda\" and \"cpu\" are valid device types, got %s", device_type.c_str());
    }
    auto device_index = r.toInt64(1);
    if (device_index < 0) {
      throw std::runtime_error("device index must be positive");
    }
    return THPDeviceSpec_New(device_type, device_index, false);
  } else if (r.idx == 2) {
    auto device_index = r.toInt64(0);
    return THPDeviceSpec_New("cuda", device_index, device_index == -1);
  }*/
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPDeviceSpec_device_type(THPDeviceSpec *self)
{
  HANDLE_TH_ERRORS
  return THPUtils_packString(deviceTypeString(self->device_type).c_str());
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPDeviceSpec_device_index(THPDeviceSpec *self)
{
  HANDLE_TH_ERRORS
  if (self->is_default) {
    Py_RETURN_NONE;
  } else {
    return THPUtils_packInt64(self->device_index);
  }
  END_HANDLE_TH_ERRORS
}

PyObject *THPDeviceSpec_cuda_device_index(THPDeviceSpec *self)
{
  HANDLE_TH_ERRORS
  if (self->device_type == THPDeviceType::CUDA) {
    return THPUtils_packInt64(self->device_index);
  }
  std::ostringstream oss;
  oss << "cuda_device_index only supported on cuda device, got: " << deviceTypeString(self->device_type);
  throw std::runtime_error(oss.str());
  END_HANDLE_TH_ERRORS
}


typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef THPDeviceSpec_properties[] = {
  {"device_type",       (getter)THPDeviceSpec_device_type, nullptr, nullptr, nullptr},
  {"device_index",      (getter)THPDeviceSpec_device_index, nullptr, nullptr, nullptr},
  {"cuda_device_index", (getter)THPDeviceSpec_cuda_device_index, nullptr, nullptr, nullptr},
  {nullptr}
};

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
  THPDeviceSpec_properties,              /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPDeviceSpec_pynew,                   /* tp_new */
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
