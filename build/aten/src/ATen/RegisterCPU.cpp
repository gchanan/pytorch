#include <ATen/RegisterCPU.h>

// @generated by aten/src/ATen/gen.py

#include <ATen/Type.h>
#include <ATen/Context.h>
#include <ATen/UndefinedType.h>
#include <ATen/core/VariableHooksInterface.h>

#include "ATen/CPUByteType.h"
#include "ATen/CPUCharType.h"
#include "ATen/CPUDoubleType.h"
#include "ATen/CPUFloatType.h"
#include "ATen/CPUIntType.h"
#include "ATen/CPULongType.h"
#include "ATen/CPUShortType.h"
#include "ATen/CPUHalfType.h"
#include "ATen/SparseCPUByteType.h"
#include "ATen/SparseCPUCharType.h"
#include "ATen/SparseCPUDoubleType.h"
#include "ATen/SparseCPUFloatType.h"
#include "ATen/SparseCPUIntType.h"
#include "ATen/SparseCPULongType.h"
#include "ATen/SparseCPUShortType.h"

namespace at {

void register_cpu_types(Context * context) {
  context->registerType(Backend::CPU, ScalarType::Byte, new CPUByteType());
  context->registerType(Backend::CPU, ScalarType::Char, new CPUCharType());
  context->registerType(Backend::CPU, ScalarType::Double, new CPUDoubleType());
  context->registerType(Backend::CPU, ScalarType::Float, new CPUFloatType());
  context->registerType(Backend::CPU, ScalarType::Int, new CPUIntType());
  context->registerType(Backend::CPU, ScalarType::Long, new CPULongType());
  context->registerType(Backend::CPU, ScalarType::Short, new CPUShortType());
  context->registerType(Backend::CPU, ScalarType::Half, new CPUHalfType());
  context->registerType(Backend::SparseCPU, ScalarType::Byte, new SparseCPUByteType());
  context->registerType(Backend::SparseCPU, ScalarType::Char, new SparseCPUCharType());
  context->registerType(Backend::SparseCPU, ScalarType::Double, new SparseCPUDoubleType());
  context->registerType(Backend::SparseCPU, ScalarType::Float, new SparseCPUFloatType());
  context->registerType(Backend::SparseCPU, ScalarType::Int, new SparseCPUIntType());
  context->registerType(Backend::SparseCPU, ScalarType::Long, new SparseCPULongType());
  context->registerType(Backend::SparseCPU, ScalarType::Short, new SparseCPUShortType());
  context->registerType(Backend::Undefined, ScalarType::Undefined, new UndefinedType());
}

} // namespace at
