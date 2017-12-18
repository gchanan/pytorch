#pragma once

#include <utility>

namespace at {

template<template <typename> class F, typename ... Args>
void dispatch_all(const Type& the_type, const char *name, Args&&... args) {

    switch(the_type.scalarType()) {
        case ScalarType::Byte:
          F<uint8_t>::apply(std::forward<Args>(args)...);
        case ScalarType::Char:
          F<int8_t>::apply(std::forward<Args>(args)...);
        case ScalarType::Double:
          F<double>::apply(std::forward<Args>(args)...);
        case ScalarType::Float:
          F<float>::apply(std::forward<Args>(args)...);
        case ScalarType::Int:
          F<int>::apply(std::forward<Args>(args)...);
        case ScalarType::Long:
          F<int64_t>::apply(std::forward<Args>(args)...);
        case ScalarType::Short:
          F<int16_t>::apply(std::forward<Args>(args)...);
        case ScalarType::Half:
          F<Half>::apply(std::forward<Args>(args)...);
        default:
            runtime_error("%s not implemented for '%s'", name, the_type.toString());
    }
}
template<template <typename> class F, typename ... Args>
void dispatch_floating_types(const Type& the_type, const char *name, Args&&... args) {
  switch(the_type.scalarType()) {
    case ScalarType::Double:
      F<double>::apply(std::forward<Args>(args)...);
    case ScalarType::Float:
      F<float>::apply(std::forward<Args>(args)...);
    case ScalarType::Half: // no native half math on either CPU or GPU.
    default:
      runtime_error("%s not implemented for '%s'", name, the_type.toString());
  }
}


}
