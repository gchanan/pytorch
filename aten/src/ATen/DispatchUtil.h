#pragma once

#include <utility>

namespace at {

template<template <typename> class F, typename ... Args>
auto dispatch_cpu_floating_point(const Type& the_type, const char * name, Args&&... args)
  -> decltype(F<double>::apply(std::forward<Args>(args)...)) {
    // should type_id work with Variables when we also want to dispatch on CUDA?
    if (the_type.backend() != Backend::CPU) {
      runtime_error("%s not implemented for '%s'", name, the_type.toString());
    }
    switch(the_type.scalarType()) {
        case ScalarType::Float:
            return F<float>::apply(std::forward<Args>(args)...);
        case ScalarType::Double:
            return F<double>::apply(std::forward<Args>(args)...);
        default:
            runtime_error("%s not implemented for '%s'", name, the_type.toString());
    }
}

}
