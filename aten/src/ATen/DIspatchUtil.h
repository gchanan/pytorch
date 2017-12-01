#pragma once

#include <utility>

namespace at {

template<template <typename> class F, typename ... Args>
auto dispatch_cpu_floating_point2(const Type& the_type, Args&&... args)
  -> decltype(F<double>::apply(std::forward<Args>(args)...)) {
//void dispatch_cpu_floating_point2(const Type & the_type, Args&&... args) {
    // should type_id work with Variables when we also want to dispatch on CUDA?
    if (the_type.backend() != Backend::CPU) {
      runtime_error("dispatch() not implemented for '%s'", the_type.toString());
    }
    switch(the_type.scalarType()) {
        case ScalarType::Float:
            return F<float>::apply(std::forward<Args>(args)...);
        case ScalarType::Double:
            return F<double>::apply(std::forward<Args>(args)...);
        //case TypeID::CPUHalf:
        //    return F<Half>::apply(std::forward<Args>(args)...);
        default:
            runtime_error("dispatch() not implemented for '%s'", the_type.toString());
    }
    //return F<double>::apply(std::forward<Args>(args)...);
}

template<template <typename> class F>
void dispatch_cpu_floating_point3(int y) {
    F<double>()(y);
}

}
