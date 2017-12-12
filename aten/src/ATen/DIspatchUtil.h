#pragma once

#include <utility>

namespace at {

template<template <typename> class F, typename ... Args>
auto dispatch_cpu_floating_point2(const Type& the_type, Args&&... args)
  -> decltype(F<double>::apply(std::forward<Args>(args)...)) {
//void dispatch_cpu_floating_point2(const Type & the_type, Args&&... args) {
    switch(the_type.ID()) {
        case TypeID::CPUDouble:
            return F<double>::apply(std::forward<Args>(args)...);
        case TypeID::CPUFloat:
            return F<float>::apply(std::forward<Args>(args)...);
        //case TypeID::CPUHalf:
        //    return F<Half>::apply(std::forward<Args>(args)...);
        default:
            runtime_error("dispatch() not implemented for '%s'",the_type.toString());
    }
    //return F<double>::apply(std::forward<Args>(args)...);
}

template<template <typename> class F>
void dispatch_cpu_floating_point3(int y) {
    F<double>()(y);
}

}
