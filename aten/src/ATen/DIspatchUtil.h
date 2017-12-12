#pragma once

namespace at {

template<template <typename> typename F, typename ... Args>
auto dispatch_cpu_floating_point(const Type & the_type, Args&&... args)
    -> decltype(F<double>(the_type,std::forward<Args>(args)...)) {
    switch(the_type.ID()) {
        case TypeID::CPUDouble:
            return F<double>(the_type,std::forward<Args>(args)...);
        case TypeID::CPUFloat:
            return F<float>(the_type,std::forward<Args>(args)...);
        case TypeID::CPUHalf:
            return F<Half>(the_type,std::forward<Args>(args)...);
        default:
            runtime_error("dispatch() not implemented for '%s'",the_type.toString());
    }
}

}
