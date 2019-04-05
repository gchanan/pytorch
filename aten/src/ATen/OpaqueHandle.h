#pragma once

#include <c10/util/intrusive_ptr.h>

namespace at {

/**
 * `OpaqueHandle` wraps a custom storage handle  of a tensor (as template param) and inherits
 * `c10::intrusive_ptr_target` so that it can be used in `OpaqueTensorImpl::opaque_handle_`.
 *
 * It currently only supports wrapping the custom handle by:
 * - Constructing with an existing custom handle by copy/move constructor.
 *
 * See `OpaqueTensorImpl::opaque_handle_`.
 */
template <typename T>
struct CAFFE2_API OpaqueHandle : c10::intrusive_ptr_target {
private:
  T handle_;

public:
  OpaqueHandle() = delete;
  OpaqueHandle(const T& handle): handle_(handle) {}
  OpaqueHandle(T&& handle): handle_(std::move(handle)) {}

  T& get_handle() {
    return handle_;
  }
};

}
