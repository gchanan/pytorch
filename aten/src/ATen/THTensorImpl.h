#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "ATen/Retainable.h"
#include "ATen/ScalarType.h"
#include <ATen/SmallVector.h>
//#include "ATen/C10TensorImpl.h"

namespace at {
class Scalar;
struct Type;
struct Storage;
struct Tensor;
} // namespace at

typedef at::SmallVector<int64_t, 6> DimVector;

namespace at {
struct THTensorImpl {
  THTensorImpl() {}
  explicit THTensorImpl(Type * type)
  : is_scalar(false), type_(type) {}

  Type & type() const {
    return *type_;
  }
  IntList sizes() { return sizes_; }
  IntList strides() { return strides_; }
  int64_t dim() { return nDimension_; }
  //virtual void * unsafeGetTH(bool retain) = 0;
  std::unique_ptr<Storage> storage() { return nullptr;} 
  friend struct Type;

  // 0-dim patchup of TH requires us to have a flag marking
  // if a Tensor should be treated as 0-dim.
  // the generated wrapper manipulates this flag.
  // the setter should never be exposed in Tensor's public API
  // because eventually we would like isScalar() to just be dim() == 0;
  bool isScalar() const {
    return is_scalar;
  }
  // this is called by the generated wrapper code when there are conditions
  // when this output tensor should be a scalar. e.g. when all inputs
  // to a function 'add' were scalars, then condition_when_scalar == true.
  // we also prevent this from getting marked as a scalar if it is not
  // the right shape afterall.
  THTensorImpl* maybeScalar(bool condition_when_scalar) {
    is_scalar = false; //force dim() to tell the truth for TH
    is_scalar = condition_when_scalar && dim() == 1 && sizes()[0] == 1;
    return this;
  }
  void setScalar(bool s) {
    is_scalar = s;
  }

  // ~~ Legacy API ~~~~
  // TH has a different view of sizes/strides/dim from ATen, but we want to use
  // the same representation ASAP for the operator library.  Therefore, for now, TH
  // will use these legacy functions, but we will do another path to remove these calls.
  //IntList th_sizes() {
  //  return is_scalar ? : sizes_;  // ARE THESE WRITEABLE?
  //}
  //IntList th_strides() {
  //  return strides;
  //}
  //void th_set_sizes()
  // void th_set_stride()
  //int64_t th_dim() {
  //  return dim;  
  //}
  //int64_t th_set_dim() {
  //
  //}

protected:
  bool is_scalar;
  Type * type_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t storage_offset_;
  Storage* storage_;
  int64_t nDimension_;

};
}