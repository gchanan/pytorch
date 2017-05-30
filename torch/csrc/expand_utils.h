#ifndef THP_EXPAND_UTILS_H
#define THP_EXPAND_UTILS_H

#include <sstream>
#include <Python.h>

template <typename ExpandType>
ExpandType *newForExpand(LIBRARY_STATE_TYPE_NOARGS);

template <typename TensorType>
int expand(LIBRARY_STATE_TYPE TensorType *r, TensorType *tensor, THLongStorage *sizes, int raiseErrors);

template <typename ExpandType, typename TensorType>
int expand_inplace(LIBRARY_STATE_TYPE ExpandType *r, ExpandType *to_expand, TensorType *tensor,
                   char *to_expand_name, char *tensor_name, bool fallback) {
  THLongStoragePtr tensor_size = THLongStorage_newWithSize(tensor->nDimension);
  THLongStorage_rawCopy(tensor_size.get(), tensor->size);
  ptrdiff_t tensor_nElem = THSize_nElement(tensor->nDimension, tensor->size);
  bool skip_expand = false;
  ptrdiff_t to_expand_nElem = THSize_nElement(to_expand->nDimension, to_expand->size);
  int ret = 0;

  bool to_expand_raise = !fallback || (to_expand_nElem != tensor_nElem);
  int to_expand_err = !skip_expand && expand<ExpandType>(LIBRARY_STATE r, to_expand, tensor_size.get(), to_expand_raise);
  if (to_expand_err != 0 && !to_expand_raise) {
    skip_expand = true; // don't do further expansions
    ret = -1;
    std::ostringstream warn;
    warn << to_expand_name << " is not broadcastable to " << tensor_name
         << ", but they have the same number of elements.  Falling back to deprecated pointwise behavior.";
    PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
  }

  if (fallback && getBackCompatBroadcastWarn()) {
    bool same_shape = THSize_isSameSizeAs(tensor->size, tensor->nDimension,
        to_expand->size, to_expand->nDimension);
    if (!same_shape && to_expand_err == 0 && (tensor_nElem == to_expand_nElem) && fallback) {
      std::ostringstream warn;
      warn << tensor_name << " and " << to_expand_name << " do not have the same shape, but are "
           << "broadcastable, and have the same number of elements.  Changing behavior in a backwards incompatible "
           << "manner to broadcasting rather than viewing as 1-dimensional.";
      PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
    }
  }

  return ret;
}

template <typename TensorType>
int expand_inplace2(LIBRARY_STATE_TYPE TensorType *r1, TensorType *r2,
                    TensorType *to_expand1, TensorType *to_expand2, TensorType *tensor,
                    char *to_expand1_name, char *to_expand2_name, char *tensor_name, bool fallback) {
  THLongStoragePtr tensor_size = THLongStorage_newWithSize(tensor->nDimension);
  THLongStorage_rawCopy(tensor_size.get(), tensor->size);
  ptrdiff_t tensor_nElem = THSize_nElement(tensor->nDimension, tensor->size);
  bool skip_expand = false;
  ptrdiff_t to_expand1_nElem = THSize_nElement(to_expand1->nDimension, to_expand1->size);
  ptrdiff_t to_expand2_nElem = THSize_nElement(to_expand2->nDimension, to_expand2->size);
  bool to_expand1_raise = !fallback || (tensor_nElem != to_expand1_nElem);
  bool to_expand2_raise = !fallback || (tensor_nElem != to_expand2_nElem);
  int ret = 0;

  int to_expand1_err = !skip_expand && expand<TensorType>(LIBRARY_STATE r1, to_expand1, tensor_size.get(),
                                                          to_expand1_raise || to_expand2_raise);
  if (to_expand1_err != 0 && !(to_expand1_raise || to_expand2_raise)) {
    skip_expand = true; // don't do further expansions
    ret = -1;
    std::ostringstream warn;
    warn << to_expand1_name << " is not broadcastable to " << tensor_name
         << ", but they have the same number of elements.  Falling back to deprecated pointwise behavior.";
    PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
  }

  int to_expand2_err = !skip_expand && expand<TensorType>(LIBRARY_STATE r2, to_expand2, tensor_size.get(),
                                                          to_expand1_raise || to_expand2_raise);
  if (to_expand2_err != 0 && !(to_expand1_raise || to_expand2_raise)) {
    skip_expand = true; // don't do further expansions
    ret = -1;
    std::ostringstream warn;
    warn << to_expand2_name << " is not broadcastable to " << tensor_name
         << ", but they have the same number of elements.  Falling back to deprecated pointwise behavior.";
    PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
  }

  if (fallback && getBackCompatBroadcastWarn()) {
    bool same_shape = THSize_isSameSizeAs(tensor->size, tensor->nDimension,
        to_expand1->size, to_expand1->nDimension);
    if (!same_shape && to_expand1_err == 0 && (tensor_nElem == to_expand1_nElem) && fallback) {
      std::ostringstream warn;
      warn << tensor_name << " and " << to_expand1_name << " do not have the same shape, but are "
           << "broadcastable, and have the same number of elements.  Changing behavior in a backwards incompatible "
           << "manner to broadcasting rather than viewing as 1-dimensional.";
      PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
    }
  }
  if (fallback && getBackCompatBroadcastWarn()) {
    bool same_shape = THSize_isSameSizeAs(tensor->size, tensor->nDimension,
        to_expand2->size, to_expand2->nDimension);
    if (!same_shape && to_expand1_err == 0 && (tensor_nElem == to_expand2_nElem) && fallback) {
      std::ostringstream warn;
      warn << tensor_name << " and " << to_expand2_name << " do not have the same shape, but are "
           << "broadcastable, and have the same number of elements.  Changing behavior in a backwards incompatible "
           << "manner to broadcasting rather than viewing as 1-dimensional.";
      PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
    }
  }

  return ret;
}

#endif
