template <>
THFloatTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THFloatTensor_new();
}

template <>
THDoubleTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THDoubleTensor_new();
}

template <>
THHalfTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THHalfTensor_new();
}

template <>
THByteTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THByteTensor_new();
}

template <>
THCharTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THCharTensor_new();
}

template <>
THShortTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THShortTensor_new();
}

template <>
THIntTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THIntTensor_new();
}

template <>
THLongTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THLongTensor_new();
}

template<>
int expand(LIBRARY_STATE_TYPE THFloatTensor *r, THFloatTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THFloatTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THDoubleTensor *r, THDoubleTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THDoubleTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THHalfTensor *r, THHalfTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THHalfTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THByteTensor *r, THByteTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THByteTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THCharTensor *r, THCharTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCharTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THShortTensor *r, THShortTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THShortTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THIntTensor *r, THIntTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THIntTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THLongTensor *r, THLongTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THLongTensor_expand(r, tensor, sizes, raiseErrors);
}
