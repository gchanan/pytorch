#include <climits>

#include "THStorageFunctions.hpp"

#include "generic/THStorage.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THStorage.cpp"
#include "THGenerateHalfType.h"

#include "generic/THStorageCopy.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THStorageCopy.cpp"
#include "THGenerateHalfType.h"

THStorage* THStorage_new(at::ScalarType scalar_type) {
  THStorage* storage = new THStorage(
      scalar_type,
      0,
      getTHDefaultAllocator(),
      true);
  return storage;
}

// Free a non-weak pointer to THStorage
void THStorage_free(THStorage* storage) {
  if (!storage) {
    return;
  }
  storage->release();
}

// Manually retains a weak reference
void THStorage_weakRetain(THStorage *weak_storage) {
  weak_storage->weak_retain();
}

// Releases a weak reference
void THStorage_weakFree(THStorage *weak_storage) {
  weak_storage->weak_release();
}

// Given a weak reference, returns a strong reference to a storage (which must
// be freed when done) or null if the storage is already dead.
THStorage* THStorage_weakLock(THStorage *weak_storage) {
  if (weak_storage->weak_lock())
    return weak_storage;
  return nullptr;
}

THDescBuff THLongStorage_sizeDesc(const THLongStorage *size) {
  return _THSizeDesc(THLongStorage_data(size), size->size);
}

ptrdiff_t THStorage_size(const THStorage *self)
{
  return self->size;
}

void THStorage_retain(THStorage *storage)
{
  if (storage) {
    storage->retain();
  }
}

/*
// I don't think you should ever call this
THStorage* THStorage_newWithData(at::ScalarType scalar_type, std::unique_ptr<at::BoundDeleter> data, ptrdiff_t size)
{
  return THStorage_newWithDataAndAllocator(scalar_type, data, size,
                                           getTHDefaultAllocator());
}
*/

void THStorage_resize(THStorage *storage, ptrdiff_t size)
{
  if (storage->resizable)
  {
    /* case when the allocator does not have a realloc defined */
    at::DataPtr old_data;
    std::swap(old_data, storage->data_ptr);
    ptrdiff_t old_size = storage->size;
    if (size != 0) {
      storage->data_ptr = storage->allocator->allocate(at::elementSize(storage->scalar_type)*size);
    }
    storage->size = size;
    if (old_data != nullptr) {
      ptrdiff_t copy_size = old_size;
      if (storage->size < copy_size) {
        copy_size = storage->size;
      }
      if (copy_size > 0) {
        memcpy(storage->data_ptr.get(), old_data.get(), at::elementSize(storage->scalar_type)*copy_size);
      }
    }
  } else {
    THError("Trying to resize storage that is not resizable");
  }
}

void THStorage_swap(THStorage *storage1, THStorage *storage2)
{
#define SWAP(val) { std::swap(storage1->val, storage2->val); }
    SWAP(scalar_type);
    SWAP(data_ptr);
    SWAP(size);
    // don't swap refcount!
    SWAP(resizable);
    SWAP(allocator);
    SWAP(finalizer);
#undef SWAP
}
