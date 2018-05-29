#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.hpp"
#else

typedef struct THCStorage
{
    void *data_ptr;
    ptrdiff_t size;
    std::atomic<int> refcount;
    char flag;
    THCDeviceAllocator *allocator;
    void *allocatorContext;
    struct THCStorage *view;
    int device;

    template <typename T>
    T * data() const {
      return unsafeData<T>();
    }

    template <typename T>
    T * data() {
      return unsafeData<T>();
    }

    template <typename T>
    T * unsafeData() const {
      return static_cast<T*>(this->data_ptr);
    }

    template <typename T>
    T * unsafeData() {
      return static_cast<T*>(this->data_ptr);
    }
} THCStorage;

#endif
