#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.hpp"
#else

typedef struct THCStorage
{
    at::ScalarType scalar_type;
    void *data_ptr;
    ptrdiff_t size;
    std::atomic<int> refcount;
    char flag;
    THCDeviceAllocator *allocator;
    void *allocatorContext;
    struct THCStorage *view;
    int device;

    template <typename T>
    inline T * data() const {
      if (scalar_type != at::CTypeToScalarType<at::cuda::from_type<T>>::to()) {
        AT_ERROR("Attempt to access Storage having data type ", at::toString(scalar_type),
                 " as pointer of type ", typeid(T).name());
      }
      return unsafeData<T>();
    }

    template <typename T>
    inline T * unsafeData() const {
      return static_cast<T*>(this->data_ptr);
    }
} THCStorage;

#endif
