#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.hpp"
#else

typedef struct THStorage
{
    at::ScalarType scalar_type;
    void *data_ptr;
    ptrdiff_t size;
    std::atomic<int> refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THStorage *view;
    template <typename T>

    inline T * data() const {
      if (scalar_type != at::CTypeToScalarType<th::from_type<real>>::to()) {
        AT_ERROR("Attempt to access Storage having data type ", at::toString(scalar_type),
                 " as pointer of type ", typeid(T).name());
      }
      return unsafeData<T>();
    }

    template <typename T>
    inline T * data() {
      static_cast<const struct THStorage *>(this)->data<T>();
    }

    template <typename T>
    inline T * unsafeData() const {
      return static_cast<T*>(this->data_ptr);
    }

    template <typename T>
    inline T * unsafeData() {
      return static_cast<T*>(this->data_ptr);
    }
} THStorage;

#endif
