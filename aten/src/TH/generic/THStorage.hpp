#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.hpp"
#else

typedef struct THStorage
{
    void *data_ptr;
    ptrdiff_t size;
    std::atomic<int> refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THStorage *view;

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
} THStorage;

#endif
