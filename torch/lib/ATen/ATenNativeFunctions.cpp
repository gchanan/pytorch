/*#include "ATenNativeFunctions.h"

namespace at {
namespace native {
static inline std::vector<Tensor> split(const Tensor &self, int64_t split_size, int64_t dim) {
  //afglaigjflagiljalgfa[5] = 55;
  int64_t dim_size = self.size(dim);
  int64_t num_splits = (dim_size + split_size - 1) / split_size;
  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits -1 ? split_size : last_split_size;
    splits[i] = self.narrow(dim, i * split_size, length);
  }
  return splits;
}

}
}
*/