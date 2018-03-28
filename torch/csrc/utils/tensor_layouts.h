#pragma once

#include <ATen/ATen.h>
#include "torch/csrc/Layout.h"

namespace torch { namespace utils {

void initializeLayouts();
const at::Type& toLayout(const at::Type& type, const THPLayout& layout);

}} // namespace torch::utils
