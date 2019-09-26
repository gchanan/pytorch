sample_size=10
feature_dim=1

import torch

weight = torch.tensor([1] * feature_dim)
inputs = torch.randn((sample_size, feature_dim))
label = torch.matmul(inputs, weight)

print(inputs, labels)
