import numpy as np
import torch

sample_size = 10
feature_dim = 1

weight = np.array([1]* feature_dim)
inputs = np.random.normal(loc=0, size=(sample_size,feature_dim))
label = np.matmul(full_input, weight)

inputs = torch.from_numpy(inputs.astype(np.float32))
labels = torch.from_numpy(label.astype(np.float32))

linear = torch.nn.Linear(1, 1, bias=False)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.5)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = linear(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
