import numpy as np
import torch

sample_size = 10
feature_dim = 1

weight = np.array([1]* feature_dim)
full_input = np.random.normal(loc=0, size=(sample_size,feature_dim))
label = np.matmul(full_input, weight)
full_input = full_input.astype(np.float32)
label = label.astype(np.float32)

inputs = torch.from_numpy(full_input)
labels = torch.from_numpy(label)

linear = torch.nn.Linear(1, 1, bias=False)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.5)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = linear(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
