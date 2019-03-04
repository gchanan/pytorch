# Tensor[N, H, W, C]
x = torch.randn(100, 50, 50, 3)

# conv expects NCHW format, but the user forgot to transpose
# Tensor[N, H, W, C]
x = torch.randn(100, 50, 50, 3)
conv(x)

# Pointwise loss on target of size (batch) and output of size (batch, 1) is wrong
# and happens frequently.

loss = lambda x, y: (x - y)**2
output = torch.randn(10, 1)
targets = torch.randn(10)
>>> loss(output, targets).shape
torch.Size(10, 10)  # This is wrong

x = torch.randn(100, 50, 50, 3)
per_example_norm = torch.randn(100)
x + per_example_norm.unsqueeze(1).unsqueeze(1).unsqueeze(1)

# would be nice if we could add per_example_norm and x without unsqeezing 3 times.
# if x had, say, 50 dimensions, unsqueezing 50 times is not scalable.

x = torch.randn(100, 50, 3)
