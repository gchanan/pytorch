# Pointwise loss on target of size (batch) and output of size
# (batch, 1) is wrong and occurs frequently.
>>>> loss = lambda x, y: (x - y)**2
>>> output = torch.randn(10, 1)
>>> targets = torch.randn(10)
>>> loss(output, targets).shape
torch.Size(10, 10)



x.contiguous(memory_format=torch.channels_last)
x.is_contiguous(memory_format=torch.channels_last)


ims1 = torch.randn(100, 50, 50, 3)  # NHWC
ims2 = torch.randn(100, 3, 50, 50)  # NCHW

def rotate(ims):
    return ims.transpose(1, 2)

rotate(ims1)
rotate(ims2)  # NOT what we wanted.
