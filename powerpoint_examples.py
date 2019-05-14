# Pointwise loss on target of size (batch) and output of size
# (batch, 1) is wrong and occurs frequently.
>>>> loss = lambda x, y: (x - y)**2
>>> output = torch.randn(10, 1)
>>> targets = torch.randn(10)
>>> loss(output, targets).shape
torch.Size(10, 10)



x.is_contiguous(memory_format=torch.channels_last)
x.contiguous(memory_format=torch.channels_last)


ims1 = torch.randn(100, 50, 50, 3)  # NHWC
ims2 = torch.randn(100, 3, 50, 50)  # NCHW

def rotate(ims):
    return ims.transpose(1, 2)

ims1 = rotate(ims1)
ims2 = rotate(ims2)  # NOT what we wanted.

ims2.align_as(ims1) * ims1

>>> tensor = torch.tensor([[1, 0], [0, 1],
                           names=['height', 'width'])

#  You can also pass a names argument to a tensor factory function
>>> tensor = torch.randn(2, 1, 2
                         names=['batch', 'height', 'width'])
tensor([[[-0.9437, -1.8355]],
        [[-1.1989, -0.4061]]],
       names=['batch', 'width', 'height'])

# Python 3.6+
>>> tensor = torch.randn(batch=2, height=1, width=2)

