import torch
from torch.autograd import Variable
import timeit

a = torch.randn(1, 52)
b = torch.randn(1, 52)
c = torch.randn(52, 52)

print("addmm tensors", timeit.timeit('a.addmm(b, c)', number=100000, globals=locals()))

av = Variable(a)
bv = Variable(b)
cv = Variable(c)
print("addmm variables", timeit.timeit('av.addmm(bv, cv)', number=100000, globals=locals()))


print("add tensors", timeit.timeit('a.add(b)', number=100000, globals=locals()))
print("add variables (python)", timeit.timeit('av.add(bv)', number=100000, globals=locals()))
f=torch._C._functions.BatchNormAdd()
print("add variables (C++)", timeit.timeit('f(av,bv)', number=100000, globals=locals()))
