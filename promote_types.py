import torch

def result_type(*args):
  return torch._promote_types(*tuple(infer_scalar_type(arg) for arg in args if participates(arg, args)))

def infer_scalar_type(arg):
  if isinstance(arg, bool):
      return torch.bool
  elif isinstance(arg, float):
    return torch.get_default_dtype()
  elif isinstance(arg, int):
      return torch.int64
  else:
    return arg.dtype

def participates(arg, args):
  if priority(arg) >= max(priority(other) for other in args):
    return True
  if category(arg) > max(category(other) for other in args if priority(other) > priority(arg)):
   return True
  return False

def priority(arg):
  if isinstance(arg, torch.Tensor) and arg.dim() > 0: return 3
  elif isinstance(arg, torch.Tensor): return 2
  else: return 1

def category(arg):
  if isinstance(arg, bool) or (isinstance(arg, torch.Tensor) and arg.dtype == torch.bool): return 0
  if isinstance(arg, float) or (isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point): return 2
  else: return 1

print(result_type(2.5, torch.tensor([[1, 2, 3]])))
print(result_type(True, torch.tensor([[1, 2, 3]])))
