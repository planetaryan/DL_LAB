import torch
tensor = torch.rand(2,3)

# Find arg max
arg_max = torch.argmax(tensor)

# Find arg min
arg_min = torch.argmin(tensor)
print(arg_max, arg_min)