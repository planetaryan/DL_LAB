import torch
tensor = torch.rand(2,3)

max = torch.max(tensor)


min = torch.min(tensor)
print(max, min)