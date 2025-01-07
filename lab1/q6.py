import torch

# Create tensor with specific shape
x_original = torch.rand(size=(1,7))

x_random = torch.rand(1,7)

ans=torch.matmul(x_random.T,x_original)

print(ans)