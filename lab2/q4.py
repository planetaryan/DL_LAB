import torch
# Define a and b as PyTorch tensors with requires_grad=True to compute gradients

x = torch.tensor(1.0, requires_grad=True)  

# Define x, y, and z based on the given equations


f=torch.exp(-(x**2)-(2*x)-(torch.sin(x)))

# Compute the gradient of z with respect to a
f.backward()

# Output the gradient dz/da
print("The gradient df/dx is:", x.grad)
