import torch

# Define a and b as PyTorch tensors with requires_grad=True to compute gradients
a = torch.tensor(1.0, requires_grad=True)  # You can replace 1.0 with any value of 'a'
b = torch.tensor(1.0, requires_grad=True)  # You can replace 1.0 with any value of 'b'

# Define x, y, and z based on the given equations
x = 2 * a + 3 * b
y = 5 * a * a + 3 * b * b * b
z = 2 * x + 3 * y

# Compute the gradient of z with respect to a
z.backward()

# Output the gradient dz/da
print("The gradient dz/da is:", a.grad)
