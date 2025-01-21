import torch


x = torch.tensor(1.0, requires_grad=True)  
y = torch.tensor(1.0, requires_grad=True)  
z = torch.tensor(1.0, requires_grad=True)  

# Define x, y, and z based on the given equations


f= torch.tanh(torch.log(1 + ((z*2*x)/torch.sin(y))))

# Compute the gradient of z with respect to a
f.backward()

# Output the gradient dz/da
print("The gradient df/dx is:", y.grad)
