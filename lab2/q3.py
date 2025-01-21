import torch


x = torch.tensor(1.0, requires_grad=True)  
b = torch.tensor(1.0, requires_grad=True)  
w = torch.tensor(1.0, requires_grad=True)  




f= (w*x) + b 
if f>=0:
    a=f
else:
    a=0


# Compute the gradient of z with respect to a
a.backward()

# Output the gradient dz/da
print("The gradient df/dx is:", x.grad)
