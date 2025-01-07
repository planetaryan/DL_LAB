import torch
x = torch.arange(1., 8.)
print(x, x.shape)


x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)

z = x.view(1, 7)
print(z, z.shape)

x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)



print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")




x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

