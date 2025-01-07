# Create a tensor
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)

# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}")
print(f"Second square bracket: {x[0][0]}")
print(f"Third square bracket: {x[0][0][0]}")

print(x[:, 0])