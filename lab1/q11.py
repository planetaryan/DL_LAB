import torch

# Set seed
torch.manual_seed(7)

# Create random tensor
tensor1 = torch.rand(size=(1, 1, 1, 10))

# Remove single dimensions
tensor2 = tensor1.squeeze()

# Print out tensors
print(tensor1, tensor1.shape)
print(tensor2, tensor2.shape)