import torch

if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available

# Create tensor (default on CPU)
tensor1 = torch.rand(2,3)
tensor2 = torch.rand(2,3)
# Tensor not on GPU
print(tensor1, tensor1.device)
print(tensor2, tensor2.device)
# Move tensor to GPU (if available)
tensor1_on_gpu = tensor1.to(device)
tensor2_on_gpu = tensor2.to(device)
print(tensor1_on_gpu)
print(tensor2_on_gpu)