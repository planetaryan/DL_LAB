import torch
import torch.nn as nn
import torch.nn.functional as F

# Input image of size (6, 6)
image = torch.rand(1, 1, 6, 6)  # Adding batch dimension and channel dimension

# Define a Conv2d layer with 1 input channel, 3 output channels, kernel size=3, stride=1, padding=0, bias=False
conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False)

# Apply Conv2d
output_conv = conv(image)
print("Output using torch.nn.Conv2d:", output_conv)
print("Output shape using torch.nn.Conv2d:", output_conv.shape)

# Get the weights from the Conv2d layer (kernel shape will be (3, 1, 3, 3))
kernel = conv.weight.data

# Perform the convolution using F.conv2d (without bias)
output_func = F.conv2d(image, kernel, stride=1, padding=0, bias=None)

print("Output using torch.nn.functional.conv2d:", output_func)
print("Output shape using torch.nn.functional.conv2d:", output_func.shape)
