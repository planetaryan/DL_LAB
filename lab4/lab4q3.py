import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset (4 samples, 2 features for binary inputs)
# Inputs: [x1, x2], Outputs: XOR of x1 and x2
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# Define the Feedforward Neural Network class
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # First layer (input -> hidden)
        self.layer1 = nn.Linear(2,2,bias=True)
        # Second layer (hidden -> output)
        self.layer2 = nn.Linear(2,1,bias=True)
        # Sigmoid activation for both layers
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply first layer and Sigmoid
        x = self.sigmoid(self.layer1(x))
        # Apply second layer and Sigmoid
        x = self.sigmoid(self.layer2(x))
        return x
    
# Instantiate the model
model = XORModel()

# Get the weights and biases from the model
weights_layer1 = model.layer1.weight
biases_layer1 = model.layer1.bias
weights_layer2 = model.layer2.weight
biases_layer2 = model.layer2.bias

print("Layer 1 Weights:\n", weights_layer1)
print("Layer 1 Biases:\n", biases_layer1)
print("Layer 2 Weights:\n", weights_layer2)
print("Layer 2 Biases:\n", biases_layer2)
