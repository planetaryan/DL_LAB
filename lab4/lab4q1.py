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

# Define the loss function (Binary Cross-Entropy Loss)
criterion = nn.BCELoss()

# Define the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training the model
epochs = 10000  # number of iterations
for epoch in range(epochs):
    # Forward pass: compute predicted output by passing inputs to the model
    predictions = model(X)
    
    # Compute the loss
    loss = criterion(predictions, y)
    
    # Backward pass: compute gradients
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights using the optimizer
    optimizer.step()
    
    # Print the loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

# Testing the model after training
with torch.no_grad():
    test_outputs = model(X)
    print("\nPredictions after training:")
    print(test_outputs.round())  # Round the outputs to 0 or 1

