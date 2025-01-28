import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset (4 samples, 2 features for binary inputs)
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

class XORNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(XORNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # input -> hidden
        self.layer2 = nn.Linear(hidden_size, output_size)  # hidden -> output
        self.relu = nn.ReLU()  # ReLU activation for hidden layer
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)  
        x = self.relu(x)  
        x = self.layer2(x)
        x = self.sigmoid(x)  
        return x  

# Define input size, hidden layer size, and output size
input_size = 2
hidden_size = 2
output_size = 1

model = XORNN(input_size, hidden_size, output_size)

# Define Mean Squared Error loss (no need for sigmoid)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 20000
for epoch in range(epochs):
    # Forward pass
    predictions = model(inputs)
    
    # Compute the loss
    loss = criterion(predictions, outputs)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 1000 epochs

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Testing after training
with torch.no_grad():
    test_outputs = model(inputs)
    print("\nPredictions after training:")
    print(test_outputs.round())  # Round the predictions to either 0 or 1
