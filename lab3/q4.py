import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define a custom dataset
class RegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Define the regression model by extending nn.Module
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Parameters w and b are defined as learnable parameters
        self.w = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.b = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self, x):
        return self.w * x + self.b  # Linear regression equation

# Input data
x = [5.0, 7.0, 12.0, 16.0, 20.0]
y = [40.0, 120.0, 180.0, 210.0, 240.0]

# Create dataset and dataloader
dataset = RegressionDataset(x, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model, loss function, and optimizer
model = RegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

loss_list = []

# Training loop
for epoch in range(100):
    epoch_loss = 0.0

    for batch_x, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()  # Reset gradients
        loss.backward()
        optimizer.step()  # Update parameters

        epoch_loss += loss.item()

    # Average loss for the epoch
    avg_loss = epoch_loss / len(dataloader)
    loss_list.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# Plot the loss over epochs
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.show()
