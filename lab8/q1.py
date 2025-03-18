import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Load data
df = pd.read_csv("natural_gas_datahub.csv")
df = df.dropna()

y = df['Price'].values
minm, maxm = y.min(), y.max()
y = (y - minm) / (maxm - minm)

# Prepare sequences
Sequence_Length = 10
X, Y = [], []

for i in range(len(y) - Sequence_Length - 1):
    X.append(y[i:i + Sequence_Length])
    Y.append(y[i + Sequence_Length])

X, Y = np.array(X), np.array(Y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=False)

class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return self.len

# Create data loaders
dataset = NGTimeSeries(x_train, y_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=256)

test_dataset = NGTimeSeries(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=256)

# Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output

# Model Initialization
model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 500
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.view(-1, Sequence_Length, 1)
        optimizer.zero_grad()
        y_pred = model(batch_x).squeeze()
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Save Model
torch.save(model.state_dict(), "rnn_natural_gas.pth")

# Load Model for Testing
model.load_state_dict(torch.load("rnn_natural_gas.pth"))
model.eval()

# Testing
predictions = []
actuals = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.view(-1, Sequence_Length, 1)
        y_pred = model(batch_x).squeeze()
        predictions.extend(y_pred.numpy())
        actuals.extend(batch_y.numpy())

# Undo Normalization
actuals = np.array(actuals) * (maxm - minm) + minm
predictions = np.array(predictions) * (maxm - minm) + minm

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(actuals, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='red', linestyle='dashed')
plt.legend()
plt.show()


# Epoch 0, Loss: 0.022681
# Epoch 50, Loss: 0.003947
# Epoch 100, Loss: 0.004586
# Epoch 150, Loss: 0.007910
# Epoch 200, Loss: 0.003950
# Epoch 250, Loss: 0.002470
# Epoch 300, Loss: 0.004261
# Epoch 350, Loss: 0.007919
# Epoch 400, Loss: 0.007218
# Epoch 450, Loss: 0.005994
