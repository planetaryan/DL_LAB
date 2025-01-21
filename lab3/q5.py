import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Input data
x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2], dtype=torch.float32).unsqueeze(1)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6], dtype=torch.float32).unsqueeze(1)

# Define the model
model = nn.Linear(1, 1)  # Linear regression model with one input and one output

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Stochastic Gradient Descent

loss_list = []

# Training loop
for epoch in range(100):
    # Forward pass: compute predictions
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # Backward pass: compute gradients
    optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters

    # Record the loss
    loss_list.append(loss.item())
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

# Extract learned parameters
w, b = model.weight.item(), model.bias.item()
print(f"Learned parameters: w = {w:.4f}, b = {b:.4f}")

# Plot the loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(loss_list, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.show()