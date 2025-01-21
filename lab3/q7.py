import torch
import matplotlib.pyplot as plt

# Input data (x) and target labels (y)
x = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32).view(-1, 1)  # Feature (reshaped to column)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).view(-1, 1)  # Labels (reshaped to column)

# Initialize weights and bias
weights = torch.tensor([0.0], requires_grad=True)  # For 1 feature
bias = torch.tensor([0.0], requires_grad=True)  # For single target

# Define learning rate and epochs
learning_rate = 0.01
num_epochs = 1000

loss_list = []

# Sigmoid function (activation function)
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Binary Cross-Entropy Loss function
def binary_cross_entropy(y_pred, y):
    return torch.mean(-(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)))

# Training loop
for epoch in range(num_epochs):
    # Forward pass: Calculate predicted probabilities using the sigmoid function
    y_pred = sigmoid(x @ weights + bias)  # y_pred = sigmoid(wx + b)
    
    # Compute the loss (binary cross-entropy)
    loss = binary_cross_entropy(y_pred, y)
    
    # Backward pass: Compute gradients
    loss.backward()  # Compute gradients of the loss with respect to weights and bias

    # Update weights and bias manually using gradient descent
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad

    # Reset gradients to zero for the next iteration
    weights.grad.zero_()
    bias.grad.zero_()

    # Record the loss for plotting
    loss_list.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Final learned parameters
learned_weights = weights.item()
learned_bias = bias.item()
print(f"Learned weight: {learned_weights:.4f}, Learned bias: {learned_bias:.4f}")

