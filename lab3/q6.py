import torch
import matplotlib.pyplot as plt

X = torch.tensor([[3.0, 8.0], [4.0, 5.0], [5.0, 7.0], [6.0, 3.0], [2.0, 1.0]])

# Target values (y)
y = torch.tensor([[-3.7], [3.5], [2.5], [11.5], [5.7]])

# Initialize weights and bias
weights = torch.tensor([[1.0], [1.0]], requires_grad=True)  
bias = torch.tensor([1.0], requires_grad=True)  

# Define learning rate and epochs
learning_rate = 0.001
num_epochs = 100

loss_list = []

# Training loop
for epoch in range(num_epochs):
    # Forward pass: Calculate predictions
    predictions = X @ weights + bias  # Matrix multiplication + bias addition

    # Compute Mean Squared Error loss
    loss = torch.mean((predictions - y) ** 2)

    # Backward pass: Compute gradients
    loss.backward()

    # Update weights and bias manually
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad

    # Reset gradients to zero for the next iteration
    weights.grad.zero_()
    bias.grad.zero_()

    # Record the loss
    loss_list.append(loss.item())
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

w1,w2=weights.detach().numpy()
b=bias.detach().numpy()
# Final parameters
print(f"Learned weights: {w1},{w2}")
print(f"Learned bias: {b}")

print(f'Answer for X1=3,X2=2: {w1*3 +w2*2 + b}')

# Plot the loss curve
plt.plot(loss_list, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.show()
