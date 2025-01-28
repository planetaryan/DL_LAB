import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# Transform: Convert to tensor and normalize
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download MNIST dataset (train and test)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size_2, output_size)  # Output layer
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after the second hidden layer
        x = self.fc3(x)  # Output layer (no activation here, softmax will be done during evaluation)
        return x


# Define the model, loss function, and optimizer
model = FeedForwardNN(input_size=28*28, hidden_size_1=128, hidden_size_2=64, output_size=10)

# Loss function: CrossEntropyLoss (suitable for classification)
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
epochs = 10
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # Flatten the images (batch_size, 28, 28) -> (batch_size, 784)
        inputs = inputs.view(inputs.size(0), -1)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Get the predicted labels
        _, predicted = torch.max(outputs.data, 1)

        # Count the number of correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate the accuracy for the current epoch
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")



# Evaluate the model on the test dataset
model.eval()  # Set the model to evaluation mode
test_correct = 0
test_total = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        # Flatten the images
        inputs = inputs.view(inputs.size(0), -1)
        
        # Forward pass
        outputs = model(inputs)
        
        # Get the predicted labels
        _, predicted = torch.max(outputs, 1)
        
        # Count the correct predictions
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # Store labels and predictions for confusion matrix
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

# Calculate accuracy
accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {accuracy:.2f}%")

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Display the confusion matrix
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
# Print the total number of learnable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of learnable parameters: {num_params}")
