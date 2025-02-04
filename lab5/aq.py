import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# CNN model definition
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)           
        )
        self.classification_head = nn.Sequential(
            nn.Linear(64, 128, bias=True),  # Adjust based on the output size of the last conv layer
            nn.ReLU(),
            nn.Linear(128, 10, bias=True)  # 10 classes for Fashion MNIST (0-9)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(features.size(0), -1))

# Load Fashion MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to train the model
def train_model(num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for data, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update the model parameters

            running_loss += loss.item()

            # Get the number of correct predictions
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Function to evaluate the model and get predictions
def evaluate_model():
    model.eval()
    all_preds = []
    all_labels = []
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = 100 * correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm)

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Function to check the number of learnable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Train the model
train_model(num_epochs=5)

# Evaluate the model
evaluate_model()

# Check number of learnable parameters
num_params = count_parameters(model)
print(f"Number of learnable parameters: {num_params}")


# Epoch 1/5, Loss: 0.5771, Accuracy: 78.83%
# Epoch 2/5, Loss: 0.3820, Accuracy: 85.93%
# Epoch 3/5, Loss: 0.3233, Accuracy: 88.12%
# Epoch 4/5, Loss: 0.2855, Accuracy: 89.46%
# Epoch 5/5, Loss: 0.2580, Accuracy: 90.55%
# Test Accuracy: 88.56%
# Number of learnable parameters: 157898