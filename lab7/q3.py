import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define transformations (resize and normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet-like normalization
])

# Download the Cats vs Dogs dataset (Assuming dataset is available locally)
train_data = datasets.ImageFolder(root='/home/student/Downloads/cats_and_dogs_filtered/cats_and_dogs_filtered/train', transform=transform)
test_data = datasets.ImageFolder(root='/home/student/Downloads/cats_and_dogs_filtered/cats_and_dogs_filtered/validation', transform=transform)

# Create DataLoader for batching
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Model with Dropout
class ModelWithDropout(nn.Module):
    def __init__(self):
        super(ModelWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply Dropout
        x = self.fc2(x)
        return x


# Model without Dropout
class ModelWithoutDropout(nn.Module):
    def __init__(self):
        super(ModelWithoutDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# Model Evaluation
def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Initialize models, criterion, and optimizer
model_with_dropout = ModelWithDropout()
model_without_dropout = ModelWithoutDropout()

criterion = nn.CrossEntropyLoss()
optimizer_with_dropout = optim.Adam(model_with_dropout.parameters(), lr=0.001)
optimizer_without_dropout = optim.Adam(model_without_dropout.parameters(), lr=0.001)

# Train model with dropout
print("Training Model with Dropout:")
train_model(model_with_dropout, train_loader, criterion, optimizer_with_dropout, num_epochs=5)
evaluate_model(model_with_dropout, test_loader, criterion)

# Save model with dropout parameters
torch.save(model_with_dropout.state_dict(), 'model_with_dropout.pth')
print("Model with Dropout saved.")

# Train model without dropout
print("\nTraining Model without Dropout:")
train_model(model_without_dropout, train_loader, criterion, optimizer_without_dropout, num_epochs=5)
evaluate_model(model_without_dropout, test_loader, criterion)

# Save model without dropout parameters
torch.save(model_without_dropout.state_dict(), 'model_without_dropout.pth')
print("Model without Dropout saved.")

# Training Model with Dropout:
# Epoch [1/5], Loss: 0.8649, Accuracy: 0.5435
# Epoch [2/5], Loss: 0.6338, Accuracy: 0.6450
# Epoch [3/5], Loss: 0.5751, Accuracy: 0.6925
# Epoch [4/5], Loss: 0.5143, Accuracy: 0.7500
# Epoch [5/5], Loss: 0.4245, Accuracy: 0.8100
# Test Loss: 0.6750, Test Accuracy: 0.6640
# Model with Dropout saved.

# Training Model without Dropout:
# Epoch [1/5], Loss: 0.9351, Accuracy: 0.5170
# Epoch [2/5], Loss: 0.6829, Accuracy: 0.5585
# Epoch [3/5], Loss: 0.6191, Accuracy: 0.6620
# Epoch [4/5], Loss: 0.5524, Accuracy: 0.7245
# Epoch [5/5], Loss: 0.4507, Accuracy: 0.7880
# Test Loss: 0.6074, Test Accuracy: 0.6740
# Model without Dropout saved.