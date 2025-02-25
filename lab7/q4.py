import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Custom Dropout Layer using Bernoulli Distribution
class CustomDropout(nn.Module):
    def __init__(self, p: float):
        super(CustomDropout, self).__init__()
        self.p = p  # The dropout rate (probability of dropping a neuron)
    
    def forward(self, x):
        if self.training:  # Dropout is applied only during training
            # Generate a mask using Bernoulli distribution
            mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            # Scale the mask by the dropout probability to maintain the expected sum
            return x * mask / (1 - self.p)
        else:
            return x

class ModelWithCustomDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(ModelWithCustomDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = CustomDropout(dropout_prob)  # Custom dropout layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply custom dropout
        x = self.fc2(x)
        return x

class ModelWithLibraryDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(ModelWithLibraryDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(dropout_prob)  # Built-in dropout layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply built-in dropout
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

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Assuming dataset is stored in the given directory paths
train_data = datasets.ImageFolder(root='/home/student/Downloads/cats_and_dogs_filtered/cats_and_dogs_filtered/train', transform=transform)
test_data = datasets.ImageFolder(root='/home/student/Downloads/cats_and_dogs_filtered/cats_and_dogs_filtered/validation', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize models
model_with_custom_dropout = ModelWithCustomDropout(dropout_prob=0.5)
model_with_library_dropout = ModelWithLibraryDropout(dropout_prob=0.5)

# Define the loss function and optimizers after model initialization
criterion = nn.CrossEntropyLoss()
optimizer_custom_dropout = optim.Adam(model_with_custom_dropout.parameters(), lr=0.001)
optimizer_library_dropout = optim.Adam(model_with_library_dropout.parameters(), lr=0.001)

# Train and evaluate the model with custom dropout
print("Training Model with Custom Dropout:")
train_model(model_with_custom_dropout, train_loader, criterion, optimizer_custom_dropout, num_epochs=5)
evaluate_model(model_with_custom_dropout, test_loader, criterion)

# Train and evaluate the model with built-in dropout
print("\nTraining Model with Built-in Dropout:")
train_model(model_with_library_dropout, train_loader, criterion, optimizer_library_dropout, num_epochs=5)
evaluate_model(model_with_library_dropout, test_loader, criterion)

# Training Model with Custom Dropout:
# Epoch [1/5], Loss: 0.8857, Accuracy: 0.5360
# Epoch [2/5], Loss: 0.6649, Accuracy: 0.6105
# Epoch [3/5], Loss: 0.6281, Accuracy: 0.6610
# Epoch [4/5], Loss: 0.5679, Accuracy: 0.7140
# Epoch [5/5], Loss: 0.5072, Accuracy: 0.7485
# Test Loss: 0.6917, Test Accuracy: 0.6540

# Training Model with Built-in Dropout:
# Epoch [1/5], Loss: 0.8182, Accuracy: 0.5455
# Epoch [2/5], Loss: 0.6594, Accuracy: 0.6165
# Epoch [3/5], Loss: 0.6030, Accuracy: 0.6730
# Epoch [4/5], Loss: 0.5623, Accuracy: 0.7135