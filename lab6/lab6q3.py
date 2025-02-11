import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

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
            nn.Linear(64, 20, bias=True),  # Adjust based on the output size of the last conv layer
            nn.ReLU(),
            nn.Linear(20, 10, bias=True)  # 10 classes for MNIST digits (0-9)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(features.size(0), -1))

# Load FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizing
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = CNNClassifier().to(device)

# Initialize the optimizer (e.g., SGD or Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if a checkpoint exists and resume training if possible
checkpoint_path = 'model_checkpoint.pth'
start_epoch = 0  # Default to 0 if no checkpoint exists

if os.path.exists(checkpoint_path):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    print(f"Resuming from epoch {start_epoch}...")

# Move model to the appropriate device (GPU or CPU)
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5  # Specify how many more epochs to train
for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for i, vdata in enumerate(train_loader):
        inputs, labels = vdata
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print training stats
    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')

    # Save a checkpoint after each epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': running_loss  # Or whatever metric you're tracking
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

# After training, save the final model state
torch.save(model.state_dict(), './final_model.pth')
print("Final model saved.")


# Resuming from epoch 1...
# Epoch [2/6], Loss: 0.3530, Accuracy: 87.31%
# Checkpoint saved at epoch 2
# Epoch [3/6], Loss: 0.3023, Accuracy: 89.06%
# Checkpoint saved at epoch 3
# Epoch [4/6], Loss: 0.2721, Accuracy: 90.10%
# Checkpoint saved at epoch 4
# Epoch [5/6], Loss: 0.2427, Accuracy: 91.24%
# Checkpoint saved at epoch 5
# Epoch [6/6], Loss: 0.2174, Accuracy: 92.11%
# Checkpoint saved at epoch 6
# Final model saved.
