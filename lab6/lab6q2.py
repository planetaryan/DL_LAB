import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt

# 1. Set up the data transformations and data loaders
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 as required by AlexNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Download the cats_and_dogs_filtered dataset (assuming it's available in this folder)
data_dir = 'cats_and_dogs_filtered'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')

train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 2. Load the pre-trained AlexNet model
model = models.alexnet(pretrained=True)

# 3. Modify the classifier to be a two-class classifier
model.classifier[6] = nn.Linear(in_features=4096, out_features=2)  # Change output to 2 classes

# 4. Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 6. Fine-tuning the model (only training the classifier part)
# Optionally freeze the convolution layers
for param in model.features.parameters():
    param.requires_grad = False

# 7. Train the model
num_epochs = 5  # You can increase the number of epochs if needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
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
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')

# 8. Evaluate the model on the validation set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total
print(f'Validation Accuracy: {val_accuracy:.2f}%')

# 9. Save the trained model
torch.save(model.state_dict(), './alexnet_cats_dogs.pth')
print("Model saved successfully!")

# Epoch [1/5], Loss: 0.2065, Accuracy: 91.90%
# Epoch [2/5], Loss: 0.0958, Accuracy: 96.60%
# Epoch [3/5], Loss: 0.0676, Accuracy: 97.25%
# Epoch [4/5], Loss: 0.0522, Accuracy: 98.25%
# Epoch [5/5], Loss: 0.0361, Accuracy: 99.05%
# Validation Accuracy: 96.20%
# Model saved successfully!