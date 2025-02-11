import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
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
model = CNNClassifier().to(device)

# Define optimizer (use Adam or SGD for example)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the model's state_dict
model.load_state_dict(torch.load("cnn_mnist_state_dict.pth"))

# Print the model's state_dict to see the sizes of the parameters
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print()

# Now, training code to update optimizer state (Example: 1 epoch of training)
model.train()  # Set model to training mode
for epoch in range(1):  # Example: 1 epoch
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Save model in 3 ways:

# a. Save the entire model
model_save_path = './model_entire.pth'
torch.save(model, model_save_path)
print(f"Entire model saved to {model_save_path}")

# b. Save only the state_dict (weights and parameters)
state_dict_save_path = './model_state_dict.pth'
torch.save(model.state_dict(), state_dict_save_path)
print(f"State dict saved to {state_dict_save_path}")

# c. Save as checkpoint (including optimizer and epoch)
checkpoint_save_path = './model_checkpoint.pth'
checkpoint = {
    'epoch': 0,  # Add the current epoch if you were training
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer's state dict
}
torch.save(checkpoint, checkpoint_save_path)
print(f"Checkpoint saved to {checkpoint_save_path}")
