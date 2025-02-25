import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import glob
import os
from PIL import Image
import torch.nn.functional as F
from matplotlib import pyplot as plt

# Custom Dataset Class for loading images
class MyDataset(Dataset):
    def __init__(self, transform=None, str="train"):
        self.imgs_path = r"/home/student/Downloads/cats_and_dogs_filtered/cats_and_dogs_filtered"
        file_list = glob.glob(os.path.join(self.imgs_path, str, "*"))
        self.data = []
        for class_path in file_list:
            class_name = os.path.basename(class_path)
            for img_path in glob.glob(os.path.join(class_path, "*.jpg")):
                self.data.append([img_path, class_name])
        
        self.class_map = {"dogs": 0, "cats": 1}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        
        if self.transform:
            img = self.transform(img)
        
        return img, class_id

# Transform to resize and convert to tensor
preprocess = T.Compose([
    T.Resize((224, 224)),  # Resize all images to 224x224
    T.ToTensor()
])

# Instantiate the dataset and dataloader without data augmentation
train_dataset = MyDataset(transform=preprocess, str="train")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 2)  # 224x224 -> 56x56 after pooling

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = self.fc1(x)
        return x


# L1 Regularization via manual calculation of L1 norm
def train_model_with_L1(dataloader, model, criterion, optimizer, l1_lambda=0.01):
    model.train()
    for epoch in range(1, 11):  # Train for 10 epochs
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # L1 regularization term
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm  # Add L1 penalty

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}, Loss (with L1): {running_loss/len(dataloader)}")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model with L1 regularization using a manual loop
train_model_with_L1(train_dataloader, model, criterion, optimizer, l1_lambda=0.01)

# Check the weights before and after training
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name} - Weight Mean: {param.data.mean()}, Weight Std: {param.data.std()}")

# Optionally, visualize some of the weights to see their sparsity
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"Visualizing weights for: {name}")
        plt.hist(param.data.cpu().numpy().flatten(), bins=50)
        plt.title(f"Histogram of {name} weights")
        plt.show()

# Epoch 1, Loss (with L1): 7.827903528062124
# Epoch 2, Loss (with L1): 5.559012753622873
# Epoch 3, Loss (with L1): 4.562219861953977
# Epoch 4, Loss (with L1): 3.6964390088641452
# Epoch 5, Loss (with L1): 2.952551247581603
# Epoch 6, Loss (with L1): 2.335782633887397
# Epoch 7, Loss (with L1): 1.8476635615030925
# Epoch 8, Loss (with L1): 1.4888232303044153
# Epoch 9, Loss (with L1): 1.2607104380925496
# Epoch 10, Loss (with L1): 1.1550198320358518
# conv1.weight - Weight Mean: 0.0011062786215916276, Weight Std: 0.06158394366502762
# conv2.weight - Weight Mean: -5.65782691808181e-08, Weight Std: 2.53840353252599e-05
# fc1.weight - Weight Mean: 1.553652317909382e-08, Weight Std: 1.3393558219831903e-05
# Visualizing weights for: conv1.weight
# Visualizing weights for: conv2.weight
# Visualizing weights for: fc1.weight