import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import glob
import os
from PIL import Image
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

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()

# L2 Regularization via weight decay (optimizer)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)  # weight_decay is L2 regularization

# Training loop with L2 regularization via weight decay
def train_model_with_weight_decay(dataloader, model, criterion, optimizer):
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

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {running_loss/len(dataloader)}")

# Train the model with L2 regularization using weight decay
train_model_with_weight_decay(train_dataloader, model, criterion, optimizer)

# L2 Regularization via manual calculation of L2 norm
def train_model_with_manual_L2(dataloader, model, criterion, optimizer, l2_lambda=0.01):
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

            # L2 regularization term
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss += l2_lambda * l2_norm  # Add L2 penalty

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}, Loss (with L2): {running_loss/len(dataloader)}")

# Train the model with L2 regularization using a manual loop
train_model_with_manual_L2(train_dataloader, model, criterion, optimizer, l2_lambda=0.01)


# Check the weights before and after training
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name} - Weight Mean: {param.data.mean()}, Weight Std: {param.data.std()}")

# Epoch 1, Loss: 0.6972616740635463
# Epoch 2, Loss: 0.6728880301354423
# Epoch 3, Loss: 0.6544129309200105
# Epoch 4, Loss: 0.6411565428688413
# Epoch 5, Loss: 0.6412099382233998
# Epoch 6, Loss: 0.6188599968713427
# Epoch 7, Loss: 0.5989879043329329
# Epoch 8, Loss: 0.5708771698058598
# Epoch 9, Loss: 0.5651433108344911
# Epoch 10, Loss: 0.547850993890611
# Epoch 1, Loss (with L2): 0.8383321695857577
# Epoch 2, Loss (with L2): 0.8240781587267679
# Epoch 3, Loss (with L2): 0.7899495732216608
# Epoch 4, Loss (with L2): 0.7801135220224895
# Epoch 5, Loss (with L2): 0.7569902293265812
# Epoch 6, Loss (with L2): 0.7359914202538748
# Epoch 7, Loss (with L2): 0.7212411155776371
# Epoch 8, Loss (with L2): 0.6997482171134343
# Epoch 9, Loss (with L2): 0.6846445041989523
# Epoch 10, Loss (with L2): 0.657124053864252
# conv1.weight - Weight Mean: 0.0014247549697756767, Weight Std: 0.09176412969827652
# conv2.weight - Weight Mean: 0.0003695842460729182, Weight Std: 0.02766379714012146
# fc1.weight - Weight Mean: 2.0989584754715906e-06, Weight Std: 0.0022219724487513304