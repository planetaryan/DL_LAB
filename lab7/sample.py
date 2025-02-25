import torch
import PIL
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import glob
import os
from matplotlib import pyplot as plt

# Gaussian noise class for data augmentation
class Gaussian(object):
    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img + torch.normal(self.mean, self.var, img.size())

# Custom dataset class to load images and apply transformations
class MyDataset(Dataset):
    def __init__(self, transform=None, str="train"):
        self.imgs_path = r"/home/student/Downloads/cats_and_dogs_filtered/cats_and_dogs_filtered"
        print(f"Looking for images in: {self.imgs_path}")
        
        if not os.path.exists(self.imgs_path):
            print(f"Error: Path {self.imgs_path} does not exist.")
        
        file_list = glob.glob(os.path.join(self.imgs_path, str, "*"))
        self.data = []
        for class_path in file_list:
            class_name = os.path.basename(class_path)
            print(f"Processing class: {class_name}")
            if not os.path.isdir(class_path):
                print(f"Warning: {class_path} is not a directory.")
            for img_path in glob.glob(os.path.join(class_path, "*.jpg")):
                print(f"Found image: {img_path}")
                self.data.append([img_path, class_name])
        
        if len(self.data) == 0:
            print("Warning: No images found in the dataset.")
        
        self.class_map = {"dogs": 0, "cats": 1}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = PIL.Image.open(img_path)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        
        if self.transform:
            img = self.transform(img)
        
        return img, class_id

# Apply resizing and data augmentation in the preprocessing pipeline
preprocess = T.Compose([
    T.Resize((224, 224)),  # Resize all images to 224x224
    T.ToTensor(),
    T.RandomHorizontalFlip(),
    T.RandomRotation(45),
    Gaussian(0, 0.15),  # Adding Gaussian noise
])

# Instantiate the dataset and dataloader with data augmentation
train_dataset_with_aug = MyDataset(transform=preprocess, str="train")
train_dataloader_with_aug = DataLoader(train_dataset_with_aug, batch_size=32, shuffle=True)

# Example of displaying a few augmented images
# Example of displaying a few augmented images
i = 0
for data in iter(train_dataloader_with_aug):
    img, _ = data
    print(f"Batch image shape: {img.shape}")
    img_single = img[0]  # Extract the first image from the batch
    img_pil = T.ToPILImage()(img_single)  # Convert to PIL Image
    plt.imshow(img_pil)
    plt.show()
    i += 1
    if i == 3:
        break


# Training function (simple example, can be extended for your full training loop)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # This is the missing import

# Simple CNN model (you can replace this with a more complex model like ResNet or VGG)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 112 * 112, 2)  # 224x224 -> 112x112 after pool

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Using F.relu correctly after importing F
        x = x.view(-1, 32 * 112 * 112)  # Flatten
        x = self.fc1(x)
        return x

# Instantiate the model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model with data augmentation
def train_model(dataloader, model, criterion, optimizer):
    model.train()  # Set the model to training mode
    for epoch in range(1, 11):  # Example: 10 epochs
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {running_loss/len(dataloader)}")

# Start the training loop
train_model(train_dataloader_with_aug, model, criterion, optimizer)

# Epoch 1, Loss: 1.3061929070760334
# Epoch 2, Loss: 0.6990582498293074
# Epoch 3, Loss: 0.6815609080450875
# Epoch 4, Loss: 0.6784006678868854
# Epoch 5, Loss: 0.6445542782072037
# Epoch 6, Loss: 0.6524916253392659
# Epoch 7, Loss: 0.6893766740011791
# Epoch 8, Loss: 0.652480715797061
# Epoch 9, Loss: 0.6376492977142334
# Epoch 10, Loss: 0.6417357760762411