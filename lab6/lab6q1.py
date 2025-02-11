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

# Load the model's state_dict
model.load_state_dict(torch.load("cnn_mnist_state_dict.pth"))

# Now, print the model's state_dict to see the sizes of the parameters
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print()


model.eval()
correct = 0
total = 0
for i, vdata in enumerate(test_loader):
    tinputs, tlabels = vdata
    tinputs = tinputs.to(device)
    tlabels = tlabels.to(device)
    toutputs = model(tinputs)
    #Select the predicted class label which has the
    # highest value in the output layer
    _, predicted = torch.max(toutputs, 1)
    print("True label:{}".format(tlabels))
    print('Predicted: {}'.format(predicted))
    # Total number of labels
    total += tlabels.size(0)
    # Total correct predictions
    correct += (predicted == tlabels).sum()
accuracy = 100.0 * correct / total
print("The overall accuracy is {}".format(accuracy))

# Save model in 3 ways:

# a. Save the entire model
model_save_path = './model_entire.pth'
torch.save(model, model_save_path)
print(f"Entire model saved to {model_save_path}")

# b. Save only the state_dict (weights and parameters)
state_dict_save_path = './model_state_dict.pth'
torch.save(model.state_dict(), state_dict_save_path)
print(f"State dict saved to {state_dict_save_path}")

# c. Save as checkpoint (including optimizer and epoch if necessary)
checkpoint_save_path = './model_checkpoint.pth'
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': None,  # If optimizer was used, save its state_dict
    'epoch': 0  # Add the current epoch if you were training
}
torch.save(checkpoint, checkpoint_save_path)
print(f"Checkpoint saved to {checkpoint_save_path}")



# Model's state_dict:
# net.0.weight     torch.Size([64, 1, 3, 3])

# net.0.bias       torch.Size([64])

# net.3.weight     torch.Size([128, 64, 3, 3])

# net.3.bias       torch.Size([128])

# net.6.weight     torch.Size([64, 128, 3, 3])

# net.6.bias       torch.Size([64])

# classification_head.0.weight     torch.Size([20, 64])

# classification_head.0.bias       torch.Size([20])

# classification_head.2.weight     torch.Size([10, 20])

# classification_head.2.bias       torch.Size([10])


# Predicted: tensor([9, 1, 0, 1, 1, 0, 1, 0, 0, 2, 2, 0, 0, 4, 1, 1, 6, 6, 0, 0, 1, 0, 1, 5,
#         1, 0, 2, 8, 4, 2, 2, 1, 6, 0, 0, 8, 1, 2, 0, 0, 0, 8, 8, 3, 1, 1, 8, 0,
#         8, 4, 5, 4, 9, 3, 6, 6, 0, 0, 6, 2, 8, 2, 8, 0], device='cuda:0')
# True label:tensor([8, 8, 5, 8, 3, 8, 9, 3, 0, 0, 2, 1, 8, 4, 4, 0, 0, 9, 5, 3, 1, 5, 1, 3,
#         5, 0, 2, 6, 1, 5, 3, 0, 8, 6, 9, 6, 9, 6, 0, 4, 1, 6, 6, 4, 9, 1, 4, 9,
#         7, 8, 2, 6, 2, 9, 7, 8, 5, 6, 9, 6, 0, 0, 8, 1], device='cuda:0')
# Predicted: tensor([8, 7, 5, 7, 6, 3, 1, 6, 8, 8, 0, 0, 7, 0, 0, 4, 8, 2, 2, 8, 0, 7, 0, 1,
#         1, 8, 8, 0, 8, 6, 8, 8, 6, 0, 6, 0, 1, 0, 8, 0, 0, 0, 4, 1, 1, 8, 0, 6,
#         0, 8, 0, 1, 0, 1, 4, 0, 2, 2, 2, 8, 2, 3, 0, 0], device='cuda:0')
# True label:tensor([3, 2, 7, 5, 8, 4, 5, 6, 8, 9, 1, 9, 1, 8, 1, 5], device='cuda:0')
# Predicted: tensor([8, 9, 6, 5, 4, 8, 5, 8, 4, 1, 0, 2, 8, 6, 1, 2], device='cuda:0')
# The overall accuracy is 7.659999847412109
# Entire model saved to ./model_entire.pth
# State dict saved to ./model_state_dict.pth
# Checkpoint saved to ./model_checkpoint.pth