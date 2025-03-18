import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import string
import unicodedata
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Load and process data
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' and c in string.ascii_letters + "'-")

def load_data():
    all_categories = []
    category_lines = {}
    for filename in glob.glob('./data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        category_lines[category] = [unicode_to_ascii(line) for line in lines]
    return category_lines, all_categories

category_lines, all_categories = load_data()
n_categories = len(all_categories)

# Convert names to tensors
def letter_to_index(letter):
    return string.ascii_letters.find(letter)

def name_to_tensor(name):
    tensor = torch.zeros(len(name), 1, len(string.ascii_letters))
    for li, letter in enumerate(name):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

# Prepare dataset
names = []
labels = []
for category, names_list in category_lines.items():
    for name in names_list:
        names.append(name)
        labels.append(all_categories.index(category))

x_train, x_test, y_train, y_test = train_test_split(names, labels, test_size=0.1, random_state=42)

class NameDataset(Dataset):
    def __init__(self, names, labels):
        self.names = names
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):
        return name_to_tensor(self.names[idx]), self.labels[idx]

train_dataset = NameDataset(x_train, y_train)
test_dataset = NameDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# Define RNN Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = self.fc(hidden[-1])
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Model Initialization
n_letters = len(string.ascii_letters)
hidden_size = 128
rnn = RNN(n_letters, hidden_size, n_categories)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.005)

# Training Loop
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for name_tensor, category_tensor in train_loader:
        hidden = rnn.init_hidden()
        optimizer.zero_grad()
        name_tensor = name_tensor.view(1, -1, n_letters)  # Ensure proper shape
        output, hidden = rnn(name_tensor, hidden)
        loss = criterion(output, category_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

# Save Model
torch.save(rnn.state_dict(), "rnn_name_classifier.pth")

# Load Model for Testing
rnn.load_state_dict(torch.load("rnn_name_classifier.pth"))
rnn.eval()

def predict(name):
    with torch.no_grad():
        hidden = rnn.init_hidden()
        name_tensor = name_to_tensor(name).view(1, -1, n_letters)  # Ensure correct shape
        output, hidden = rnn(name_tensor, hidden)
        category_index = torch.argmax(output).item()
        return all_categories[category_index]


# Testing
test_names = ["Aoife", "Giovanni", "Ivanov", "Smith"]
for name in test_names:
    print(f'{name} -> Predicted language: {predict(name)}')

# Epoch 1/5, Loss: 2.1896
# Epoch 2/5, Loss: 2.2311
# Epoch 3/5, Loss: 2.3091
# Epoch 4/5, Loss: 2.2976
# Epoch 5/5, Loss: 2.2194

# Aoife -> Predicted language: English
# Giovanni -> Predicted language: English
# Ivanov -> Predicted language: Russian
# Smith -> Predicted language: English