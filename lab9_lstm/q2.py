import torch
import torch.nn as nn
import torch.optim as optim
import zipfile
import os
import urllib.request
import string
import random
import time
import math

# Download and extract the dataset
url = "https://download.pytorch.org/tutorial/data.zip"
data_path = "data.zip"
urllib.request.urlretrieve(url, data_path)
with zipfile.ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall(".")

# Load names from files
data_dir = "./data/names/"
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicode_to_ascii(s):
    """Convert string to ASCII, removing accents."""
    return ''.join(
        c for c in s if c in all_letters
    )

# Read names and organize by language
category_lines = {}
all_categories = []
for filename in os.listdir(data_dir):
    category = os.path.splitext(filename)[0]
    all_categories.append(category)
    with open(os.path.join(data_dir, filename), encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
        category_lines[category] = [unicode_to_ascii(line) for line in lines]

n_categories = len(all_categories)  # Should be 18

# Prepare training data: select a few thousand names
n_samples_per_category = 200  # Adjust to get ~3600 total samples
training_data = []
for category in all_categories:
    names = random.sample(category_lines[category], min(n_samples_per_category, len(category_lines[category])))
    for name in names:
        training_data.append((name, category))

random.shuffle(training_data)
print(f"Total training samples: {len(training_data)}")

def line_to_tensor(line):
    tensor = torch.zeros(len(line), n_letters)
    for li, letter in enumerate(line):
        tensor[li][all_letters.find(letter)] = 1
    # Add batch dimension: (sequence_length, n_letters) -> (1, sequence_length, n_letters)
    tensor = tensor.unsqueeze(0)
    return tensor

# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.fc(output[:, -1, :])  # Take the last time step
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        # Hidden state shape: (num_layers, batch_size, hidden_size)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

# Model parameters
hidden_size = 128
model = LSTMClassifier(n_letters, hidden_size, n_categories)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(category_tensor, line_tensor):
    hidden = model.init_hidden()
    model.zero_grad()
    output, hidden = model(line_tensor, hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()

# Training loop
n_iters = 100000
print_every = 5000
plot_every = 1000
current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'

start = time.time()

for iter in range(1, n_iters + 1):
    # Sample a random training pair
    name, category = random.choice(training_data)
    line_tensor = line_to_tensor(name)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print progress
    if iter % print_every == 0:
        guess = all_categories[torch.argmax(output, dim=1).item()]
        correct = '✓' if guess == category else f'✗ ({category})'
        print(f'{iter} {iter/n_iters*100:.0f}% ({time_since(start)}) {loss:.4f} {name} / {guess} {correct}')

    # Record average loss
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# Prediction function
def predict(input_line, n_predictions=3):
    print(f'\n> {input_line}')
    with torch.no_grad():
        line_tensor = line_to_tensor(unicode_to_ascii(input_line))
        hidden = model.init_hidden()
        output, hidden = model(line_tensor, hidden)
        probs = torch.exp(output)
        topv, topi = probs.topk(n_predictions, dim=1)
        predictions = []
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            predictions.append((value, all_categories[category_index]))
        for value, category in predictions:
            print(f'{category}: {value:.2f}')

# Test some example names
test_names = ['Dostoevsky', 'Satoshi', 'O’Connell', 'Müller']
for name in test_names:
    predict(name)


# 5000 5% (0m 5s) 3.1031 Bonnay / English ✗ (French)
# 10000 10% (0m 10s) 0.4290 Niall / Irish ✓
# 15000 15% (0m 15s) 0.0025 Tubylov / Russian ✓
# 20000 20% (0m 20s) 0.0192 Idane / Japanese ✓
# 25000 25% (0m 25s) 0.2586 Lao / Chinese ✓
# 30000 30% (0m 30s) 0.0001 O'Loughlin / Irish ✓
# 35000 35% (0m 34s) 0.0012 O'Doherty / Irish ✓
# 40000 40% (0m 39s) 0.0079 Favreau / French ✓
# 45000 45% (0m 44s) 0.3548 Olivier / French ✓
# 50000 50% (0m 49s) 0.0058 Hirasi / Japanese ✓
# 55000 55% (0m 54s) 0.0003 Wojewdka / Polish ✓
# 60000 60% (0m 59s) 0.0001 Acciai / Italian ✓
# 65000 65% (1m 4s) 0.1828 Yong / Chinese ✓
# 70000 70% (1m 9s) 0.0002 Desrosiers / French ✓
# 75000 75% (1m 14s) 0.0044 Kouba / Czech ✓
# 80000 80% (1m 19s) 0.0097 Snell / Dutch ✓
# 85000 85% (1m 24s) 0.0838 Eadie / English ✓
# 90000 90% (1m 29s) 0.0312 Piller / Czech ✓
# 95000 95% (1m 34s) 0.0244 Duval / French ✓
# 100000 100% (1m 39s) 0.0000 Hlebanov / Russian ✓

# > Dostoevsky
# Russian: 1.00
# Irish: 0.00
# Czech: 0.00

# > Satoshi
# Japanese: 1.00
# Polish: 0.00
# Arabic: 0.00

# > O’Connell
# Irish: 1.00
# German: 0.00
# English: 0.00

# > Müller
# German: 0.98
# Czech: 0.01
# Scottish: 0.01