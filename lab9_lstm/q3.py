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
all_letters = string.ascii_letters + " .,;'-" + "$"  # $ as end-of-string token
n_letters = len(all_letters)

def unicode_to_ascii(s):
    """Convert string to ASCII, removing accents."""
    return ''.join(c for c in s if c in all_letters[:-1]) + '$'

# Read all names
all_names = []
for filename in os.listdir(data_dir):
    with open(os.path.join(data_dir, filename), encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
        all_names.extend([unicode_to_ascii(line) for line in lines])

# Select a few thousand names
n_samples = 4000
training_names = random.sample(all_names, min(n_samples, len(all_names)))
print(f"Total training names: {len(training_names)}")

# Convert character to index and vice versa
char_to_idx = {char: idx for idx, char in enumerate(all_letters)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Convert name to tensor
def name_to_tensor(name):
    tensor = torch.zeros(len(name), n_letters)
    for i, char in enumerate(name):
        tensor[i][char_to_idx[char]] = 1
    return tensor

# LSTM Model
class LSTMCharPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCharPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

# Model parameters
hidden_size = 128
model = LSTMCharPredictor(n_letters, hidden_size, n_letters)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(input_tensor, target_tensor):
    hidden = model.init_hidden()
    model.zero_grad()
    output, hidden = model(input_tensor.unsqueeze(0), hidden)  # Add batch dimension
    loss = criterion(output.squeeze(0), target_tensor)
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
    # Sample a random name
    name = random.choice(training_names)
    
    # Prepare input and target tensors
    input_tensor = name_to_tensor(name[:-1])  # All chars except last
    target_tensor = torch.tensor([char_to_idx[char] for char in name[1:]], dtype=torch.long)
    
    output, loss = train(input_tensor, target_tensor)
    current_loss += loss

    # Print progress
    if iter % print_every == 0:
        # Get predicted characters
        _, topi = output.topk(1, dim=-1)
        predicted_chars = [idx_to_char[idx.item()] for idx in topi.squeeze()]
        predicted = ''.join(predicted_chars)
        correct = name[1:]  # Target sequence
        print(f'{iter} {iter/n_iters*100:.0f}% ({time_since(start)}) {loss:.4f} '
              f'Input: {name[:-1]}, Predicted: {predicted}, Target: {correct}')

    # Record average loss
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# Prediction function
def predict_next_char(start_string, max_len=20):
    model.eval()
    with torch.no_grad():
        # Prepare input
        start_string = unicode_to_ascii(start_string)
        if not start_string.endswith('$'):
            start_string = start_string[:-1] + '$'
        input_tensor = name_to_tensor(start_string[:-1])
        hidden = model.init_hidden()
        
        # Get predictions
        output, hidden = model(input_tensor.unsqueeze(0), hidden)
        predicted_chars = []
        
        # Start with the last character of input
        current_input = input_tensor[-1:].unsqueeze(0)  # Shape: (1, 1, n_letters)
        
        for _ in range(max_len):
            output, hidden = model(current_input, hidden)
            _, topi = output.topk(1, dim=-1)
            predicted_idx = topi.item()
            predicted_char = idx_to_char[predicted_idx]
            predicted_chars.append(predicted_char)
            if predicted_char == '$':
                break
            # Prepare next input
            current_input = torch.zeros(1, 1, n_letters)
            current_input[0, 0, predicted_idx] = 1
        
        return ''.join(predicted_chars)

# Test some starting strings
test_strings = ['Joh', 'Smi', 'Kat', 'Mül']
for s in test_strings:
    print(f'\nStarting string: {s}')
    predicted = predict_next_char(s)
    print(f'Predicted next sequence: {predicted}')


# 5000 5% (0m 5s) 2.3343 Input: Adabir, Predicted: ranan$, Target: dabir$
# 10000 10% (0m 10s) 2.0170 Input: Nasetkin, Predicted: ahan$on$, Target: asetkin$
# 15000 15% (0m 16s) 2.2856 Input: Toien, Predicted: unn$$, Target: oien$
# 20000 20% (0m 21s) 1.9998 Input: Serejin, Predicted: hranin$, Target: erejin$
# 25000 25% (0m 27s) 1.6328 Input: Ataev, Predicted: larv$, Target: taev$
# 30000 30% (0m 32s) 1.3081 Input: Antoun, Predicted: btour$, Target: ntoun$
# 35000 35% (0m 37s) 1.9741 Input: Poole, Predicted: arrl$, Target: oole$
# 40000 40% (0m 43s) 1.4996 Input: Kawasie, Predicted: alanae$, Target: awasie$
# 45000 45% (0m 48s) 1.3306 Input: Glaziev, Predicted: aeninv$, Target: laziev$
# 50000 50% (0m 53s) 1.6719 Input: Sumner, Predicted: hm$er$, Target: umner$
# 55000 55% (0m 59s) 0.9906 Input: Yampolsky, Predicted: anaolsky$, Target: ampolsky$
# 60000 60% (1m 4s) 1.2804 Input: Otton, Predicted: 'son$, Target: tton$
# 65000 65% (1m 9s) 1.3511 Input: Baroch, Predicted: abach$, Target: aroch$
# 70000 70% (1m 15s) 1.0454 Input: Nessler, Predicted: ass$er$, Target: essler$
# 75000 75% (1m 20s) 1.1660 Input: Beinenson, Predicted: alrerson$, Target: einenson$
# 80000 80% (1m 25s) 0.7939 Input: Yachnik, Predicted: anhnik$, Target: achnik$
# 85000 85% (1m 31s) 1.8937 Input: Hankoev, Predicted: andovv$, Target: ankoev$
# 90000 90% (1m 36s) 1.2471 Input: Lytkin,, Predicted: ezkon,$, Target: ytkin,$
# 95000 95% (1m 41s) 1.1026 Input: Hanevich, Predicted: adanich$, Target: anevich$
# 100000 100% (1m 47s) 1.0406 Input: Avagimoff, Predicted: bdnimoff$, Target: vagimoff$

# Starting string: Joh
# Predicted next sequence: anov$

# Starting string: Smi
# Predicted next sequence: gon$

# Starting string: Kat
# Predicted next sequence: an$

# Starting string: Mül
# Predicted next sequence: ona$