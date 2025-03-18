import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string

# Preparing the dataset
text = "hello world this is a character level rnn model example"
chars = sorted(list(set(text)))  # Unique characters
char_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_char = {i: ch for i, ch in enumerate(chars)}
n_chars = len(chars)

# Create sequences and labels
seq_length = 5  # Length of input sequence
X, Y = [], []
for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    X.append([char_to_index[ch] for ch in seq])
    Y.append(char_to_index[label])

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

def one_hot_encoding(index, size):
    vec = torch.zeros(size)
    vec[index] = 1
    return vec

X_one_hot = torch.stack([torch.stack([one_hot_encoding(i, n_chars) for i in seq]) for seq in X])

# Define the RNN Model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # Take last output
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


# Model Initialization
hidden_size = 128
model = CharRNN(n_chars, hidden_size, n_chars)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 1000
for epoch in range(epochs):
    hidden = model.init_hidden(X_one_hot.shape[0])  # Match batch size
    optimizer.zero_grad()
    output, hidden = model(X_one_hot, hidden)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Function to Predict Next Character
def predict_next_char(seq):
    with torch.no_grad():
        hidden = model.init_hidden(1)
        seq_tensor = torch.stack([one_hot_encoding(char_to_index[ch], n_chars) for ch in seq]).unsqueeze(0)
        output, hidden = model(seq_tensor, hidden)
        predicted_index = torch.argmax(output).item()
        return index_to_char[predicted_index]

# Example Prediction
input_seq = "hello world this i"
next_char = predict_next_char(input_seq)
print(f'Input: {input_seq}, Predicted next char: {next_char}')

# Epoch 0, Loss: 2.8818
# Epoch 100, Loss: 0.0002
# Epoch 200, Loss: 0.0001
# Epoch 300, Loss: 0.0001
# Epoch 400, Loss: 0.0001
# Epoch 500, Loss: 0.0000
# Epoch 600, Loss: 0.0000
# Epoch 700, Loss: 0.0000
# Epoch 800, Loss: 0.0000
# Epoch 900, Loss: 0.0000
# Input: hello world this i, Predicted next char: s