# Importing the libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Prepare the data: X[i to i+10] is the input, X[i + 11] is the output
df = pd.read_csv("daily.csv")

# Preprocess the data - Drop NA values in the dataset
df = df.dropna()
y = df['Price'].values
x = np.arange(1, len(y), 1)

print(len(y))

# Normalize the input range between 0 and 1
minm = y.min()
maxm = y.max()
print(minm, maxm)
y = (y - minm) / (maxm - minm)

# Create input-output sequences
Sequence_Length = 10
X = []
Y = []

for i in range(0, 5900):
    list1 = []
    for j in range(i, i + Sequence_Length):
        list1.append(y[j])
    X.append(list1)
    Y.append(y[j + 1])

# Convert from list to array
X = np.array(X)
Y = np.array(Y)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.10, random_state=42, shuffle=False
)

# Define a custom Dataset
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

# Initialize dataset and dataloader
dataset = NGTimeSeries(x_train, y_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=256)

# Create the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc1(torch.relu(output))
        return output

model = LSTMModel()

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 1500
for i in range(epochs):
    for j, data in enumerate(train_loader):
        x_batch = data[0].view(-1, Sequence_Length, 1)
        y_batch = data[1]
        
        y_pred = model(x_batch).reshape(-1)
        loss = criterion(y_pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 50 == 0:
        print(i, "th iteration : ", loss.item())

# Test set predictions
test_set = NGTimeSeries(x_test, y_test)
test_x = test_set[:][0].view(-1, Sequence_Length, 1)
test_y = test_set[:][1]

test_pred = model(test_x).view(-1)

# Plot predicted vs original
plt.plot(test_pred.detach().numpy(), label='Predicted')
plt.plot(test_y.view(-1), label='Original')
plt.legend()
plt.show()

# Undo normalization
y = y * (maxm - minm) + minm
y_pred = test_pred.detach().numpy() * (maxm - minm) + minm

# Final plot
plt.plot(y, label="Original full series")
plt.plot(range(len(y) - len(y_pred), len(y)), y_pred, label="Test predictions")
plt.legend()
plt.show()

# 0 th iteration :  0.027409639209508896
# 50 th iteration :  0.008162334561347961
# 100 th iteration :  0.009358358569443226
# 150 th iteration :  0.0083739859983325
# 200 th iteration :  0.001522127422504127
# 250 th iteration :  0.0009009289788082242
# 300 th iteration :  0.0006944317719899118
# 350 th iteration :  0.00033002166310325265
# 400 th iteration :  0.00021986600768286735
# 450 th iteration :  0.00035527057480067015
# 500 th iteration :  0.0014624276664108038
# 550 th iteration :  0.00030724593671038747
# 600 th iteration :  0.00023937353398650885
# 650 th iteration :  0.00019794148101937026
# 700 th iteration :  0.00013970857253298163
# 750 th iteration :  0.0001183517015306279
# 800 th iteration :  0.00023570407938677818
# 850 th iteration :  0.00019092090951744467
# 900 th iteration :  0.00015466420154552907
# 950 th iteration :  0.00013027017121203244
# 1000 th iteration :  0.00014689732051920146
# 1050 th iteration :  0.00016157205391209573
# 1100 th iteration :  0.001139433472417295
# 1150 th iteration :  0.00018279427604284137
# 1200 th iteration :  8.847579010762274e-05
# 1250 th iteration :  9.150204277830198e-05
# 1300 th iteration :  0.0009815554367378354
# 1350 th iteration :  0.00041881806100718677
# 1400 th iteration :  0.001040607807226479
# 1450 th iteration :  0.00027875660452991724