import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchdiffeq import odeint

# Define the ODE function as before
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, hidden_dim)
        )

    def forward(self, t, x):
        return self.net(x)

# Define the ODE block that integrates the ODE function
class ODEBlock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0, 1])):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.t = t  # integration interval

    def forward(self, x):
        # Integrate the ODE starting from initial condition x over time t
        out = odeint(self.odefunc, x, self.t)
        return out[-1]  # Return the final state

# Define the full Neural ODE model for MNIST
class NeuralODEMNIST(nn.Module):
    def __init__(self, hidden_dim=64):
        super(NeuralODEMNIST, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # After two poolings, image size reduces from 28x28 to 7x7
        self.fc_in = nn.Linear(32 * 7 * 7, hidden_dim)
        self.odeblock = ODEBlock(ODEFunc(hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_in(x)
        x = self.odeblock(x)
        x = self.fc_out(x)
        return x

# Set up training parameters
batch_size = 64
learning_rate = 0.001
epochs = 5

# Define transforms for the MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST mean and std
])

# Download and load the MNIST training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralODEMNIST().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(1, epochs+1):
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}")

# Evaluation on test dataset
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # The class with the highest logit is the prediction
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

