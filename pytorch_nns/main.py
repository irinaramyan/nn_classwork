import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from models import SimpleNN
import torch.nn as nn
import torch.optim as optim

# Define a transformation to normalize the images to be between 0 and 1
transform = transforms.Compose([
transforms.ToTensor(), # Convert the image to a PyTorch tensor
transforms.Normalize((0.5,), (0.5,)) # Normalize the pixel values to be between 0 and 1
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoader for training and testing datasets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Print the dimensions of a single data sample
sample_image, sample_label = train_dataset[0]
#print("Dimensions of a single data sample:", sample_image.shape)

model = SimpleNN()
# loss
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train_epoch(train_loader, model):
    avg_loss = []
    for x, y in train_loader:
        # forward pass
        y_pred = model(x)

        # loss
        loss = criterion(y_pred, y)
        avg_loss.append(loss.item())

        # backward pass 
        optimizer.zero_grad() # clears the grads 
        loss.backward() # calculates the grads 
        optimizer.step() # updates the params
    return np.mean(avg_loss)

def train_function(train_loader, model, epochs):
    loss_saver = []
    for epoch in range(epochs):
        loss = train_epoch(train_loader, model)
        loss_saver.append(loss)
        print(f"Loss for epoch {epoch}: {loss}" )

    return loss_saver

saver = train_function(train_loader, model, 100)

plt.plot(saver, color="PaleTurquoise")
plt.title("Loss per Epoch")
plt.xlabel("n_epoch")
plt.ylabel("loss")
plt.show()