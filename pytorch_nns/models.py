import torch
import torch.nn as nn
from torchsummary import summary

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten() # Flatten the input image
        self.fc1 = nn.Linear(28 * 28, 128) # First hidden layer (input: 28*28, output: 128)
        self.relu1 = nn.ReLU() # ReLU activation
        self.fc2 = nn.Linear(128, 64) # Second hidden layer (input: 128, output: 64)
        self.relu2 = nn.ReLU() # ReLU activation
        self.fc3 = nn.Linear(64, 10) # Output layer (input: 64, output: 10 for classification)
        self.softmax = nn.Softmax(dim= 1) # Softmax activation for output layer

    def forward(self, x):
        x = self.flatten(x) # Flatten the input
        x = self.relu1(self.fc1(x)) # First hidden layer with ReLU
        x = self.relu2(self.fc2(x)) # Second hidden layer with ReLU
        x = self.fc3(x) # Output layer
        x = self.softmax(x) # Apply Softmax to output
        return x
    
if __name__ == "__main__":
    model = SimpleNN()
    summary(model, input_size=(1, 28, 28))