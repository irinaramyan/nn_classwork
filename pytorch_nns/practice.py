import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
import torch.optim as optim 
from sklearn.model_selection import train_test_split

# creating synthetic data
np.random.seed(0)
torch.manual_seed(3)
x = (np.arange(-300, 300))
y = (x + np.random.randn(600)*6)**2 + 5 # adding noise

# normalization 
# x = x / max(x)
# print(x)
x = (x / max(x)).reshape(-1, 1)
y = y / max(y) # normalized vals
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# visualizing
# plt.scatter(x, y, color="dodgerblue")
# plt.title("Data Visualization")
# plt.show()

# converting data to tensors 
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# nn creation 
class Regression(nn.Module):
    def __init__(self, in_neurons, out_neurons):
        super().__init__()
        self.linear = nn.Linear(in_neurons, out_neurons)
        self.relu = nn.ReLU()
        self.linear_h1 = nn.Linear(out_neurons, 1)
        self.linear_h2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.linear(x) # .flatten() # input layer
        x = self.relu(x) # relu
        x = self.linear_h1(x) # hidden layer
        # x = self.relu(x) # relu
        # x = self.linear_h2(x) # output layer
        return x 

# creating a model with 1 feature and 3 hidden neurons
model = Regression(1, 5)

# loss 
criterion = nn.MSELoss()

# optimization
optimizer = optim.SGD(model.parameters(), lr=0.01)

# training the nn 
def training_loop(x, y, model):
    epochs = 100
    loss_saver = []
    for epoch in range(epochs):

        # forward pass 
        y_pred = model(x)

        # loss 
        loss = criterion(y_pred, y)
        loss_saver.append(loss.item())

        # backward pass 
        optimizer.zero_grad() # clearing the grads
        loss.backward() # calculating grads 
        optimizer.step() # updating the weights 

        if epoch > 2 and abs(loss_saver[-1] - loss_saver[-2]) < 1e-5:
            print(f"Stopped at epoch {epoch}, Loss: {loss}")
            return loss_saver, epoch
    
    return loss_saver, epochs

losses, epoch = training_loop(x_train, y_train, model)

with torch.no_grad():
    y_pred = model(x_test)
    test_loss = criterion(y_pred, y_test)

# plt.scatter(range(len(losses)), losses, c=losses)
# plt.show()

plt.scatter(y_test, y_pred)
plt.scatter(x_test, y_test, color="dodgerblue")
plt.show()

# print(y_pred)