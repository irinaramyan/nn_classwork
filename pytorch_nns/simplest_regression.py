import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px

np.random.seed(0)
torch.manual_seed(0) # fixes nandomness in torch
x = np.arange(100)
x = x.reshape(-1, 1)  # looks like [1], [2]...
print(x)
y = 2 * x.flatten() - 12 # making 1-dim
y_noised = y + np.random.randn(100) # noise added
print(y)

# Normalization 
x_norm = (x - x.mean()) / x.std()
y_norm = (y - y.mean()) / y.std()
y_noised = (y_noised - y_noised.mean()) / y_noised.std()

x = torch.tensor(x_norm, dtype=torch.float32)
y = torch.tensor(y_norm, dtype=torch.float32)
y_noised = torch.tensor(y_noised, dtype=torch.float32)

epochs = 10
class LinearRegressionModel(nn.Module):
    def __init__(self, in_neurons, out_neurons): 
        super().__init__()
        self.linear = nn.Linear(in_neurons, out_neurons) # torch creates weights and biases inside

    def forward(self, x): # forward pass process 
        return self.linear(x).flatten()
    

def training(model, x, y):
    # Loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr = 0.1)
    # parameter = parameter - lr * gradient

    
    loss_saver = []
    for epoch in range(epochs):
        
        # forward pass
        y_pred = model(x)

        # loss calculation
        loss = criterion(y_pred, y)
        loss_saver.append(loss.item())

        # backward pass
        optimizer.zero_grad() # clears the grads
        loss.backward() # computes grads
        optimizer.step() # updates params

    return loss_saver


model = LinearRegressionModel(1, 1)
losses = training(model, x, y)
# plt.plot(losses, color="hotpink", label="clean")

# model = LinearRegressionModel(1, 1)
# plt.plot(training(model, x, y_noised), color="DodgerBlue", label="noisy")
# plt.title("Clean and noisy data losses")
# plt.xlabel("n_epoch")
# plt.ylabel("loss")
# plt.legend()
# plt.show()


plot = px.scatter_3d(
    x=np.arange(epochs),
    y=np.zeros(10),
    z=losses,
    labels={'x':'Epoch', 'y':'Fake dim', 'z':'Loss'},
    title='3D Loss Plot',
    color=np.arange(1, epochs+1),      
    color_continuous_scale='Viridis', 
)

plot.show()