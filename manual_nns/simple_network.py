import numpy as np 
import matplotlib.pyplot as plt

#activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#derivatives of activation functions for chain rule
def relu_derivative(x):
    return np.where(x > 0, 1, 0) #filtering: return 1 if x>0, else 0

def sigmoid_derivative(x):
    return x * (1 - x)

#forward pass
def forward_pass(x, w1, b1, w2, b2):
    #hidden layer
    z1 = np.dot(x, w1) + b1 
    a1 = relu(z1)

    #output layer 
    z2 = np.dot(a1, w2) + b2 
    a2 = sigmoid(z2)

    return z1, a1, z2, a2

#backward pass
def backward_pass(x, y, z1, a1, z2, a2, w1, w2, b1, b2):
    """
    calculating the gradients, i.e. the derivatives of the weights and biases.
    to get to the latter, we use the chain rule. we have to get through
    everything the weights and biases depend on.
    """
    deriv_z2 = a2 - y  #bc we have log loss + sigmoid
    deriv_a1 = np.dot(deriv_z2, w2.T)
    deriv_z1 = deriv_a1 * relu_derivative(z1) 

    #now the gradients of the weights and biases
    grad_w2 = np.dot(a1.T, deriv_z2)
    grad_b2 = np.sum(deriv_z2, axis=0, keepdims=True)
    grad_w1 = np.dot(x.T, deriv_z1)
    grad_b1 = np.sum(deriv_z1, axis=0, keepdims=True)

    return grad_w1, grad_b1, grad_w2, grad_b2

#updating the params with gradient descent
def update_params(w1, w2, b1, b2, grad_w1, grad_b1, grad_w2, grad_b2, lr):
    w1 = w1 - lr * grad_w1
    b1 = b1 - lr * grad_b1
    w2 = w2 - lr * grad_w2
    b2 = b2 - lr * grad_b2

    return w1, b1, w2, b2 

#calculating loss
def log_loss(y, y_hat):
    eps = 1e-8
    y_hat = np.clip(y_hat, eps, 1 - eps) #prevents nans
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def fit(x, y, w1, w2, b1, b2, lr, epochs):
    #tracking the process so we can plot it
    w1_saver, b1_saver, w2_saver, b2_saver, loss_saver = [], [], [], [], []

    for epoch in range(epochs):
        """
        pipeline in one epoch using our functions:
        forward pass => loss calculaton => saving loss =>
        backward pass => updating params => saving params =>
        check progress every 100 epochs
        """
        z1, a1, z2, a2 = forward_pass(x, w1, b1, w2, b2)

        loss = log_loss(y, a2)
        loss_saver.append(loss)

        grad_w1, grad_b1, grad_w2, grad_b2 = backward_pass(x, y, z1, a1, z2, a2, w1, w2, b1, b2)

        w1, b1, w2, b2 = update_params(w1, w2, b1, b2, grad_w1, grad_b1, grad_w2, grad_b2, lr)

        w1_saver.append(w1)
        b1_saver.append(b1)
        w2_saver.append(w2)
        b2_saver.append(b2)

        if epoch % 100 == 0:
            print(f"epochs: {epoch}, loss: {loss}")

    return w1_saver, b1_saver, w2_saver, b2_saver, loss_saver
        

#testing and plotting
np.random.seed(56)
n_features = 2
n_samples = 200
hidden_neurons = 3
x = np.random.randn(n_samples, n_features)
y = (x[:, 0] + x[:, 1] > 0).astype(int).reshape(-1, 1)
w1 = np.random.randn(n_features, hidden_neurons) * 0.01
b1 = np.zeros((1, hidden_neurons))
w2 = np.random.randn(hidden_neurons, 1) * 0.01
b2 = np.zeros((1, 1))

w1_saver, b1_saver, w2_saver, b2_saver, loss_saver = fit(x, y, w1, w2, b1, b2, 0.01, 1000)

plt.plot(loss_saver, color="olive")
plt.show()