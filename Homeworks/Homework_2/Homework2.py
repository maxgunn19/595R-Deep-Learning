import numpy as np
import math
import matplotlib.pyplot as plt
import os

# -------- activation functions -------
def relu(z):
    return np.maximum(0, z)

def relu_back(xbar, z):
    # Gradient is xbar if z > 0 else 0
    return xbar * (z > 0) 

identity = lambda z: z

identity_back = lambda xbar, z: xbar
# -------------------------------------------


# ---------- initialization -----------
def initialization(nin, nout):
    # Using a small scale factor to prevent initial saturation
    W = np.random.randn(nout, nin) * 0.01
    b = np.zeros((nout, 1))
    return W, b
# -------------------------------------


# -------- loss functions -----------
def mse(yhat, y):
    # Mean Squared Error: 1/n * sum((yhat - y)^2)
    return np.mean((yhat - y) ** 2)

def mse_back(yhat, y):
    # Gradient of MSE: 2/n * (yhat - y)
    n = yhat.shape[1]
    return 2 * (yhat - y) / n
# -----------------------------------


# ------------- Layer ------------
class Layer:

    def __init__(self, nin, nout, activation=identity):
        self.W, self.b = initialization(nin, nout)
        self.activation = activation

        if activation == relu:
            self.activation_back = relu_back
        if activation == identity:
            self.activation_back = identity_back

        self.Wbar = None
        self.bbar = None

        # initialize cache
        self.cache = {}

    def forward(self, X, train=True):
        # Linear step: Z = WX + b
        Z = np.dot(self.W, X) + self.b

        #Activation step: A = f(Z)
        A = self.activation(Z)

        # save cache
        if train:
            self.cache['X'] = X
            self.cache['Z'] = Z

        return A

    def backward(self, Xnewbar):
        # 1. Backprop through activiation: Zbar = Xbar * f'(Z)
        Zbar = self.activation_back(Xnewbar, self.cache['Z'])

        # 2. Backprop through linear step:
        # Wbar = Zbar * X^T
        # bbar = sum(Zbar) across samples

        X = self.cache['X']
        self.Wbar = np.dot(Zbar, X.T)
        self.bbar = np.sum(Zbar, axis=1, keepdims=True)

        # 3. Compute Xbar for previous layer: Xbar = W^T * Zbar
        Xbar = np.dot(self.W.T, Zbar)

        return Xbar


class Network:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss_func = loss
        # Fix: Initialize the cache dictionary
        self.cache = {} 

        if loss == mse:
            self.loss_back = mse_back

    def forward(self, X, y, train=True):
        yhat = X
        
        # Change 'layers' to 'layer' to avoid shadowing the list name
        for layer in self.layers: 
            yhat = layer.forward(yhat, train=train)

        L = self.loss_func(yhat, y)

        # save cache
        if train:
            self.cache['X'] = X
            self.cache['y'] = y

        return L, yhat

    def backward(self, yhat, y):
        # Initial gradient from loss function
        grad = self.loss_back(yhat, y)

        # Propagate backward through all layers
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


class GradientDescent:

    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, network):
        for layer in network.layers:
            layer.W -= self.alpha * layer.Wbar
            layer.b -= self.alpha * layer.bbar

if __name__ == '__main__':

    # ---------- data preparation ----------------
    # Initialize lists for the numeric data and the string data
    numeric_data = []

    # Read the text file
    data_path = os.path.join(os.path.dirname(__file__), "Data", "auto-mpg.data")

    with open(data_path, 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split()

            # Check if any of the first 8 columns contain '?'
            if '?' in columns[:8]:
                continue  # Skip this line if there's a missing value

            # Convert the first 8 columns to floats and append to numeric_data
            numeric_data.append([float(value) for value in columns[:8]])

    # Convert numeric_data to a numpy array for easier manipulation
    numeric_array = np.array(numeric_data)

    # Shuffle the numeric array and the corresponding string array
    nrows = numeric_array.shape[0]
    indices = np.arange(nrows)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    shuffled_numeric_array = numeric_array[indices]

    # Split into training (80%) and test (20%) sets
    split_index = int(0.8 * nrows)

    train_numeric = shuffled_numeric_array[:split_index]
    test_numeric = shuffled_numeric_array[split_index:]

    # separate inputs/outputs
    Xtrain = train_numeric[:, 1:]
    ytrain = train_numeric[:, 0]

    Xtest = test_numeric[:, 1:]
    ytest = test_numeric[:, 0]

    # normalize
    Xmean = np.mean(Xtrain, axis=0)
    Xstd = np.std(Xtrain, axis=0)
    ymean = np.mean(ytrain)
    ystd = np.std(ytrain)

    Xtrain = (Xtrain - Xmean) / Xstd
    Xtest = (Xtest - Xmean) / Xstd
    ytrain = (ytrain - ymean) / ystd
    ytest = (ytest - ymean) / ystd

    # reshape arrays (opposite order of pytorch, here we have nx x ns).
    # I found that to be more conveient with the way I did the math operations, but feel free to setup
    # however you like.
    Xtrain = Xtrain.T
    Xtest = Xtest.T
    ytrain = np.reshape(ytrain, (1, len(ytrain)))
    ytest = np.reshape(ytest, (1, len(ytest)))

    # ------------------------------------------------------------

    l1 = Layer(7, 16, relu)
    l2 = Layer(16, 8, relu)
    l3 = Layer(8, 1, identity) # Regression output

    layers = [l1, l2, l3]
    network = Network(layers, mse)

    alpha = 0.05
    optimizer = GradientDescent(alpha)

    epochs = 2000

    train_losses = []
    test_losses = []


    for i in range(epochs):
        L_train, yhat_train = network.forward(Xtrain, ytrain, train=True)
        network.backward(yhat_train, ytrain)
        optimizer.step(network)

        # TODO: run test set
        L_test, yhat_test = network.forward(Xtest, ytest, train=False)

        train_losses.append(L_train)
        test_losses.append(L_test)

        if i % 100 == 0:
            print(f"Epoch {i}: Train Loss = {L_train}, Test Loss = {L_test}")


    # --- inference ----
    _, yhat = network.forward(Xtest, ytest, train=False)

    # unnormalize
    yhat = (yhat * ystd) + ymean
    ytest = (ytest * ystd) + ymean

    print("-" * 30)
    print("Final Statistics:")
    print("Average error (mpg):", np.mean(np.abs(yhat - ytest)))

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()


    plt.figure()
    plt.plot(ytest.T, yhat.T, "o")
    plt.plot([10, 45], [10, 45], "--")

    print("avg error (mpg) =", np.mean(np.abs(yhat - ytest)))

    plt.show()