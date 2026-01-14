import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np

class VanillaNetwork(nn.Module):

    def __init__(self):
        
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(13, 24),      # (numer of inputs, number of outputs)
            nn.ReLU(),          
            nn.Linear(24, 36),
            nn.ReLU(),
            nn.Linear(36, 1),
        )
    
    def forward(self, x):
        return self.network(x)
    
def train(dataloader, model, loss_fn, optimizer):

    model.train()   # set the model to training mode

    num_batches = len(dataloader)
    train_loss = 0

    for batch, (X, y) in dataloader:

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() # backpropagation
        optimizer.step() # update the parameters
        optimizer.zero_grad() # zero the gradients for the next step

    train_loss /= num_batches

    return train_loss

    print(f"Train Loss: {train_loss/num_batches:.4f} \n")

    
def test(dataloader, model, loss_fn):
    model.eval()

    num_batches = len(dataloader)
    test_loss = 0    

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Test Loss: {test_loss:.4f} \n")

    return test_loss

def load_data(batch_size):
    data = np.loadtxt("")

    features = torch.tensor(data[:, :-1], dtype=torch.float32)     # all columns except the last one are features
    targets = torch.tensor(data[:, -1].reshape(-1, 1), dtype=torch.float32)       # the last column is the target variable

    dataset = TensorDataset(features, targets)

    train_ds, test_ds =random_split(dataset, [0.80, 0.20]) # split the dataset into 80% training and 20% testing

    train_dataloader = DataLoader(train_ds, batch_size=batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)

    return train_dataloader, test_dataloader

if __name__ == "__main__":


    model = VanillaNetwork()
    loss_fn = nn.MSELoss() # mean squared error loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3) # Adam optimizer with learning rate of 0.001
    batch_size = 64

    train_dataloader, test_dataloader = load_data(batch_size)

    epochs = 1000
    train_losses = []
    test_losses = []

    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        test_loss = test(test_dataloader, model, loss_fn)
        train_losses.append(train_loss)
        test_losses.append(test_loss)


    plt.figure()
    plt.plot(range(epochs), train_losses)
    plt.plot(range(epochs), test_losses)
    plt.legend(["Train Loss", "Test Loss"])
    plt.show()