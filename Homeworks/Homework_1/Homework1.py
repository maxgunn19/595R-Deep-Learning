import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import os

class VanillaNetwork(nn.Module):

    def __init__(self):
        
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(7, 64),      # (numer of inputs, number of outputs)
            nn.LeakyReLU(0.1),          
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        return self.network(x)
    
def train(dataloader, model, loss_fn, optimizer):

    model.train()   # set the model to training mode

    num_batches = len(dataloader)
    total_train_loss = 0

    for (X, y) in dataloader:

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() # backpropagation
        optimizer.step() # update the parameters
        optimizer.zero_grad() # zero the gradients for the next step
        total_train_loss += loss.item()

        avg_loss = total_train_loss / num_batches

    print(f"Train Loss: {avg_loss:.4f} \n")

    return avg_loss

    
def test(dataloader, model, loss_fn):
    model.eval()

    num_batches = len(dataloader)
    total_mse = 0
    total_mae = 0   # Calculate Mean Absolute Error as well

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            total_mse += loss_fn(pred, y).item()
            total_mae += torch.abs(pred - y).mean().item()  # Calculate MAE for the batch

    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches

    print(f"Test MSE Loss: {avg_mse:.4f} \n")
    print(f"Average Absolute Error: {avg_mae:.4f} \n")

    return avg_mse

def load_data(batch_size):
    # usecols=range(8) ignores the car names at the end 
    data_path = os.path.join(os.path.dirname(__file__), "Data", "auto-mpg.data-original")
    raw_data = np.genfromtxt(
        data_path, 
        missing_values="NA", 
        filling_values=np.nan, 
        usecols=range(8)
    )

    # Remove rows with NA (like the Citroen on line 11) [cite: 4]
    raw_data = raw_data[~np.isnan(raw_data).any(axis=1)]
    data = raw_data.astype(np.float32)

    # Normalize features (cols 1-7) 
    mean = np.mean(data[:, 1:], axis=0)
    std = np.std(data[:, 1:], axis=0)
    data[:, 1:] = (data[:, 1:] - mean) / (std + 1e-8)

    features = torch.tensor(data[:, 1:], dtype=torch.float32)
    targets = torch.tensor(data[:, 0].reshape(-1, 1), dtype=torch.float32)
    
    dataset = TensorDataset(features, targets)

    # Calculate the exact length for the split
    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds =random_split(dataset, [train_size, test_size]) # split the dataset into 80% training and 20% testing

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) # create a dataloader for the training set with shuffling
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)

    return train_dataloader, test_dataloader

if __name__ == "__main__":


    model = VanillaNetwork()
    loss_fn = nn.MSELoss() # mean squared error loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3) # Adam optimizer with learning rate of 0.001
    batch_size = 

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