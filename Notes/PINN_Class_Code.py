'''
Class notes for PINN implementation.

Damped Harmonic Oscillator Example (Based on tutorial from Ben Moseley, 2022)
y(t=0) = 1
y'(t=0) = 0
(techincally only 1 boundary condition because both are at t=0)

m (d^2y/dt^2) + mu (dy/dt) + k y = 0
'''

# STEP 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch import nn


class MLP(nn.Module):
    '''
    1 input (t)
    1 output (y)
    4 hidden layers each with a width of 32
    Tanh activation function
    '''
    # STEP 2: Define the Neural Network
    def __init__(self, hlayers, width): # hlayers = number of hidden layers
        super(MLP, self).__init__()

        # Create layers for nn.Sequential
        layers = []
        layers.append(nn.Linear(1, width))
        layers.append(nn.Tanh())
        for _ in range(hlayers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 1))

        self.network = nn.Sequential(*layers)       # * unpacks the list into positional arguments (splatting)

    def forward(self, t): # t is size 1,1
            return self.network(t)
    

def residuals(model, t, params): # t size: (ncol, 1)
         
    # Unpack parameters
    m, mu, k = params
       
    # evaluate model
    y = model(t)       # y size: (ncol, 1)

    # compute derivatives
    dydt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]             # create_graph=True to enable higher order derivatives
    d2ydt2 = torch.autograd.grad(dydt, t, grad_outputs=torch.ones_like(dydt), create_graph=True)[0]

    # compute residuals m (d^2y/dt^2) + mu (dy/dt) + k y = 0
    res = m * d2ydt2 + mu * dydt + k * y
    return res

def boundary(model, tbc): # tbc: (nbc, 1)
    '''
    Enforce boundary conditions:
    y(tbc) = 1
    dydt(tbc) = 0
    '''
    ybc = model(tbc)   # ybc size: (nbc, 1)

    # Compute first derivative at boundary
    dydtbc = torch.autograd.grad(ybc, tbc, grad_outputs=torch.ones_like(ybc), create_graph=True)[0]

    # Compute boundary residuals
    ybc_res = ybc - 1.0
    dydtbc_res = dydtbc - 0.0

    return ybc_res, dydtbc_res

def datapoints():
    ncol = 50    # Number of collocation points
    max_time = 1.0

    tdata = torch.zeros(1,1)  # Placeholder for data points if needed
    tcol = torch.linspace(0, max_time, ncol).reshape(ncol,1)   # Collocation points

    return tdata, tcol

def train(tbc, tcol, params, model, lossfn, optimizer, epochs):
    model.train()
    optimizer.zero_grad()

    bc1, bc2 = boundary(model, tbc)

    lossbc1 = torch.mean(bc1**2)
    lossbc2 = torch.mean(bc2**2)

    rcol = residuals(model, tcol, params)
    loss_col = torch.mean(rcol**2)

    lambda1, lambda2 = 1e-1, 1e-4

    # print(lossbc1.item(), " ", lossbc2.item(), " ", loss_col.item())

    loss = lossbc1 + lambda1 * lossbc2 + lambda2 * loss_col

    loss.backward()
    optimizer.step()

    return loss.item()


# Example usage:
if __name__ == "__main__":
    # Define model
    model = MLP(hlayers=4, width=32)

    # Define parameters
    m = 1.0
    mu = 4.0
    k = 400
    params = (m, mu, k) 

    

