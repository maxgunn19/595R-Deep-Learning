import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Neural Network with Learnable PDE Parameters
class PINN_Inverse(nn.Module):
    def __init__(self):
        super(PINN_Inverse, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )
        # lambda1 and lambda2 are learnable parameters
        self.lambda1 = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.lambda2 = nn.Parameter(torch.tensor([0.0], requires_grad=True))

    def forward(self, t, x):
        tx = torch.cat([t, x], dim=1)
        return self.net(tx)

# 2. Updated PDE Residual for Inverse Problem
def pde_residual(u_net, t, x, l1, l2):
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    u = u_net(t, x)
    
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    # Governing equation with unknown coefficients
    f = u_t + l1 * u * u_x + l2 * u_xx
    return f

# 3. Data Preparation
data = np.loadtxt("burgers.txt") 


x_data = torch.tensor(data[:, 0:1], dtype=torch.float32)
t_data = torch.tensor(data[:, 1:2], dtype=torch.float32)
u_data = torch.tensor(data[:, 2:3], dtype=torch.float32)

# 4. Training Loop
model = PINN_Inverse()
# Optimizer includes both NN weights and the lambda parameters!!
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

loss_list = []

for epoch in range(5001):
    optimizer.zero_grad()
    
    # Data Loss (pretty much: How well the network fits the observations)
    u_pred = model(t_data, x_data)
    loss_u = mse_loss(u_pred, u_data)
    
    # Physics Loss (pretty much: How well the governing equation is satisfied)
    f_pred = pde_residual(model, t_data, x_data, model.lambda1, model.lambda2)
    loss_f = mse_loss(f_pred, torch.zeros_like(f_pred))
    
    loss = loss_u + loss_f
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.5f}, "
              f"λ1 = {model.lambda1.item():.4f}, λ2 = {model.lambda2.item():.6f}")
    
    loss_list.append(loss.item())

# 5. Visualization 

# --- Figure 1: Heatmap/Contour Plot ---
plt.figure(figsize=(8, 6))
t_flat = np.linspace(0, 1, 100)
x_flat = np.linspace(-1, 1, 200)
T, X = np.meshgrid(t_flat, x_flat)

t_tensor = torch.tensor(T.flatten()[:, None], dtype=torch.float32)
x_tensor = torch.tensor(X.flatten()[:, None], dtype=torch.float32)

with torch.no_grad():
    u_pred_plot = model(t_tensor, x_tensor).reshape(200, 100).numpy()

cp = plt.contourf(T, X, u_pred_plot, levels=50, cmap='rainbow')
plt.colorbar(cp)
plt.title(f"Predicted u(t, x) \nlambda_1 {model.lambda1.item():.3f}, lambda_2  {model.lambda2.item():.5f}")
plt.xlabel("t")
plt.ylabel("x")
plt.show()

# --- Figure 2: Slice at t = 0.75 ---
plt.figure(figsize=(8, 6))
x_slice = torch.linspace(-1, 1, 100).view(-1, 1)
t_slice = torch.ones_like(x_slice) * 0.75
with torch.no_grad():
    u_slice = model(t_slice, x_slice).numpy()

plt.plot(x_slice.numpy(), u_slice, 'r--', linewidth=2, label='PINN Prediction')
plt.title("Slice at t = 0.75")
plt.xlabel("x")
plt.ylabel("u(t, x)")
plt.legend()
plt.grid(True)
plt.show()

# --- Figure 3: Training Loss ---
plt.figure(figsize=(8, 5))
plt.semilogy(loss_list, color='blue')
plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss (Log Scale)")
plt.grid(True, which="both", ls="-")
plt.show()