import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# 1. Neural Network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
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

    def forward(self, t, x):
        tx = torch.cat([t, x], dim=1)
        return self.net(tx)

# 2. Loss Function
def pde_residual(u_net, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    u = u_net(t, x)
    
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    f = u_t + u * u_x - (0.01 / np.pi) * u_xx
    return f

# 3. Data Prep
sampler = qmc.LatinHypercube(d=2)
samples = sampler.random(n=10000)
t_f = torch.tensor(samples[:, 0:1], dtype=torch.float32)
x_f = torch.tensor(samples[:, 1:2] * 2 - 1, dtype=torch.float32)

x_init = torch.linspace(-1, 1, 50).view(-1, 1)
t_init = torch.zeros_like(x_init)
u_init = -torch.sin(np.pi * x_init)

t_bound = torch.linspace(0, 1, 25).view(-1, 1)
x_bound_pos = torch.ones_like(t_bound)
x_bound_neg = -torch.ones_like(t_bound)
u_bound = torch.zeros_like(t_bound)

t_u = torch.cat([t_init, t_bound, t_bound], dim=0)
x_u = torch.cat([x_init, x_bound_neg, x_bound_pos], dim=0)
u_train = torch.cat([u_init, u_bound, u_bound], dim=0)

# 4. Training Loop
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

loss_list = []

for epoch in range(5001):
    optimizer.zero_grad()
    
    u_pred = model(t_u, x_u)
    loss_u = mse_loss(u_pred, u_train)
    
    f_pred = pde_residual(model, t_f, x_f)
    loss_f = mse_loss(f_pred, torch.zeros_like(f_pred))
    
    loss = loss_u + loss_f
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.5f}")
    
    loss_list.append(loss.item())

# 5. Visualization (Separate Figures)

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
plt.title("Predicted $u(t, x)$")
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
plt.ylabel("$u(t, x)$")
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