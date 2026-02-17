# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torchdiffeq import odeint_adjoint as odeint 
import pinodedata

# ==========================================
# 1. Setup Device and Precision
# ==========================================

# idk y but cpu is faster than gpu 
device = torch.device('cpu') 
print(f"Using device: {device}")

# random seed
torch.manual_seed(42)
np.random.seed(42)

# Set default dtype to float64 (Double)
torch.set_default_dtype(torch.float64)

# ==========================================
# 2. Generate and Prepare Data
# ==========================================
# suggested params
nt_train = 11
nt_test = 21
t_train = torch.linspace(0.0, 1.0, nt_train).to(device)
t_test = torch.linspace(0.0, 1.0, nt_test).to(device)

ntrain = 600   
ntest = 100    
ncol = 10000   

# pinodedata.getdata returns numpy arrays
Xtrain_np, Xtest_np, Xcol_np, fcol_np, Amap = pinodedata.getdata(
    ntrain, ntest, ncol, t_train.cpu().numpy(), t_test.cpu().numpy()
)

# Xtrain shape: (ntrain, nt_train, nx)
train_x = torch.tensor(Xtrain_np, dtype=torch.float64).to(device)
# Xtest shape: (ntest, nt_test, nx)
test_x = torch.tensor(Xtest_np, dtype=torch.float64).to(device)
# Collocation points: (ncol, nx)
col_x = torch.tensor(Xcol_np, dtype=torch.float64).to(device)
# Physics vector field f(x) at collocation points: (ncol, nx)
col_fx = torch.tensor(fcol_np, dtype=torch.float64).to(device)

print(f"Training Data Shape: {train_x.shape}")
print(f"Collocation Data Shape: {col_x.shape}")

# %%
# ==========================================
# 3. Define Model Architecture
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=2):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=2, output_dim=128):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim=2):
        super(LatentODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    # odeint expects forward to take (t, z)
    def forward(self, t, z):
        return self.net(z)

# Initialize models and move to device
input_dim = 128
latent_dim = 2

encoder = Encoder(input_dim, latent_dim).to(device)
decoder = Decoder(latent_dim, input_dim).to(device)
diffeq_solver = LatentODEFunc(latent_dim).to(device)

# optimize parameters together
params = list(encoder.parameters()) + list(decoder.parameters()) + list(diffeq_solver.parameters())
optimizer = optim.Adam(params, lr=1e-4)

# ==========================================
# 4. Define Physics Loss Function
# ==========================================
def get_physics_loss(col_x, col_fx, encoder, diffeq_solver):
    # 1. Pass collocation points through encoder to get z
    col_x.requires_grad_(True)
    z_col = encoder(col_x)
    
    # 2. Compute Latent Dynamics predicted by ODE net: h(z)
    z_dot_pred = diffeq_solver(0.0, z_col)
    
    # 3. Compute "True" Physics projected to latent space: J_phi * f(x)    
    z0 = z_col[:, 0]
    grad_z0 = torch.autograd.grad(
        outputs=z0,
        inputs=col_x,
        grad_outputs=torch.ones_like(z0),
        create_graph=True, # Needed for training
        retain_graph=True
    )[0] # Shape: (Batch, 128)
    
    z1 = z_col[:, 1]
    grad_z1 = torch.autograd.grad(
        outputs=z1,
        inputs=col_x,
        grad_outputs=torch.ones_like(z1),
        create_graph=True,
        retain_graph=True
    )[0] # Shape: (Batch, 128)
    
    z0_dot_physics = torch.sum(grad_z0 * col_fx, dim=1, keepdim=True)
    z1_dot_physics = torch.sum(grad_z1 * col_fx, dim=1, keepdim=True)
    
    # Concatenate to get the full projected physics vector
    z_dot_physics = torch.cat([z0_dot_physics, z1_dot_physics], dim=1)
    
    # 4. Physics Loss: MSE between ODE prediction and Projected Physics
    loss_phy = torch.mean((z_dot_pred - z_dot_physics)**2)
    
    # 5. Collocation Reconstruction Loss (Eq 9 in paper)
    recon_col = decoder(z_col)
    loss_col_recon = torch.mean((col_x - recon_col)**2)
    
    return loss_phy + loss_col_recon

# ==========================================
# 5. Training Loop
# ==========================================
nbatches = 10 # Batching to save memory
batch_size_traj = int(ntrain / nbatches)
batch_size_col = int(ncol / nbatches)
mse_loss_history = []

print(f"Batching: {nbatches} batches.")
print(f"  Trajectories per batch: {batch_size_traj}")
print(f"  Collocation pts per batch: {batch_size_col}")

epochs = 1000 
print(f"Starting training on {device}... (Printing every 1 epoch)")

start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    
    # Shuffle indices
    perm_traj = torch.randperm(ntrain)
    perm_col = torch.randperm(ncol)
    
    total_loss = 0
    
    for i in range(nbatches):
        optimizer.zero_grad()
        
        # --- Prepare Batch Data ---
        idx_traj = perm_traj[i * batch_size_traj : (i + 1) * batch_size_traj]
        idx_col = perm_col[i * batch_size_col : (i + 1) * batch_size_col]
        
        batch_x = train_x[idx_traj]     # (Batch, Time, Dim)
        batch_col_x = col_x[idx_col]    # (Batch_Col, Dim)
        batch_col_fx = col_fx[idx_col]  # (Batch_Col, Dim)
        
        # --- 1. Data Loss (Eq 4 & 5) ---
        # A. Encode initial condition x(t=0) to get z0
        z0 = encoder(batch_x[:, 0, :]) # Shape: (Batch, 2)
        
        # B. Integrate Latent ODE
        # odeint returns (Time, Batch, LatentDim). We want (Batch, Time, LatentDim)
        z_pred_traj = odeint(
                            diffeq_solver, 
                            z0, 
                            t_train, 
                            method='rk4', 
                            options={'step_size': 0.05}  # Tunable parameter
                            ).permute(1, 0, 2)
        
        # C. Decode entire trajectory to get x_hat
        x_recon_traj = decoder(z_pred_traj)
        
        # D. Data Loss (Reconstruction of trajectory)
        loss_data = torch.mean((batch_x - x_recon_traj)**2)
        
        # --- 2. Physics Loss ---
        loss_physics = get_physics_loss(batch_col_x, batch_col_fx, encoder, diffeq_solver)
        
        # --- 3. Backward ---
        loss = loss_data + loss_physics
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Print progress every epoch
    if epoch % 1 == 0:
        duration = time.time() - epoch_start
        print(f"Epoch {epoch:04d} | Loss: {total_loss/nbatches:.6f} | Time: {duration:.2f}s")

    # Store MSE loss history for plotting later
    mse_loss_history.append(total_loss / nbatches)

    # Check Test Error every 10 epochs
    if epoch % 10 == 0 and epoch > 0:
        with torch.no_grad():
            z0_test = encoder(test_x[:, 0, :])
            z_pred_test = odeint(diffeq_solver, z0_test, t_test).permute(1, 0, 2)
            x_pred_test = decoder(z_pred_test)
            test_mse = torch.mean((test_x - x_pred_test)**2).item()
            print(f"   >>> Test MSE: {test_mse:.6f}")
            
            if test_mse < 0.03: # Stop if we hit the target
                print("Target MSE reached! Stopping early.")
                break

print("Training Complete.")

# %%
# ==========================================
# 6. Visualization
# ==========================================
# Helper to get "True" latent space for comparison
def get_true_z(X, A):
    # X: (Batch, Time, Dim)
    # A: (Dim, 2)
    A_pinv = np.linalg.pinv(A)
    A_pinv_t = torch.tensor(A_pinv.T, dtype=torch.float64).to(device)
    # Project
    Z3 = torch.matmul(X, A_pinv_t)
    return torch.sign(Z3) * torch.abs(Z3)**(1/3)

# Get Predictions on Test Set
with torch.no_grad():
    z0_test = encoder(test_x[:, 0, :])
    z_pred_test = odeint(diffeq_solver, z0_test, t_test).permute(1, 0, 2)
    
    # Get True Z for comparison
    z_true_test = get_true_z(test_x, Amap)
    
    # Move to CPU for plotting (if not already there)
    z_pred_np = z_pred_test.cpu().numpy()
    z_true_np = z_true_test.cpu().numpy()

# Define fixed bounds based on the Duffing Oscillator's lobes
x_lim = [-2.0, 2.0]
y_lim = [-2.0, 2.0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Plot True Latent Space
axes[0].set_title("True Latent Dynamics (True Encoder)")
for i in range(min(50, ntest)): 
    axes[0].plot(z_true_np[i, :, 0], z_true_np[i, :, 1], 'b-', alpha=0.5)
    axes[0].plot(z_true_np[i, 0, 0], z_true_np[i, 0, 1], 'bo', markersize=4) 

# Plot Learned Latent Space
axes[1].set_title("Learned Latent Dynamics (Neural ODE)")
for i in range(min(50, ntest)):
    axes[1].plot(z_pred_np[i, :, 0], z_pred_np[i, :, 1], 'r-', alpha=0.5)
    axes[1].plot(z_pred_np[i, 0, 0], z_pred_np[i, 0, 1], 'ro', markersize=4)

# Apply identical formatting to both
for ax in axes:
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal') # Ensures the scale 1:1 for x and y

plt.tight_layout()
plt.show()

# Plot MSE Loss History
plt.figure(figsize=(8, 5))
plt.plot(mse_loss_history, label='MSE Loss (Data + Physics)', color='purple')
plt.title("Training Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.show()

# %%
