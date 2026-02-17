# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint # Use adjoint for better memory efficiency
import pinodedata # Import the provided data file
import time

# %%
# 1. Setup Device and Precision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# print name of the GPU if available
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
device = torch.device('cpu')



# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set default dtype to float64 (Double) as recommended for ODE solvers
torch.set_default_dtype(torch.float64)

# 2. Generate Data
# The homework prompt suggests these parameters for the "expanded" range
nt_train = 11
nt_test = 21
t_train = torch.linspace(0.0, 1.0, nt_train).to(device)
t_test = torch.linspace(0.0, 1.0, nt_test).to(device)

ntrain = 600   # User mentioned ~600
ntest = 100    # User mentioned 100
ncol = 10000   # User mentioned ~10,000

# Get data from the provided file
# Note: pinodedata.getdata returns numpy arrays
Xtrain_np, Xtest_np, Xcol_np, fcol_np, Amap = pinodedata.getdata(
    ntrain, ntest, ncol, t_train.cpu().numpy(), t_test.cpu().numpy()
)

# Convert to Tensors and move to GPU
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

    # odeint expects forward to take (t, y)
    def forward(self, t, z):
        return self.net(z)

# Initialize models
input_dim = 128
latent_dim = 2

encoder = Encoder(input_dim, latent_dim).to(device)
decoder = Decoder(latent_dim, input_dim).to(device)
diffeq_solver = LatentODEFunc(latent_dim).to(device)

# We optimize all parameters together
params = list(encoder.parameters()) + list(decoder.parameters()) + list(diffeq_solver.parameters())
optimizer = optim.Adam(params, lr=1e-4) # LR from supplement [cite: 13]

# %%
def get_physics_loss(col_x, col_fx, encoder, diffeq_solver):
    # 1. Pass collocation points through encoder to get z
    # We must enable grad for input col_x to compute d(phi)/dx
    col_x.requires_grad_(True)
    z_col = encoder(col_x)
    
    # 2. Compute Latent Dynamics predicted by ODE net: h(z)
    # This is "z_dot_predicted"
    # We use t=0.0 as a dummy since our ODE is autonomous (time-invariant)
    z_dot_pred = diffeq_solver(0.0, z_col)
    
    # 3. Compute "True" Physics projected to latent space: J_phi * f(x)
    # z_col has shape (Batch, 2). We need derivatives of z0 and z1 w.r.t x independently.
    
    # Gradient of z[:, 0] w.r.t x
    z0 = z_col[:, 0]
    grad_z0 = torch.autograd.grad(
        outputs=z0,
        inputs=col_x,
        grad_outputs=torch.ones_like(z0),
        create_graph=True, # Needed for training
        retain_graph=True
    )[0] # Shape: (Batch, 128)
    
    # Gradient of z[:, 1] w.r.t x
    z1 = z_col[:, 1]
    grad_z1 = torch.autograd.grad(
        outputs=z1,
        inputs=col_x,
        grad_outputs=torch.ones_like(z1),
        create_graph=True,
        retain_graph=True
    )[0] # Shape: (Batch, 128)
    
    # Project physics f(x) onto these gradients (Dot product)
    # row-wise dot product: sum(grad * fx, dim=1)
    z0_dot_physics = torch.sum(grad_z0 * col_fx, dim=1, keepdim=True)
    z1_dot_physics = torch.sum(grad_z1 * col_fx, dim=1, keepdim=True)
    
    # Concatenate to get the full projected physics vector
    z_dot_physics = torch.cat([z0_dot_physics, z1_dot_physics], dim=1)
    
    # 4. Physics Loss: MSE between ODE prediction and Projected Physics
    loss_phy = torch.mean((z_dot_pred - z_dot_physics)**2)
    
    # 5. Collocation Reconstruction Loss (Eq 9 in paper)
    # Ensure encoder/decoder consistency on collocation points
    recon_col = decoder(z_col)
    loss_col_recon = torch.mean((col_x - recon_col)**2)
    
    return loss_phy + loss_col_recon
# %%
nbatches = 10
batch_size_traj = int(ntrain / nbatches)
batch_size_col = int(ncol / nbatches)

optimizer = optim.Adam(params, lr=1e-3) 
epochs = 1000

print(f"Starting training...")

start_time = time.time()

for epoch in range(epochs):
    
    epoch_start = time.time()
    perm_traj = torch.randperm(ntrain)
    perm_col = torch.randperm(ncol)
    
    total_loss = 0
    
    for i in range(nbatches):
        optimizer.zero_grad()
        
        # --- Prepare Batch Data ---
        idx_traj = perm_traj[i * batch_size_traj : (i + 1) * batch_size_traj]
        idx_col = perm_col[i * batch_size_col : (i + 1) * batch_size_col]
        
        batch_x = train_x[idx_traj]
        batch_col_x = col_x[idx_col]
        batch_col_fx = col_fx[idx_col]
        
        # --- 1. Data Loss ---
        # Encode
        z0 = encoder(batch_x[:, 0, :])
        
        # Integrate (The slow part!)
        # We use t_train which has 11 steps.
        z_pred_traj = odeint(diffeq_solver, z0, t_train).permute(1, 0, 2)
        
        # Decode
        x_recon_traj = decoder(z_pred_traj)
        
        # Loss
        loss_data = torch.mean((batch_x - x_recon_traj)**2)
        
        # --- 2. Physics Loss ---
        # This involves autograd, which is also heavy
        loss_physics = get_physics_loss(batch_col_x, batch_col_fx, encoder, diffeq_solver)
        
        # --- 3. Update ---
        loss = loss_data + loss_physics
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print every SINGLE epoch so you know it's alive
    if epoch % 1 == 0:
        duration = time.time() - epoch_start
        print(f"Epoch {epoch} | Loss: {total_loss/nbatches:.5f} | Time: {duration:.2f}s")
        
    # Stop early if it's "good enough" (User goal: MSE < 0.1)
    if epoch % 10 == 0 and epoch > 0:
        with torch.no_grad():
            z0_test = encoder(test_x[:, 0, :])
            z_pred_test = odeint(diffeq_solver, z0_test, t_test).permute(1, 0, 2)
            x_pred_test = decoder(z_pred_test)
            test_mse = torch.mean((test_x - x_pred_test)**2).item()
            print(f"   >>> Test MSE: {test_mse:.5f}")
            
            if test_mse < 0.05: # Stop if we hit the target early
                print("Target MSE reached! Stopping early.")
                break

print("Training Complete.")

# %%
# Visualization
# We use the 'true_encoder' logic from pinodedata.py, but translated to torch
# Z3 = X @ pinv(A).T
# Z = sign(Z3) * abs(Z3)^(1/3)

def get_true_z(X, A):
    # X: (Batch, Time, Dim) or (Batch, Dim)
    # A: (Dim, 2)
    # We need A_pinv: (2, Dim)
    A_pinv = np.linalg.pinv(A)
    A_pinv_t = torch.tensor(A_pinv.T, dtype=torch.float64).to(device)
    
    # Project
    Z3 = torch.matmul(X, A_pinv_t)
    return torch.sign(Z3) * torch.abs(Z3)**(1/3)

# 1. Get Predictions on Test Set
with torch.no_grad():
    z0_test = encoder(test_x[:, 0, :])
    # Integrate over test time
    z_pred_test = odeint(diffeq_solver, z0_test, t_test).permute(1, 0, 2)
    
    # Get True Z for comparison (Using the true encoder function on the inputs)
    z_true_test = get_true_z(test_x, Amap)
    
    # Move to CPU for plotting
    z_pred_np = z_pred_test.cpu().numpy()
    z_true_np = z_true_test.cpu().numpy()

# 2. Plotting
plt.figure(figsize=(12, 5))

# Plot True Latent Space (What it should look like)
plt.subplot(1, 2, 1)
for i in range(min(50, ntest)): # Plot first 50
    plt.plot(z_true_np[i, :, 0], z_true_np[i, :, 1], 'b-', alpha=0.5)
    plt.plot(z_true_np[i, 0, 0], z_true_np[i, 0, 1], 'bo') # Start point
plt.title("True Latent Dynamics (True Encoder)")
plt.xlabel("z1")
plt.ylabel("z2")
plt.grid(True)

# Plot Learned Latent Space
plt.subplot(1, 2, 2)
for i in range(min(50, ntest)):
    plt.plot(z_pred_np[i, :, 0], z_pred_np[i, :, 1], 'r-', alpha=0.5)
    plt.plot(z_pred_np[i, 0, 0], z_pred_np[i, 0, 1], 'ro')
plt.title("Learned Latent Dynamics (Neural ODE)")
plt.xlabel("z1")
plt.ylabel("z2")
plt.grid(True)

plt.tight_layout()
plt.show()

# Final Test MSE
print(f"Final Test MSE: {torch.mean((test_x - decoder(z_pred_test))**2).item():.6f}")