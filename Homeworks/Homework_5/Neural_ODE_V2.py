import pandas as pd
import torch                # Import the main library
import torch.nn as nn       # Import the Neural Network submodule as 'nn'
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np

HIDDEN_DIM = 128
LR_INITIAL = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0         # Prevents loss explosions
EPOCHS_PER_STAGE = 500
LR_DECAY_FACTOR = 0.5    # Multiplier for LR after each stage
STAGES = [4, 8, 12, 16, 20]


# ==========================================
# 1. LOAD AND COMBINE DATA
# ==========================================

# Check for GPU (Optional, but good practice)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
device = torch.device("cpu")  # For simplicity, we'll use CPU for this homework

train_path = r"sumanthvrao\daily-climate-time-series-data\versions\3\DailyDelhiClimateTrain.csv"
test_path = r"sumanthvrao\daily-climate-time-series-data\versions\3\DailyDelhiClimateTest.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df = pd.concat([df_train, df_test], ignore_index=True)

# ==========================================
# 2. RESAMPLE TO MONTHLY AVERAGES
# ==========================================
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
df_features = df[features]
df_monthly = df_features.resample('ME').mean()

# ==========================================
# 3. NORMALIZE THE DATA
# ==========================================
train_mean = df_monthly.mean()
train_std = df_monthly.std()
df_normalized = (df_monthly - train_mean) / train_std

# ==========================================
# 4. SPLIT AND CONVERT TO TENSORS
# ==========================================
n_train = 20
train_data = df_normalized.iloc[:n_train]
test_data = df_normalized.iloc[n_train:]

# Create time steps
t_all = np.arange(len(df_normalized))
t_train_np = t_all[:n_train]
t_test_np = t_all[n_train:]

# CONVERT TO PYTORCH TENSORS
y_train = torch.tensor(train_data.values, dtype=torch.float64, device=device)
t_train = torch.tensor(t_train_np, dtype=torch.float64, device=device)
y_test = torch.tensor(test_data.values, dtype=torch.float64, device=device)
t_test = torch.tensor(t_test_np, dtype=torch.float64, device=device)

# CHECK SHAPES
# print(f"y_train shape: {y_train.shape}") 
# print(f"t_train shape: {t_train.shape}") 

class NeuralODE(nn.Module):
    def __init__(self, data_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.Tanh(),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, data_dim)
        )
        
        self.double()

    def odefunc(self, t, y): 
        return self.network(y)

    def forward(self, y0, t_steps):
        # SOLVING THE IVP (Initial Value Problem)
        yhat = odeint(self.odefunc, y0, t_steps, method='rk4', rtol=1e-7)  # Runge-Kutta 4th order method
        return yhat
    

# ==========================================
# STEP 3: INCREMENTAL TRAINING LOOP
# ==========================================
model = NeuralODE(data_dim=4, hidden_dim=HIDDEN_DIM).to(device)
model = model.double()

optimizer = torch.optim.Adam(model.parameters(), lr=LR_INITIAL, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda stage: LR_DECAY_FACTOR**stage)

def train_step(y_batch, t_batch, model, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    
    # FORWARD PASS
    pred = model(y_batch[0], t_batch)
    
    # CALCULATE LOSS
    loss = loss_fn(pred, y_batch)
    
    # BACKPROPAGATION
    loss.backward()
    optimizer.step()
    return loss.item()

# SETUP
# Input dim is 4 (temp, humidity, wind, pressure).
model = NeuralODE(data_dim=4, hidden_dim=32)

model = model.double()

# OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# TRAINING STAGES

print("Starting Incremental Training...")

test_loss_history = []

for stage_idx, cutoff in enumerate(STAGES):
    print(f"\n--- Stage {stage_idx+1}: Training on first {cutoff} months ---")
    
    t_sub = t_train[:cutoff]
    y_sub = y_train[:cutoff]
    
    for epoch in range(EPOCHS_PER_STAGE):
        loss = train_step(y_sub, t_sub, model, optimizer, loss_fn)
        test_loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Loss = {loss:.6f}")

    # ==========================================
    # PLOTTING AFTER EACH STAGE
    # ==========================================
    
    '''
    with torch.no_grad():
        # Ask model to predict across the ENTIRE training timeline to see how it extrapolates
        full_pred = model(y_train[0], t_train)
    
    # Convert to numpy for plotting
    t_np = t_train.numpy()
    y_np = y_train.numpy()
    pred_np = full_pred.numpy()
    
    plt.figure(figsize=(10, 6))
    plt.suptitle(f"End of Stage {stage_idx+1} (Trained on first {cutoff} months)")
    
    # Subplot 1: Temperature
    plt.subplot(2, 2, 1)
    plt.plot(t_np, y_np[:, 0], 'o', label="Actual")
    plt.plot(t_np, pred_np[:, 0], '-', label="Predicted")
    plt.axvline(x=cutoff-1, color='r', linestyle=':', label="Cutoff")
    plt.title("Temperature (Normalized)")
    plt.legend()

    # Subplot 2: Humidity
    plt.subplot(2, 2, 2)
    plt.plot(t_np, y_np[:, 1], 'o')
    plt.plot(t_np, pred_np[:, 1], '-')
    plt.axvline(x=cutoff-1, color='r', linestyle=':')
    plt.title("Humidity (Normalized)")

    # Subplot 3: Wind Speed
    plt.subplot(2, 2, 3)
    plt.plot(t_np, y_np[:, 2], 'o')
    plt.plot(t_np, pred_np[:, 2], '-')
    plt.axvline(x=cutoff-1, color='r', linestyle=':')
    plt.title("Wind Speed (Normalized)")

    # Subplot 4: Pressure
    plt.subplot(2, 2, 4)
    plt.plot(t_np, y_np[:, 3], 'o')
    plt.plot(t_np, pred_np[:, 3], '-')
    plt.axvline(x=cutoff-1, color='r', linestyle=':')
    plt.title("Pressure (Normalized)")
    
    plt.tight_layout()
    plt.show()
    '''

# ==========================================
# STEP 4: FINAL FORECASTING (TESTING)
# ==========================================

print("\n--- Final Forecasting on Test Data ---")
with torch.no_grad():
    # We want to predict from t=0 all the way to the end of the TEST set.
    # We create a new time tensor that combines train and test times.
    t_full = torch.cat([t_train, t_test])
    
    # Predict using the initial condition from the very beginning (Month 0)
    forecast = model(y_train[0], t_full)

# Plotting the final result
t_full_np = t_full.numpy()
forecast_np = forecast.numpy()

plt.figure(figsize=(12, 6))
plt.title("Final Forecast: Extrapolating to Test Data")

# Plot Temperature as the main example
plt.plot(t_full_np, forecast_np[:, 0], 'b-', linewidth=2, label="Model Forecast")
# Plot Training Data
plt.plot(t_train.numpy(), y_train.numpy()[:, 0], 'go', label="Training Data")
# Plot Test Data
plt.plot(t_test.numpy(), y_test.numpy()[:, 0], 'rx', label="Test Data (Ground Truth)")

plt.axvline(x=n_train-1, color='k', linestyle='--', label="Train/Test Split")
plt.legend()
plt.xlabel("Month")
plt.ylabel("Temperature (Normalized)")
plt.show()



# ==========================================
# Plot test loss vs epochs log scale in y-axis

plt.figure(figsize=(10, 5))
plt.plot(test_loss_history, 'm-')
plt.title("Test Loss History During Training")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.yscale("log")
plt.grid()
plt.show()
