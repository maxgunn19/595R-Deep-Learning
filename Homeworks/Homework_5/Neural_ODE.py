import pandas as pd
import torch                # Import the main library
import torch.nn as nn       # Import the Neural Network submodule as 'nn'
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import time
import random
import os


HIDDEN_DIM = 256
LR_INITIAL = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0         # Prevents loss explosions
EPOCHS_PER_STAGE = 500
LR_DECAY_FACTOR = 0.5    # Multiplier for LR after each stage
STAGES = [4, 8, 12, 16, 20]

output_base = "NeuralODE_Results"

existing_folders = [d for d in os.listdir() if os.path.isdir(d) and d.startswith(output_base)]

max_num = 0
for folder in existing_folders:
    try:
        num = int(folder.split()[-1])
        if num > max_num:
            max_num = num
    except ValueError:
        continue

next_num = max_num + 1
output_folder = f"{output_base} {next_num}"

os.makedirs(output_folder, exist_ok=True)
print(f"Created output folder: {output_folder}")


# 1. LOAD AND COMBINE DATA
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
device = torch.device("cpu")  # For simplicity, we'll use CPU for this homework

train_path = r"sumanthvrao\daily-climate-time-series-data\versions\3\DailyDelhiClimateTrain.csv"
test_path = r"sumanthvrao\daily-climate-time-series-data\versions\3\DailyDelhiClimateTest.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df = pd.concat([df_train, df_test], ignore_index=True)

# 2. RESAMPLE TO MONTHLY AVERAGES
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
df_features = df[features]
df_monthly = df_features.resample('ME').mean()


# 3. NORMALIZE THE DATA
train_mean = df_monthly.mean()
train_std = df_monthly.std()
df_normalized = (df_monthly - train_mean) / train_std

# 4. SPLIT AND CONVERT TO TENSORS
n_train = 20
train_data = df_normalized.iloc[:n_train]
test_data = df_normalized.iloc[n_train:]

# Create time steps
t_all = np.arange(len(df_normalized))
t_train_np = t_all[:n_train]
t_test_np = t_all[n_train:]

# 5. CONVERT TO PYTORCH TENSORS
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
        yhat = odeint(self.odefunc, y0, t_steps, method='rk4', rtol=1e-7)  # Runge-Kutta 4th order method
        # yhat = odeint(self.odefunc, y0, t_steps, method='dopri5', rtol=1e-3, atol=1e-3) (slow res 2)
        return yhat
    

# INCREMENTAL TRAINING LOOP
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

def set_seed(seed=42):
    # 1. Set Python random seed
    random.seed(seed)
    
    # 2. Set NumPy random seed
    np.random.seed(seed)
    
    # 3. Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    
    # 4. Ensure deterministic algorithms (slightly slower but consistent)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Set the seed for reproducibility


# SETUP
# Input dim is 4 (temp, humidity, wind, pressure).
model = NeuralODE(data_dim=4, hidden_dim=32)
model = model.double()

# OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# TRAINING STAGES

print("Starting Incremental Training...")

test_loss_history = []

# Lets time the training process
start_time = time.time()

MOD_EPOCHS = 0  # We will increment this after each stage to allow more training as we add more data

for stage_idx, cutoff in enumerate(STAGES):
    print(f"\n--- Stage {stage_idx+1}: Training on first {cutoff} months ---")
    
    t_sub = t_train[:cutoff]
    y_sub = y_train[:cutoff]

    MOD_EPOCHS = int(EPOCHS_PER_STAGE + EPOCHS_PER_STAGE * (stage_idx))  # Increase epochs with each stage

    if stage_idx == 0:
        MOD_EPOCHS = EPOCHS_PER_STAGE  # Start with a base number of epochs for the first stage

    for epoch in range(MOD_EPOCHS):
        loss = train_step(y_sub, t_sub, model, optimizer, loss_fn)
        test_loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Loss = {loss:.6f}")

    # PLOTTING AFTER EACH STAGE
    with torch.no_grad():
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
    save_path = os.path.join(output_folder, f"stage_{stage_idx+1}_results.png")
    plt.savefig(save_path)
    print(f"Saved stage {stage_idx+1} plot to: {save_path}")
    plt.close()  # Closes the figure memory to free up RAM

# FINAL FORECASTING (TESTING)
print("\n--- Final Forecasting on Test Data ---")
with torch.no_grad():
    t_full = torch.cat([t_train, t_test])
    
    # Predict using the initial condition from the very beginning (Month 0)
    # This outputs normalized data
    forecast_norm = model(y_train[0], t_full)

# --- DATA UN-NORMALIZATION ---
# Convert tensors to numpy
t_full_np = t_full.numpy()
forecast_norm_np = forecast_norm.numpy()
t_train_np = t_train.numpy()
y_train_norm_np = y_train.numpy()
t_test_np = t_test.numpy()
y_test_norm_np = y_test.numpy()

# Retrieve the scaling factors used in Step 3
# train_mean and train_std are pandas Series, so we get .values
mean_vals = train_mean.values
std_vals = train_std.values

# Inverse transform formula: Original = (Normalized * Std) + Mean
# Broadcasting ensures this applies to all time steps for the 4 features
forecast_original = (forecast_norm_np * std_vals) + mean_vals
y_train_original = (y_train_norm_np * std_vals) + mean_vals
y_test_original = (y_test_norm_np * std_vals) + mean_vals

# --- PLOTTING ---
# Create a figure with 4 rows and 1 column, sharing the x-axis
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Define plotting styles to match the reference image
# Colors: Blue, Orange, Green, Purple
plot_configs = [
    {"title": "Mean temperature", "ylabel": "Celsius", "color": "#0099ff"}, 
    {"title": "Humidity", "ylabel": "g/m³ of water", "color": "#dd6633"},        
    {"title": "Wind speed", "ylabel": "km/h", "color": "#339933"},      
    {"title": "Mean pressure", "ylabel": "hPa", "color": "#cc66cc"}     
]

for i, ax in enumerate(axes):
    config = plot_configs[i]
    color = config["color"]
    
    # 1. Plot Model Fit (Training Section) - Solid Line
    # We slice the forecast to the length of training data
    ax.plot(t_train_np, forecast_original[:len(t_train_np), i], 
            color=color, linestyle='-', linewidth=2, label="Fit")
            
    # 2. Plot Model Forecast (Test Section) - Dashed Line
    # We slice from the end of training to the end of the data
    ax.plot(t_test_np, forecast_original[len(t_train_np):, i], 
            color=color, linestyle='--', linewidth=2, label="Forecast")

    # 3. Plot Observations (Training) - Dots with black outline
    ax.scatter(t_train_np, y_train_original[:, i], 
               color=color, edgecolors='black', s=50, zorder=3, label="Observations")

    # 4. Plot Observations (Test) - Dots with black outline
    ax.scatter(t_test_np, y_test_original[:, i], 
               color=color, edgecolors='black', s=50, zorder=3)

    # Styling formatting
    ax.set_title(config["title"], fontsize=14, pad=10)
    ax.set_ylabel(config["ylabel"], fontsize=10)
    
    # Only put "Time" label on the bottom plot
    if i == len(axes) - 1:
        ax.set_xlabel("Time (Months)", fontsize=10)
    
    # Remove top and right borders (spines) for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add light grid
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)

legend_elements = [
    Line2D([0], [0], color='gray', lw=2, label='Fit'),
    Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Forecast'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label='Observations')
]
# Place legend in the plot area (upper right)
axes[-1].legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, edgecolor='black')

plt.tight_layout()

final_path = os.path.join(output_folder, "final_forecast_unnormalized.png")
plt.savefig(final_path)
print(f"Saved final forecast to {final_path}")

plt.close()

# Plot loss history with a log scale
plt.figure(figsize=(8, 5))
plt.plot(test_loss_history, color='blue')
plt.yscale('log')
plt.title("Training Loss History (Log Scale)", fontsize=14)
plt.xlabel("Epochs", fontsize=10)
plt.ylabel("MSE Loss (Log Scale)", fontsize=10)
plt.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
loss_path = os.path.join(output_folder, "training_loss_history.png")
plt.savefig(loss_path)
print(f"Saved training loss history to {loss_path}")
plt.close()