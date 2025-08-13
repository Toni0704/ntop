import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Load Data ===
df = pd.read_csv("nTop ASME Hackathon Data.csv")  # Replace with your file

# Original inputs
vin = df["Velocity Inlet"].values
xc = df["X Cell Size"].values
yz = df["YZ Cell Size"].values

# Engineered features
A_cs = yz ** 2                             # cross-sectional area
Dh = yz                                    # hydraulic diameter
V_cell = xc * (yz ** 2)                    # volume per cell
AR = xc / yz                               # aspect ratio
rho, mu = 1.225, 1.81e-5                   # air properties
Re = (rho * vin * Dh) / mu                 # Reynolds number
vin_sq = vin ** 2                          # velocity squared

# Stack all features
X = np.column_stack([vin, xc, yz, A_cs, Dh, V_cell, AR, Re, vin_sq])

# Targets
y = df[["AvgVelocity", "Mass", "PressureDrop", "Surface Area"]].values

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_t = torch.tensor(x_scaler.fit_transform(X_train), dtype=torch.float32)
X_test_t = torch.tensor(x_scaler.transform(X_test), dtype=torch.float32)
y_train_t = torch.tensor(y_scaler.fit_transform(y_train), dtype=torch.float32)
y_test_t = torch.tensor(y_scaler.transform(y_test), dtype=torch.float32)

# === NN Model ===
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(X_train_t.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            # nn.ReLU()  # Ensure outputs are always positive
        )
    def forward(self, x):
        return self.net(x)

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Training ===
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/500], Loss: {loss.item():.6f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    preds = model(X_test_t).numpy()
    preds = y_scaler.inverse_transform(preds)
    y_true = y_scaler.inverse_transform(y_test_t.numpy())

rmse_per_target = np.sqrt(np.mean((preds - y_true) ** 2, axis=0))
overall_rmse = np.sqrt(np.mean((preds - y_true) ** 2))

# log_path = "experiment_results.log"
# with open(log_path, "a") as f:
#     f.write("\n--- NN Feature Engineering Results ---\n")
#     for name, val in zip(["AvgVelocity", "Mass", "PressureDrop", "SurfaceArea"], rmse_per_target):
#         f.write(f"{name}: {val:.4f}\n")
#     f.write(f"Overall RMSE: {overall_rmse:.4f}\n")

print("\nRMSE per target:")
for name, val in zip(["AvgVelocity", "Mass", "PressureDrop", "SurfaceArea"], rmse_per_target):
    print(f"{name}: {val:.4f}")
print(f"\nOverall RMSE: {overall_rmse:.4f}")