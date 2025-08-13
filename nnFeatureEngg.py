import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


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

from sklearn.model_selection import KFold

# === K-Fold Cross Validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
r2_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1} ---")

    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale per fold
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_t = torch.tensor(x_scaler.fit_transform(X_train), dtype=torch.float32)
    X_test_t = torch.tensor(x_scaler.transform(X_test), dtype=torch.float32)
    y_train_t = torch.tensor(y_scaler.fit_transform(y_train), dtype=torch.float32)
    y_test_t = torch.tensor(y_scaler.transform(y_test), dtype=torch.float32)

    # Define model
    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(300):  # fewer epochs per fold for speed
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).numpy()
        preds = y_scaler.inverse_transform(preds)
        y_true = y_scaler.inverse_transform(y_test_t.numpy())

    fold_rmse = np.sqrt(np.mean((preds - y_true) ** 2))
    fold_r2 = r2_score(y_true, preds, multioutput='uniform_average')

    rmse_scores.append(fold_rmse)
    r2_scores.append(fold_r2)

    print(f"Fold RMSE: {fold_rmse:.4f} | Fold R²: {fold_r2:.4f}")

print("\n=== Final CV Results ===")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")


# log_path = "experiment_results.log"
# with open(log_path, "a") as f:
#     f.write("\n--- NN Feature Engineering Results ---\n")
#     for name, val in zip(["AvgVelocity", "Mass", "PressureDrop", "SurfaceArea"], rmse_per_target):
#         f.write(f"{name}: {val:.4f}\n")
#     f.write(f"Overall RMSE: {overall_rmse:.4f}\n")

# print("\nRMSE per target:")
# for name, val in zip(["AvgVelocity", "Mass", "PressureDrop", "SurfaceArea"], rmse_per_target):
#     print(f"{name}: {val:.4f}")
# print(f"\nOverall RMSE: {overall_rmse:.4f}")

# print("\nR2 per target:")
# for name, val in zip(["AvgVelocity", "Mass", "PressureDrop", "SurfaceArea"], r2_per_target):
#     print(f"{name}: {val:.4f}")
# print(f"\nOverall R2: {overall_r2:.4f}")