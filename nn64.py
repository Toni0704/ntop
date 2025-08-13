import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ---- 1. Load Data ----
df = pd.read_csv("nTop ASME Hackathon Data.csv")

X = df[["Velocity Inlet", "X Cell Size", "YZ Cell Size"]].values
y = df[["AvgVelocity", "Mass", "PressureDrop", "Surface Area"]].values

# ---- 2. Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 3. Standardize ----
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# ---- 4. Torch Tensors ----
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# ---- 5. Model ----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.layers(x)

model = Net()

# ---- 6. Loss & Optimizer ----
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- 7. Training ----
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# ---- 8. Evaluation ----
model.eval()
with torch.no_grad():
    preds = model(X_test_t).numpy()
    preds = y_scaler.inverse_transform(preds)
    y_true = y_scaler.inverse_transform(y_test_t.numpy())

# ---- 9. RMSE ----
rmse_per_target = np.sqrt(np.mean((preds - y_true)**2, axis=0))
overall_rmse = np.sqrt(np.mean((preds - y_true)**2))


# log_path = "experiment_results.log"
# with open(log_path, "a") as f:
#     f.write("\n--- NN64x64 Results ---\n")
#     for name, rmse in zip(["AvgVelocity", "Mass", "PressureDrop", "Surface Area"], rmse_per_target):
#         f.write(f"{name}: {rmse:.4f}\n")
#     f.write(f"Overall RMSE: {overall_rmse:.4f}\n")

print("\nRMSE per target:")
for name, rmse in zip(["AvgVelocity", "Mass", "PressureDrop", "Surface Area"], rmse_per_target):
    print(f"{name}: {rmse:.4f}")

print(f"\nOverall RMSE: {overall_rmse:.4f}")
