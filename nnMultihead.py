# multihead_nn.py
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# Helpers
# --------------------------
def get_col(df, options, required=True):
    """Return the first existing column among 'options'."""
    for c in options:
        if c in df.columns:
            return df[c].values.astype(float)
    if required:
        raise KeyError(f"None of the columns found: {options}")
    return None

def build_features(df):
    # Original inputs (robust to naming with/without spaces)
    vin = get_col(df, ["Velocity Inlet", "VelocityInlet"])
    xc  = get_col(df, ["X Cell Size", "XCellSize"])
    yz  = get_col(df, ["YZ Cell Size", "YZCellSize"])

    # Physics-inspired engineered features
    A_cs   = yz ** 2                 # cross-sectional area (square cell face)
    Dh     = yz                      # hydraulic diameter proxy
    V_cell = xc * (yz ** 2)          # volume per cell
    AR     = xc / np.clip(yz, 1e-9, None)  # aspect ratio
    rho, mu = 1.225, 1.81e-5         # air properties (kg/m^3, Pa·s)
    Re     = (rho * vin * Dh) / mu   # Reynolds number
    vin_sq = vin ** 2                # velocity squared (captures inertial effects)

    # Optional stabilizers
    inv_xc = 1.0 / np.clip(xc, 1e-9, None)
    inv_yz = 1.0 / np.clip(yz, 1e-9, None)
    log_vin = np.log(np.clip(vin, 1e-9, None))
    log_yz  = np.log(np.clip(yz,  1e-9, None))

    X = np.column_stack([
        vin, xc, yz,
        A_cs, Dh, V_cell, AR, Re, vin_sq,
        inv_xc, inv_yz, log_vin, log_yz
    ])
    return X

def build_targets(df):
    y_cols = [
        ("AvgVelocity", ["AvgVelocity", "Average Velocity", "Avg Velocity"]),
        ("Mass", ["Mass"]),
        ("PressureDrop", ["PressureDrop", "Pressure Drop"]),
        ("Surface Area", ["Surface Area", "SurfaceArea"])
    ]
    ys = []
    names = []
    for name, options in y_cols:
        vals = get_col(df, options)
        ys.append(vals.reshape(-1, 1))
        names.append(name)
    y = np.hstack(ys)
    return y, names

# --------------------------
# Model
# --------------------------
class MultiHeadRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # Shared backbone
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 64),
            # nn.ReLU()
        )
        # Four heads with different capacities
        # AvgVelocity head (medium)
        self.head_avg = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Softplus()  # enforce positivity
        )
        # Mass head (shallow)
        self.head_mass = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # nn.Softplus()
        )
        # PressureDrop head (deeper)
        self.head_dp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Softplus()
        )
        # Surface Area head (shallow/medium)
        self.head_sa = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Softplus()
        )

    def forward(self, x):
        h = self.trunk(x)
        y1 = self.head_avg(h)
        y2 = self.head_mass(h)
        y3 = self.head_dp(h)
        y4 = self.head_sa(h)
        return torch.cat([y1, y2, y3, y4], dim=1)

# --------------------------
# Training / Eval
# --------------------------
def rmse(a, b, axis=0):
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))

def main(args):
    # Load
    df = pd.read_csv(args.csv)

    # Features / Targets
    X = build_features(df)
    y, target_names = build_targets(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # Scale inputs only (keep y in physical units since we enforce positivity)
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test  = x_scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    # Model
    model = MultiHeadRegressor(in_dim=X_train.shape[1])

    # Loss weighting to balance different magnitudes
    # Use inverse variance of each target on the train split
    y_std = np.std(y_train, axis=0, ddof=1)
    y_std[y_std < 1e-9] = 1.0
    inv_var = torch.tensor(1.0 / (y_std ** 2), dtype=torch.float32)  # shape (4,)

    def weighted_mse(pred, target):
        # per-target MSE then weight
        mse = (pred - target) ** 2  # (N,4)
        mse_mean = mse.mean(dim=0)  # (4,)
        return (mse_mean * inv_var).mean()

    # Optionally emphasize PressureDrop a bit more
    if args.boost_dp > 1.0:
        dp_boost = torch.tensor([1.0, 1.0, args.boost_dp, 1.0], dtype=torch.float32)
        def weighted_mse(pred, target):
            mse = (pred - target) ** 2
            mse_mean = mse.mean(dim=0)
            return (mse_mean * inv_var * dp_boost).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.5)

    # Training
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = weighted_mse(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if scheduler is not None and epoch % args.step_lr == 0:
            scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{args.epochs}] Loss: {loss.item():.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()
        y_true = y_test

    rmse_targets = rmse(preds, y_true, axis=0)
    overall = rmse(preds, y_true, axis=None)

    print("\nRMSE per target:")
    for name, r in zip(target_names, rmse_targets):
        print(f"{name}: {r:.4f}")
    print(f"\nOverall RMSE: {overall:.4f}")
    
    # --- Logging results to file ---
    log_path = "experiment_results.log"
    with open(log_path, "a") as f:
        f.write("\n--- MultiHead NN Results ---\n")
        for name, r in zip(target_names, rmse_targets):
            f.write(f"{name}: {r:.4f}\n")
        f.write(f"Overall RMSE: {overall:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="nTop ASME Hackathon Data.csv",
                        help="Path to the dataset CSV")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--step_lr", type=int, default=300, help="StepLR step size (epochs)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--boost_dp", type=float, default=1.3,
                        help=">1.0 to emphasize PressureDrop in the loss")
    args = parser.parse_args()
    main(args)
