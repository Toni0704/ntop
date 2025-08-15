
# Cell 1 / 3: Train Model 1
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import time

# reproducibility
np.random.seed(0)
torch.manual_seed(0)

# CONFIG
CSV_PATH = "/content/nTop ASME Hackathon Data train csv.csv"   # adjust path if needed
DEVICE = torch.device("cpu")  # change to "cuda" if you have GPU

# load and check
df = pd.read_csv(CSV_PATH)
print("Loaded CSV with shape:", df.shape)

required = ["Velocity Inlet", "X Cell Size", "YZ Cell Size", "AvgVelocity", "Mass", "PressureDrop", "Surface Area"]
assert all(c in df.columns for c in required), f"CSV missing required columns. Need: {required}"

# prepare Model1 data: inputs (x, yz) -> outputs (mass, surface_area)
xc = df["X Cell Size"].values.astype(float)
yz = df["YZ Cell Size"].values.astype(float)
X1 = np.column_stack([xc, yz])          # shape (N,2)
y1 = np.column_stack([df["Mass"].values.astype(float),
                      df["Surface Area"].values.astype(float)])  # (N,2) order: mass, area

# train-test split
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=42)

# scalers
x1_scaler = StandardScaler().fit(X1_tr)
y1_scaler = StandardScaler().fit(y1_tr)
X1_tr_s = x1_scaler.transform(X1_tr)
X1_te_s = x1_scaler.transform(X1_te)
y1_tr_s = y1_scaler.transform(y1_tr)
y1_te_s = y1_scaler.transform(y1_te)

# torch tensors
X1_tr_t = torch.tensor(X1_tr_s, dtype=torch.float32, device=DEVICE)
y1_tr_t = torch.tensor(y1_tr_s, dtype=torch.float32, device=DEVICE)
X1_te_t = torch.tensor(X1_te_s, dtype=torch.float32, device=DEVICE)
y1_te_t = torch.tensor(y1_te_s, dtype=torch.float32, device=DEVICE)

# Model1 architecture (simple)
class Model1(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model1 = Model1(in_dim=2, out_dim=2).to(DEVICE)
opt1 = optim.Adam(model1.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# train
EPOCHS1 = 400
t0 = time.time()
for ep in range(EPOCHS1):
    model1.train()
    opt1.zero_grad()
    preds = model1(X1_tr_t)
    loss = loss_fn(preds, y1_tr_t)
    loss.backward()
    opt1.step()
print(f"Model1 trained in {time.time()-t0:.1f}s")

# evaluate (train & test)
model1.eval()
with torch.no_grad():
    p_tr_s = model1(X1_tr_t).cpu().numpy()
    p_te_s = model1(X1_te_t).cpu().numpy()
p_tr = y1_scaler.inverse_transform(p_tr_s)
p_te = y1_scaler.inverse_transform(p_te_s)

print("\nModel1 performance (Mass, Surface Area):")
for i, name in enumerate(["Mass (g)", "Surface Area (mm^2)"]):
    rmse_tr = np.sqrt(mean_squared_error(y1_tr[:, i], p_tr[:, i]))
    r2_tr = r2_score(y1_tr[:, i], p_tr[:, i])
    rmse_te = np.sqrt(mean_squared_error(y1_te[:, i], p_te[:, i]))
    r2_te = r2_score(y1_te[:, i], p_te[:, i])
    print(f"{name:20s} | Train RMSE {rmse_tr:.3f}  R2 {r2_tr:.4f} | Test RMSE {rmse_te:.3f}  R2 {r2_te:.4f}")

# helper: predict_model1 (numpy-friendly)
def predict_model1(x_mm, yz_mm):
    xin = np.array([[x_mm, yz_mm]], dtype=float)
    xs = x1_scaler.transform(xin)
    xt = torch.tensor(xs, dtype=torch.float32, device=DEVICE)
    model1.eval()
    with torch.no_grad():
        out_s = model1(xt).cpu().numpy()
    out = y1_scaler.inverse_transform(out_s)[0]
    return {"mass": float(out[0]), "surface_area": float(out[1])}

# quick example
print("\nExample Model1 prediction for first row:")
print("GT mass,area:", float(df['Mass'].iloc[0]), float(df['Surface Area'].iloc[0]))
print("Predicted   :", predict_model1(float(xc[0]), float(yz[0])))

# Make predictions for all rows in the original DataFrame using Model 1
predicted_results = []
for index, row in df.iterrows():
    x_mm = row["X Cell Size"]
    yz_mm = row["YZ Cell Size"]
    prediction = predict_model1(x_mm, yz_mm)
    predicted_results.append(prediction)

# Convert the list of dictionaries to a DataFrame
predicted_df = pd.DataFrame(predicted_results)

# Add the predicted columns to the original DataFrame
df['Predicted Mass (g)'] = predicted_df['mass']
df['Predicted Surface Area (mm^2)'] = predicted_df['surface_area']

# Display the updated DataFrame with predicted values
display(df.head(20))

model1.eval()
X1_full_s = x1_scaler.transform(np.column_stack([xc, yz]))
X1_full_t = torch.tensor(X1_full_s, dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    p1_full_s = model1(X1_full_t).cpu().numpy()
p1_full = y1_scaler.inverse_transform(p1_full_s)  # shape (N,2): mass, area
mass_pred = p1_full[:, 0]
area_pred = p1_full[:, 1]

# 2) assemble Model2 inputs (primary inputs + model1 outputs)
vin = df["Velocity Inlet"].values.astype(float)
X2 = np.column_stack([xc, yz, vin, mass_pred, area_pred])   # shape (N,5)
y2 = np.column_stack([df["AvgVelocity"].values.astype(float),
                      df["PressureDrop"].values.astype(float)])  # shape (N,2)

# split
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)

# scalers
x2_scaler = StandardScaler().fit(X2_tr)
y2_scaler = StandardScaler().fit(y2_tr)
X2_tr_s = x2_scaler.transform(X2_tr)
X2_te_s = x2_scaler.transform(X2_te)
y2_tr_s = y2_scaler.transform(y2_tr)
y2_te_s = y2_scaler.transform(y2_te)

# tensors
X2_tr_t = torch.tensor(X2_tr_s, dtype=torch.float32, device=DEVICE)
y2_tr_t = torch.tensor(y2_tr_s, dtype=torch.float32, device=DEVICE)
X2_te_t = torch.tensor(X2_te_s, dtype=torch.float32, device=DEVICE)
y2_te_t = torch.tensor(y2_te_s, dtype=torch.float32, device=DEVICE)

class Model2(nn.Module):
    def __init__(self, in_dim=5, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model2 = Model2(in_dim=5, out_dim=2).to(DEVICE)
opt2 = optim.Adam(model2.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# train Model2
EPOCHS2 = 500
t0 = time.time()
for ep in range(EPOCHS2):
    model2.train()
    opt2.zero_grad()
    preds2 = model2(X2_tr_t)
    loss2 = loss_fn(preds2, y2_tr_t)
    loss2.backward()
    opt2.step()
print(f"Model2 trained in {time.time()-t0:.1f}s")

# evaluate Model2
model2.eval()
with torch.no_grad():
    p2_tr_s = model2(X2_tr_t).cpu().numpy()
    p2_te_s = model2(X2_te_t).cpu().numpy()
p2_tr = y2_scaler.inverse_transform(p2_tr_s)
p2_te = y2_scaler.inverse_transform(p2_te_s)

print("\nModel2 performance (AvgVelocity, PressureDrop):")
for i, name in enumerate(["AvgVelocity (mm/s)", "PressureDrop (Pa)"]):
    rmse_tr = np.sqrt(mean_squared_error(y2_tr[:, i], p2_tr[:, i]))
    r2_tr = r2_score(y2_tr[:, i], p2_tr[:, i])
    rmse_te = np.sqrt(mean_squared_error(y2_te[:, i], p2_te[:, i]))
    r2_te = r2_score(y2_te[:, i], p2_te[:, i])
    print(f"{name:22s} | Train RMSE {rmse_tr:.3f}  R2 {r2_tr:.4f} | Test RMSE {rmse_te:.3f}  R2 {r2_te:.4f}")

# helper: predict_model2 (uses model1 prediction internally)
def predict_model2(x_mm, yz_mm, vin_mms):
    # get model1 predicted mass & area (numpy)
    m1 = predict_model1(x_mm, yz_mm)
    inp = np.array([[x_mm, yz_mm, vin_mms, m1["mass"], m1["surface_area"]]], dtype=float)
    xs = x2_scaler.transform(inp)
    xt = torch.tensor(xs, dtype=torch.float32, device=DEVICE)
    model2.eval()
    with torch.no_grad():
        out_s = model2(xt).cpu().numpy()
    out = y2_scaler.inverse_transform(out_s)[0]
    return {"avg_velocity": float(out[0]), "pressure_drop": float(out[1])}

# quick sanity
print("\nExample Model2 prediction for first row:")
print("GT avg_vel, pd:", float(df['AvgVelocity'].iloc[0]), float(df['PressureDrop'].iloc[0]))
print("Predicted    :", predict_model2(float(xc[0]), float(yz[0]), float(vin[0])))

# Save scaler arrays as torch tensors for use in optimization cell
x1_mean = x1_scaler.mean_; x1_scale = x1_scaler.scale_
y1_mean = y1_scaler.mean_; y1_scale = y1_scaler.scale_
x2_mean = x2_scaler.mean_; x2_scale = x2_scaler.scale_
y2_mean = y2_scaler.mean_; y2_scale = y2_scaler.scale_

import torch
x1_mean_t = torch.tensor(x1_mean, dtype=torch.float32, device=DEVICE)
x1_scale_t = torch.tensor(x1_scale, dtype=torch.float32, device=DEVICE)
y1_mean_t = torch.tensor(y1_mean, dtype=torch.float32, device=DEVICE)
y1_scale_t = torch.tensor(y1_scale, dtype=torch.float32, device=DEVICE)

x2_mean_t = torch.tensor(x2_mean, dtype=torch.float32, device=DEVICE)
x2_scale_t = torch.tensor(x2_scale, dtype=torch.float32, device=DEVICE)
y2_mean_t = torch.tensor(y2_mean, dtype=torch.float32, device=DEVICE)
y2_scale_t = torch.tensor(y2_scale, dtype=torch.float32, device=DEVICE)

# Cell 3 / 3: End-to-end pipeline (model1 -> model2) inside autograd; robust gradient optimizer
# Assumes model1, model2, and scaler tensors exist in memory from Cell 1 & 2.

import time

# constants (from problem)
MAX_MASS_G = 125.0
MAX_PRESSURE_DROP = 8000.0
MIN_AVG_VELOCITY = 520.0
LB_X, UB_X = 10.0, 25.0
LB_YZ, UB_YZ = 10.0, 25.0
LB_V, UB_V = 2500.0, 3500.0

# pipeline: x_t, yz_t, vin_t are torch tensors shape (1,) or (1,1)
def pipeline_torch(x_t, yz_t, vin_t):
    # prepare model1 input and scale
    in1 = torch.cat([x_t.unsqueeze(1), yz_t.unsqueeze(1)], dim=1)  # shape (1,2)
    in1_s = (in1 - x1_mean_t) / x1_scale_t
    out1_s = model1(in1_s)                             # scaled outputs
    out1 = out1_s * y1_scale_t + y1_mean_t            # physical outputs (mass, area)
    mass_t = out1[:, 0].unsqueeze(1)                  # (1,1)
    area_t = out1[:, 1].unsqueeze(1)                  # (1,1)

    # model2 input is [x, yz, vin, mass, area]
    in2 = torch.cat([x_t.unsqueeze(1), yz_t.unsqueeze(1), vin_t.unsqueeze(1), mass_t, area_t], dim=1)  # (1,5)
    in2_s = (in2 - x2_mean_t) / x2_scale_t
    out2_s = model2(in2_s)
    out2 = out2_s * y2_scale_t + y2_mean_t
    avg_vel_t = out2[:, 0].squeeze()
    pd_t = out2[:, 1].squeeze()
    return mass_t.squeeze(), area_t.squeeze(), avg_vel_t.squeeze(), pd_t.squeeze()

# smooth constraint violation (squared hinge)
def violation_tensor(mass_t, pd_t, avg_vel_t):
    v1 = torch.relu(mass_t - MAX_MASS_G)
    v2 = torch.relu(pd_t - MAX_PRESSURE_DROP)
    v3 = torch.relu(MIN_AVG_VELOCITY - avg_vel_t)
    return v1*v1 + v2*v2 + v3*v3

# single-start optimizer (repair + maximize + LBFGS)
def optimize_start(x0, yz0, vin0, steps_repair=200, steps_opt=500, lr=0.02, penalty_init=1e3, penalty_growth=1.02):
    # primal variables (1-d tensors)
    x_var = torch.tensor([x0], dtype=torch.float32, device=DEVICE, requires_grad=True)
    yz_var = torch.tensor([yz0], dtype=torch.float32, device=DEVICE, requires_grad=True)
    vin_var = torch.tensor([vin0], dtype=torch.float32, device=DEVICE, requires_grad=True)

    # Phase A: feasibility repair
    optA = optim.Adam([x_var, yz_var, vin_var], lr=lr)
    pen_w = penalty_init
    for i in range(steps_repair):
        optA.zero_grad()
        mass_t, area_t, avg_vel_t, pd_t = pipeline_torch(x_var, yz_var, vin_var)
        pen = violation_tensor(mass_t, pd_t, avg_vel_t)
        loss = pen_w * pen
        loss.backward()
        optA.step()
        with torch.no_grad():
            x_var.clamp_(LB_X, UB_X); yz_var.clamp_(LB_YZ, UB_YZ); vin_var.clamp_(LB_V, UB_V)
        pen_w *= penalty_growth

    # Phase B: maximize area with penalty
    optB = optim.Adam([x_var, yz_var, vin_var], lr=lr)
    pen_w = max(pen_w, 1e4)
    for i in range(steps_opt):
        optB.zero_grad()
        mass_t, area_t, avg_vel_t, pd_t = pipeline_torch(x_var, yz_var, vin_var)
        pen = violation_tensor(mass_t, pd_t, avg_vel_t)
        loss = -(area_t) + pen_w * pen
        loss.backward()
        optB.step()
        with torch.no_grad():
            x_var.clamp_(LB_X, UB_X); yz_var.clamp_(LB_YZ, UB_YZ); vin_var.clamp_(LB_V, UB_V)

    # LBFGS polish (optional)
    def closure():
        optLB.zero_grad()
        m_t, a_t, av_t, p_t = pipeline_torch(x_var, yz_var, vin_var)
        pen = violation_tensor(m_t, p_t, av_t)
        L = -(a_t) + pen_w * pen
        L.backward()
        return L
    optLB = optim.LBFGS([x_var, yz_var, vin_var], max_iter=40, line_search_fn="strong_wolfe")
    try:
        optLB.step(closure)
    except Exception:
        pass
    with torch.no_grad():
        x_var.clamp_(LB_X, UB_X); yz_var.clamp_(LB_YZ, UB_YZ); vin_var.clamp_(LB_V, UB_V)

    # final evaluate
    m_t, a_t, av_t, p_t = pipeline_torch(x_var.detach(), yz_var.detach(), vin_var.detach())
    out = {"mass": float(m_t.detach().cpu().numpy()),
           "area": float(a_t.detach().cpu().numpy()),
           "avg_vel": float(av_t.detach().cpu().numpy()),
           "pd": float(p_t.detach().cpu().numpy())}
    feasible = (out["mass"] < MAX_MASS_G and out["pd"] < MAX_PRESSURE_DROP and out["avg_vel"] > MIN_AVG_VELOCITY)
    return feasible, float(x_var.item()), float(yz_var.item()), float(vin_var.item()), out

# multi-start wrapper with prints
def multi_start_optimize(n_starts=60, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    best_area = -1e18
    t0 = time.time()
    for i in range(n_starts):
        x0 = float(rng.uniform(LB_X, UB_X))
        yz0 = float(rng.uniform(LB_YZ, UB_YZ))
        vin0 = float(rng.uniform(LB_V, UB_V))
        feasible, xb, yb, vb, out = optimize_start(x0, yz0, vin0,
                                                   steps_repair=180, steps_opt=400, lr=0.002,
                                                   penalty_init=1e6, penalty_growth=1.02)
        print(f"Start {i+1}/{n_starts} -> Feasible: {feasible} | x: {xb:.3f} | yz: {yb:.3f} | vin: {vb:.3f} | area: {out['area']:.3f} | mass: {out['mass']:.3f} | pd: {out['pd']:.1f} | avg_vel: {out['avg_vel']:.2f}")
        if feasible and out["area"] > best_area:
            best_area = out["area"]
            best = (xb, yb, vb, out)
    print(f"Multi-start finished in {time.time()-t0:.1f}s")
    return best

# run
print("\nRunning multi-start gradient-based optimization (Augmented Lagrangian-like)...")
best = multi_start_optimize(n_starts=60, seed=42)

if best:
    xb, yb, vb, out = best
    print("\n=== Best feasible solution ===")
    print(f"Cell Size X (mm): {xb:.6f}")
    print(f"Cell Size Y/Z (mm): {yb:.6f}")
    print(f"Inlet Velocity (mm/s): {vb:.6f}")
    print(f"Surface Area (mm^2): {out['area']:.6f}")
    print(f"Mass (g): {out['mass']:.6f}")
    print(f"Pressure Drop (Pa): {out['pd']:.6f}")
    print(f"Avg Velocity (mm/s): {out['avg_vel']:.6f}")
else:
    print("No feasible solution found. Increase n_starts / steps or penalty strength.")

predict_model2(22.17, 20.55, 3004.14)

predict_model1(22.17, 20.55)

predict_model1(0, 0)

