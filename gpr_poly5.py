import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize

# --------------------------
# 1. Load and split data
# --------------------------
output_col = "PressureDrop"  # <-- set your output column name here
input_cols = ["Velocity Inlet", "X Cell Size", "YZ Cell Size"]

df = pd.read_csv("nTop ASME Hackathon Data.csv")  # <-- replace with your file name
X = df[input_cols].values
y = df[output_col].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 2. Standardize
# --------------------------
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

X_train_s = scaler_X.transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.transform(y_train).ravel()
y_test_s = scaler_y.transform(y_test).ravel()

# --------------------------
# 3. Polynomial physics model (degree 5)
# --------------------------
poly = PolynomialFeatures(degree=5, include_bias=True)
X_poly_train = poly.fit_transform(X_train_s)

# Initial coefficients from least squares
linreg = LinearRegression().fit(X_poly_train, y_train_s)
theta_init = np.concatenate(([linreg.intercept_], linreg.coef_[1:]))

# Helper: physics model prediction
def physics_model(theta, X_input):
    X_poly = poly.transform(X_input)
    return X_poly @ theta

# --------------------------
# 4. Negative log marginal likelihood wrapper
# --------------------------
def nll_all_params(params):
    theta_phys = params[:X_poly_train.shape[1]]
    log_l, log_sf, log_sn = params[X_poly_train.shape[1]:]

    y_phys = physics_model(theta_phys, X_train_s)
    resid = y_train_s - y_phys

    l = np.exp(log_l)
    sf2 = np.exp(log_sf)**2
    sn2 = np.exp(log_sn)**2
    kernel = C(sf2) * RBF(length_scale=l) + WhiteKernel(noise_level=sn2)

    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
    gp.fit(X_train_s, resid)
    return -gp.log_marginal_likelihood()

# --------------------------
# 5. Optimize all params jointly
# --------------------------
l0, sf0, sn0 = 1.0, 1.0, 0.1
init_params = np.concatenate([theta_init, [np.log(l0), np.log(sf0), np.log(sn0)]])

res = minimize(nll_all_params, init_params, method='L-BFGS-B')
opt_params = res.x

theta_opt = opt_params[:X_poly_train.shape[1]]
log_l, log_sf, log_sn = opt_params[X_poly_train.shape[1]:]
l_opt, sf_opt, sn_opt = np.exp(log_l), np.exp(log_sf), np.exp(log_sn)

print("Optimized physics coefficients:", theta_opt)
print(f"Optimized GP hyperparameters: length_scale={l_opt:.3f}, sigma_f={sf_opt:.3f}, sigma_n={sn_opt:.3f}")

# --------------------------
# 6. Train final GP on residuals
# --------------------------
y_phys_train = physics_model(theta_opt, X_train_s)
resid_train = y_train_s - y_phys_train
kernel_opt = C(sf_opt**2) * RBF(length_scale=l_opt) + WhiteKernel(noise_level=sn_opt**2)
gp_final = GaussianProcessRegressor(kernel=kernel_opt, optimizer=None, normalize_y=False)
gp_final.fit(X_train_s, resid_train)

# --------------------------
# 7. Evaluate on test set
# --------------------------
y_phys_test = physics_model(theta_opt, X_test_s)
y_resid_pred, _ = gp_final.predict(X_test_s, return_std=True)
y_pred_s = y_phys_test + y_resid_pred

# Inverse transform to original scale
y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
y_test_orig = y_test.ravel()

r2 = r2_score(y_test_orig, y_pred)
mse = mean_squared_error(y_test_orig, y_pred)
print(f"Test R²: {r2:.4f}")
print(f"Test MSE: {mse:.4f}")

# --------------------------
# 8. Plot for each input variable
# --------------------------
n_grid = 300
mean_X = np.mean(X_train_s, axis=0)

for dim, label in enumerate(input_cols):
    X_plot = np.tile(mean_X, (n_grid, 1))
    grid = np.linspace(np.min(X_train_s[:, dim]), np.max(X_train_s[:, dim]), n_grid)
    X_plot[:, dim] = grid

    y_phys_plot = physics_model(theta_opt, X_plot)
    y_resid_mean, y_resid_std = gp_final.predict(X_plot, return_std=True)
    y_total_mean = y_phys_plot + y_resid_mean
    y_total_std = y_resid_std

    y_total_mean_orig = scaler_y.inverse_transform(y_total_mean.reshape(-1, 1)).ravel()
    y_total_upper = scaler_y.inverse_transform((y_total_mean + 2*y_total_std).reshape(-1, 1)).ravel()
    y_total_lower = scaler_y.inverse_transform((y_total_mean - 2*y_total_std).reshape(-1, 1)).ravel()
    grid_orig = scaler_X.inverse_transform(X_plot)[:, dim]

    plt.figure(figsize=(8, 5))
    plt.plot(grid_orig, y_total_mean_orig, 'b-', label='Physics + GP Prediction')
    plt.fill_between(grid_orig, y_total_lower, y_total_upper, color='b', alpha=0.2, label='Uncertainty (±2σ)')
    plt.scatter(X_train[:, dim], y_train, c='k', s=20, label='Train data', alpha=0.6)
    plt.xlabel(label)
    plt.ylabel(output_col)
    plt.title(f"GP with Physics Mean - {label} vs {output_col}")
    plt.legend()
    plt.show()
