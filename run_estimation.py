#!/usr/bin/env python3
"""Parameter estimation pipeline — translates parameter_estimization.m.

Generates simulation data for 10 replenishment cycles, applies the grey
forecasting approach (AGO → IRLS) to estimate λ, α, β, and produces
Figures 3–5 from the paper.
"""

import os
import numpy as np
from scipy.io import savemat

from grey_inventory.simulation import (
    inventory_level,
    inventory_level_simulation,
)
from grey_inventory.estimation import (
    lambda_initial,
    lambda_to_alpha_beta,
    irls,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Model & simulation settings
# ═══════════════════════════════════════════════════════════════════════════════

# true parameters (ground truth)
alpha_true = 120.0
beta_true = 10.0
lam_true = 0.05

# simulation settings
SEED = 100
RNG = np.random.default_rng(SEED)

m = 10                                          # number of replenishment cycles
Q_vector = 360 + RNG.integers(0, 121, size=m)   # Q_i  ∈ [360, 480]
p_vector = 5.0 + 3.0 * RNG.random(m)            # p_i  ∈ [5, 8]
std_dev = 5.0                                   # demand noise σ_d
time0 = 0.0
delta_t = 1.0                                   # daily resolution

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Generate true & simulated inventory data
# ═══════════════════════════════════════════════════════════════════════════════

time_true, demand_true, level_diff_true, level_true_t0 = [], [], [], []
time_simu, demand_simu, level_diff_simu = [], [], []
time_true_t0, demand_true_t0 = [], []

for i in range(m):
    # --- exact (noise-free) trajectory ---
    t_t, d_t, ld_t, l_t = inventory_level(
        alpha_true, beta_true, p_vector[i], lam_true, time0, delta_t, Q_vector[i])
    time_true.append(t_t)
    time_true_t0.append(np.concatenate([[time0], t_t]))
    demand_true_t0.append(np.concatenate([[alpha_true - beta_true * p_vector[i]], d_t]))
    level_diff_true.append(ld_t)
    level_true_t0.append(np.concatenate([[Q_vector[i]], l_t]))

    # --- noisy simulation ---
    t_s, d_s, ld_s, _ = inventory_level_simulation(
        alpha_true, beta_true, p_vector[i], std_dev, lam_true,
        time0, delta_t, Q_vector[i], rng=RNG)
    time_simu.append(t_s)
    demand_simu.append(d_s)
    level_diff_simu.append(ld_s)

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Accumulated Generating Operator (AGO)
# ═══════════════════════════════════════════════════════════════════════════════

time_simu_t0, level_simu_t0 = [], []
for i in range(m):
    t_i = np.concatenate([[time0], time_simu[i]])
    dt = np.diff(t_i)
    ld_i = level_diff_simu[i]
    # AGO:  L(t_j) = Q + Σ Δt_k · l(t_k)
    L_i = np.concatenate([[Q_vector[i]], Q_vector[i] + np.cumsum(ld_i * dt)])
    time_simu_t0.append(t_i)
    level_simu_t0.append(L_i)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Train / test split & parameter estimation
# ═══════════════════════════════════════════════════════════════════════════════

n_train = int(0.8 * m)  # 8

time_train = time_simu[:n_train]
demand_train = demand_simu[:n_train]
ldiff_train = level_diff_simu[:n_train]
level_train = level_simu_t0[:n_train]
p_train = p_vector[:n_train]

# --- initial λ from inventory regression (linear) ---
lam0, inv_var = lambda_initial(time0, time_train, demand_train,
                               ldiff_train, level_train)
print(f"lambda0 = {lam0:.5f}   (true lambda = {lam_true:.4f})")
print(f"sigma2_I (initial) = {inv_var:.4f}")

# --- initial alpha, beta from demand regression ---
alpha0, beta0, dem_var = lambda_to_alpha_beta(
    time0, time_train, p_train, demand_train, lam0)
print(f"alpha0 = {alpha0:.4f},  beta0 = {beta0:.4f}")
print(f"sigma2_d (initial) = {dem_var:.4f}")

# --- IRLS ---
weight0 = np.array([1.0 / dem_var, 1.0 / inv_var])
lam_est, history = irls(time0, p_train, time_train, demand_train,
                         ldiff_train, level_train, weight0, lam0)

alpha_est, beta_est, _ = lambda_to_alpha_beta(
    time0, time_train, p_train, demand_train, lam_est)

print(f"\nIRLS converged after {len(history)} iterations")
print(f"lambda_hat  = {lam_est:.5f}    (delta = {lam_est - lam0:.5f})")
print(f"alpha_hat  = {alpha_est:.4f}   (true alpha = {alpha_true:.1f})")
print(f"beta_hat   = {beta_est:.4f}    (true beta = {beta_true:.1f})")

# save estimated parameters (compatible with run_optimization.py)
outdir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(outdir, exist_ok=True)
np.savez(os.path.join(outdir, "parameters.npz"),
         alpha_estimate=alpha_est, beta_estimate=beta_est,
         lambda_estimate=lam_est)
# also save as .mat for MATLAB compatibility
savemat(os.path.join(outdir, "parameter.mat"),
        {"alpha_estimate": alpha_est, "beta_estimate": beta_est,
         "lambda_estimate": lam_est})

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Fitted trajectories (for plotting)
# ═══════════════════════════════════════════════════════════════════════════════

time_fit_t0, demand_fit_t0 = [], []
level_fit_t0, level_diff_fit = [], []

for i in range(m):
    t_f, d_f, ld_f, l_f = inventory_level(
        alpha_est, beta_est, p_vector[i], lam_est,
        time0, delta_t, Q_vector[i])
    time_fit_t0.append(np.concatenate([[time0], t_f]))
    demand_fit_t0.append(np.concatenate([[alpha_est - beta_est * p_vector[i]], d_f]))
    level_diff_fit.append(ld_f)
    level_fit_t0.append(np.concatenate([[Q_vector[i]], l_f]))

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Figures 3–5
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CM = 1 / 2.54
figdir = os.path.join(os.path.dirname(__file__), "figure")
os.makedirs(figdir, exist_ok=True)

# --- helper: subplot label  (a), (b), ... ---
def label(i):
    return f"({chr(97 + i)})"


# ---- Fig 3: Demand ----------------------------------------------------------
fig, axes = plt.subplots(2, m // 2, figsize=(40 * CM, 20 * CM))
axes = axes.flatten()
for i in range(m):
    ax = axes[i]
    ax.plot(time_true_t0[i], demand_true_t0[i], "o-", ms=6, lw=1.5,
            label="Standard demand" if i == 2 else "")
    ax.plot(time_simu[i], demand_simu[i], "^-", ms=6, lw=1.5,
            label="Simulated demand" if i == 2 else "")
    ax.plot(time_fit_t0[i], demand_fit_t0[i], "s-", ms=6, lw=1.5,
            label="Fitted demand" if i == 2 else "")
    ax.set_xlabel("Time / Day", fontsize=14)
    ax.set_ylabel("Demand", fontsize=14)
    ax.set_title(f"{label(i)} The {i + 1}th ordering cycle", fontsize=14)
    ax.tick_params(labelsize=12)
    if i == 2:
        ax.legend(loc="lower left", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "simulation_demand.pdf"))
fig.savefig(os.path.join(figdir, "simulation_demand.png"), dpi=150)
plt.close(fig)
print("Fig 3 saved → figure/simulation_demand.pdf")

# ---- Fig 4: Inventory changes -------------------------------------------------
fig, axes = plt.subplots(2, m // 2, figsize=(40 * CM, 20 * CM))
axes = axes.flatten()
for i in range(m):
    ax = axes[i]
    ax.plot(time_true[i], level_diff_true[i], "o-", ms=6, lw=1.5,
            label="Standard inv. change" if i == 2 else "")
    ax.plot(time_simu[i], level_diff_simu[i], "^-", ms=6, lw=1.5,
            label="Simulated inv. change" if i == 2 else "")
    ax.plot(time_fit_t0[i][1:], level_diff_fit[i], "s-", ms=6, lw=1.5,
            label="Fitted inv. change" if i == 2 else "")
    ax.set_xlabel("Time / Day", fontsize=14)
    ax.set_ylabel("Inventory change", fontsize=14)
    ax.set_title(f"{label(i)} The {i + 1}th ordering cycle", fontsize=14)
    ax.tick_params(labelsize=12)
    if i == 2:
        ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "simulation_diff_level.pdf"))
fig.savefig(os.path.join(figdir, "simulation_diff_level.png"), dpi=150)
plt.close(fig)
print("Fig 4 saved → figure/simulation_diff_level.pdf")

# ---- Fig 5: Inventory levels --------------------------------------------------
fig, axes = plt.subplots(2, m // 2, figsize=(40 * CM, 20 * CM))
axes = axes.flatten()
for i in range(m):
    ax = axes[i]
    ax.plot(time_true_t0[i], level_true_t0[i], "o-", ms=6, lw=1.5,
            label="Standard inv. level" if i == 2 else "")
    ax.plot(time_simu_t0[i], level_simu_t0[i], "^-", ms=6, lw=1.5,
            label="Simulated inv. level" if i == 2 else "")
    ax.set_xlabel("Time / Day", fontsize=14)
    ax.set_ylabel("Inventory level", fontsize=14)
    ax.set_title(f"{label(i)} The {i + 1}th ordering cycle", fontsize=14)
    ax.tick_params(labelsize=12)
    if i == 2:
        ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "simulation_level.pdf"))
fig.savefig(os.path.join(figdir, "simulation_level.png"), dpi=150)
plt.close(fig)
print("Fig 5 saved → figure/simulation_level.pdf")

print("\nDone — estimated parameters saved to output/parameters.npz")
