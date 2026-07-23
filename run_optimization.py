#!/usr/bin/env python3
"""Inventory optimisation — translates joint_opt.m & profit_compare.m.

Loads estimated parameters, constructs profit surfaces, finds optimal
(p*, T*, Q*) for both exact and approximate profit functions, and
produces Figures 6–8 from the paper.
"""

import os
import numpy as np
from scipy.optimize import root

from grey_inventory.optimization import profit, profit_approx

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Load parameters
# ═══════════════════════════════════════════════════════════════════════════════

# true (simulation) parameters
alpha_true = 120.0
beta_true = 10.0
lam_true = 0.05

# estimated parameters
datadir = os.path.join(os.path.dirname(__file__), "output")
data = np.load(os.path.join(datadir, "parameters.npz"))
alpha_est = float(data["alpha_estimate"])
beta_est = float(data["beta_estimate"])
lam_est = float(data["lambda_estimate"])

# economic (cost) parameters
c = 4.0        # unit purchase cost
h = 0.02       # holding cost / unit / day
K = 100.0      # fixed ordering cost

# search intervals
p_sim_lo, p_sim_hi = c, alpha_true / beta_true      # [4, 12]
p_est_lo, p_est_hi = c, alpha_est / beta_est
T_lo, T_hi = 1.0, 14.0

print(f"True params:      α={alpha_true}, β={beta_true}, λ={lam_true}")
print(f"Estimated params: α={alpha_est:.4f}, β={beta_est:.4f}, λ={lam_est:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Gradient functions for exact profit (equation 20)
# ═══════════════════════════════════════════════════════════════════════════════

def profit_grad_exact(vars_, alpha, beta, lam):
    """Returns [∂P/∂p, ∂P/∂T] for the exact profit function (19)."""
    p, T = vars_[0], vars_[1]
    Dp = alpha - beta * p
    exp_lamT = np.exp(-lam * T)

    # ∂P/∂p  — eq. (20a)
    dP_dp = ((alpha - 2 * beta * p) * lam - beta * h) / (lam ** 2 * T) \
            * (1.0 - exp_lamT) \
            - ((alpha - 2 * beta * p) * c - beta * (c * lam + h)) / lam

    # ∂P/∂T  — eq. (20b)
    term = Dp * (p * lam + h)
    dP_dT = (term * T * lam * exp_lamT
             - term * (exp_lamT - 1.0)
             - K * lam ** 2) / (lam ** 2 * T ** 2)

    return np.array([dP_dp, dP_dT])


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Gradient functions for approximate profit (equations 22, 25)
# ═══════════════════════════════════════════════════════════════════════════════

def profit_grad_approx(vars_, alpha, beta, lam):
    """Returns [∂P̃/∂p, ∂P̃/∂T] for the approximate profit function (21).

    ∂P̃/∂T = −(α−βp)(pλ+h)/2  +  K/T²       (22)
    ∂P̃/∂p = (α−2βp) − β·[2c − p − √(2K(pλ+h)/(α−βp))]  — derived from (25)
    """
    p, T = vars_[0], vars_[1]
    Dp = alpha - beta * p
    if Dp <= 0 or p < 0 or T <= 0:
        return np.array([1e6, 1e6])

    # ∂P̃/∂T
    dP_dT = -0.5 * Dp * (p * lam + h) + K / (T ** 2)

    # ∂P̃/∂p — from eq. (25):
    #   (α−βp) − β·[p − c − sqrt(K(pλ+h)/(2(α−βp)))] = 0
    # rearranged as residual
    sqrt_term = np.sqrt(max(K * (p * lam + h) / (2.0 * Dp), 0))
    dP_dp = Dp - beta * (p - c - sqrt_term)

    return np.array([dP_dp, dP_dT])


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Solve for optimal (p*, T*) — exact & approximate, sim & est params
# ═══════════════════════════════════════════════════════════════════════════════

results = {}

# 4a. Solve approximate first (more robust, gives good initial guess)
for label, alpha, beta, lam in [
    ("sim_approx",  alpha_true, beta_true, lam_true),
    ("est_approx",  alpha_est,  beta_est,  lam_est),
]:
    p0 = 0.5 * (c + alpha / beta)
    T0 = 4.0
    sol = root(profit_grad_approx, [p0, T0], args=(alpha, beta, lam),
               method="hybr", options=dict(maxfev=1000))
    if not sol.success:
        print(f"  WARNING: {label} solver did not converge: {sol.message}")

    p_opt = sol.x[0]
    T_opt = sol.x[1]
    Q_opt = (alpha - beta * p_opt) * T_opt
    P_opt = profit_approx(alpha, beta, p_opt, lam, c, h, K, T_opt)
    results[label] = dict(p=p_opt, T=T_opt, Q=Q_opt, profit=P_opt)
    print(f"{label:12s}:  p*={p_opt:.4f},  T*={T_opt:.4f},  "
          f"Q*={Q_opt:.4f},  P*={P_opt:.4f}")

# 4b. Solve exact — use approximate solution as initial guess,
#     directly minimize negative profit (more robust than gradient root-finding)
for label, alpha, beta, lam, approx_key in [
    ("sim_exact",   alpha_true, beta_true, lam_true, "sim_approx"),
    ("est_exact",   alpha_est,  beta_est,  lam_est,  "est_approx"),
]:
    from scipy.optimize import minimize

    p0 = results[approx_key]["p"]
    T0 = results[approx_key]["T"]
    res = minimize(
        lambda x: -profit(alpha, beta, x[0], lam, c, h, K, x[1]),
        [p0, T0],
        bounds=[(c, alpha / beta), (0.5, 14.0)],
        method="L-BFGS-B",
        options=dict(maxiter=2000, ftol=1e-12),
    )
    p_opt, T_opt = res.x
    Q_opt = (alpha - beta * p_opt) * T_opt
    P_opt = profit(alpha, beta, p_opt, lam, c, h, K, T_opt)
    results[label] = dict(p=p_opt, T=T_opt, Q=Q_opt, profit=P_opt)
    print(f"{label:12s}:  p*={p_opt:.4f},  T*={T_opt:.4f},  "
          f"Q*={Q_opt:.4f},  P*={P_opt:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Figures 6–8
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

CM = 1 / 2.54
figdir = os.path.join(os.path.dirname(__file__), "figure")
os.makedirs(figdir, exist_ok=True)

pg_sim = np.linspace(p_sim_lo, p_sim_hi, 50)
pg_est = np.linspace(p_est_lo, p_est_hi, 50)
Tg = np.linspace(T_lo, T_hi, 50)
Pg_sim, Tg_sim = np.meshgrid(pg_sim, Tg)
Pg_est, Tg_est = np.meshgrid(pg_est, Tg)


def _surf(ax, P, T, Z, opt_p, opt_T, opt_P, color, xl, yl, zl):
    """Helper: plot a single 3-D surface + optimum marker."""
    ax.plot_surface(P, T, Z, cmap=cm.viridis, alpha=0.85, antialiased=True)
    ax.plot([opt_p], [opt_T], [opt_P], marker='*', markersize=15,
            markerfacecolor=color, markeredgecolor='k')
    ax.set_xlabel(xl, fontsize=14)
    ax.set_ylabel(yl, fontsize=14)
    ax.set_zlabel(zl, fontsize=14)
    ax.set_zlim(-100, 120)
    ax.tick_params(labelsize=12)


# ---- Fig 6: Exact profit surfaces -------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, subplot_kw={"projection": "3d"}, figsize=(30 * CM, 15 * CM))

Z_sim = profit(alpha_true, beta_true, Pg_sim, lam_true, c, h, K, Tg_sim)
Z_est = profit(alpha_est, beta_est, Pg_est, lam_est, c, h, K, Tg_est)

_surf(ax1, Pg_sim, Tg_sim, Z_sim,
      results["sim_exact"]["p"], results["sim_exact"]["T"],
      results["sim_exact"]["profit"], "lime",
      "Price", "Ordering cycle", "Profit")
_surf(ax2, Pg_est, Tg_est, Z_est,
      results["est_exact"]["p"], results["est_exact"]["T"],
      results["est_exact"]["profit"], "red",
      "Price", "Ordering cycle", "Profit")

fig.tight_layout()
fig.savefig(os.path.join(figdir, "profit_opt.pdf"))
fig.savefig(os.path.join(figdir, "profit_opt.png"), dpi=150)
plt.close(fig)
print("Fig 6 saved → figure/profit_opt.pdf")

# ---- Fig 7: Approximate profit surfaces -------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, subplot_kw={"projection": "3d"}, figsize=(30 * CM, 15 * CM))

Z_sim_a = profit_approx(alpha_true, beta_true, Pg_sim, lam_true, c, h, K, Tg_sim)
Z_est_a = profit_approx(alpha_est, beta_est, Pg_est, lam_est, c, h, K, Tg_est)

_surf(ax1, Pg_sim, Tg_sim, Z_sim_a,
      results["sim_approx"]["p"], results["sim_approx"]["T"],
      results["sim_approx"]["profit"], "lime",
      "Price", "Ordering cycle", "Profit")
_surf(ax2, Pg_est, Tg_est, Z_est_a,
      results["est_approx"]["p"], results["est_approx"]["T"],
      results["est_approx"]["profit"], "red",
      "Price", "Ordering cycle", "Profit")

fig.tight_layout()
fig.savefig(os.path.join(figdir, "profit_appro_opt.pdf"))
fig.savefig(os.path.join(figdir, "profit_appro_opt.png"), dpi=150)
plt.close(fig)
print("Fig 7 saved → figure/profit_appro_opt.pdf")

# ---- Fig 8: Profit surface comparison ---------------------------------------
# Fig 8(a): sim params — exact vs approx
fig, (ax1, ax2) = plt.subplots(
    1, 2, subplot_kw={"projection": "3d"}, figsize=(30 * CM, 15 * CM))

Tg_short = np.linspace(1, 7, 50)
Pg_sim_s, Tg_sim_s = np.meshgrid(pg_sim, Tg_short)

Z_ex_s = profit(alpha_true, beta_true, Pg_sim_s, lam_true, c, h, K, Tg_sim_s)
Z_ap_s = profit_approx(alpha_true, beta_true, Pg_sim_s, lam_true, c, h, K, Tg_sim_s)

ax1.plot_surface(Pg_sim_s, Tg_sim_s, Z_ex_s, cmap=cm.viridis, alpha=0.7)
ax1.plot_surface(Pg_sim_s, Tg_sim_s, Z_ap_s, cmap=cm.plasma, alpha=0.7)
ax1.set_xlabel("Price", fontsize=14)
ax1.set_ylabel("Ordering cycle", fontsize=14)
ax1.set_zlabel("Profit", fontsize=14)
ax1.legend(["Profit surface", "Approx. profit surface"],
           loc="upper right", fontsize=10)
ax1.tick_params(labelsize=12)

# Fig 8(b): est params — exact vs approx
Pg_est_s, Tg_est_s = np.meshgrid(pg_est, Tg_short)
Z_ex_e = profit(alpha_est, beta_est, Pg_est_s, lam_est, c, h, K, Tg_est_s)
Z_ap_e = profit_approx(alpha_est, beta_est, Pg_est_s, lam_est, c, h, K, Tg_est_s)

ax2.plot_surface(Pg_est_s, Tg_est_s, Z_ex_e, cmap=cm.viridis, alpha=0.7)
ax2.plot_surface(Pg_est_s, Tg_est_s, Z_ap_e, cmap=cm.plasma, alpha=0.7)
ax2.set_xlabel("Price", fontsize=14)
ax2.set_ylabel("Ordering cycle", fontsize=14)
ax2.set_zlabel("Profit", fontsize=14)
ax2.legend(["Profit surface", "Approx. profit surface"],
           loc="upper right", fontsize=10)
ax2.tick_params(labelsize=12)

fig.tight_layout()
fig.savefig(os.path.join(figdir, "profit_opt_compare.pdf"))
fig.savefig(os.path.join(figdir, "profit_opt_compare.png"), dpi=150)
plt.close(fig)
print("Fig 8 saved → figure/profit_opt_compare.pdf")

print("\nDone — all figures saved to figure/")
