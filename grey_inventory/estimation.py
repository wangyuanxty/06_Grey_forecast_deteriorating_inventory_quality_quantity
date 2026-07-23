"""Grey forecasting parameter estimation — the core IRLS pipeline.

Translations of:
  lambda_initial.m       → lambda_initial()
  lambda2alphabeta.m     → lambda_to_alpha_beta()
  lambda_inventoryres.m  → lambda_inventory_residual_var()
  lambda_residual.m      → lambda_residual()
  IRLS.m                 → irls()
"""

import numpy as np
from scipy.optimize import least_squares


# ── lambda_initial.m ─────────────────────────────────────────────────────────

def lambda_initial(time0, time_train, demand_train, level_diff_train,
                   level_train):
    """Initial estimate of λ from the inventory regression equation (linear).

    Equation (13a):  l + d = −λ · ½(Lⱼ + Lⱼ₋₁) · Δt

    Parameters
    ----------
    time0           : float               Order-arrival time.
    time_train      : list of 1-D arrays   Sampling times per cycle.
    demand_train    : list of 1-D arrays   Observed demands per cycle.
    level_diff_train: list of 1-D arrays   Observed inventory changes per cycle.
    level_train     : list of 1-D arrays   AGO-reconstructed inventory levels
                                            (includes initial Q at index 0).

    Returns
    -------
    lambda_0 : float   Initial estimate of λ.
    res_var  : float   Residual variance σ̂²_I.
    """
    H_inv, Y_inv = [], []

    for i in range(len(time_train)):
        time_i = np.concatenate([[time0], time_train[i]])
        dt = np.diff(time_i)
        demand_i = demand_train[i]
        level_diff_i = level_diff_train[i]
        level_i = level_train[i]   # includes Q at [0]

        # regressor: −½ (Lⱼ + Lⱼ₋₁) · Δt
        inv_i = -0.5 * (level_i[1:] + level_i[:-1]) * dt
        H_inv.append(inv_i)
        Y_inv.append(level_diff_i + demand_i)

    H_inv = np.concatenate(H_inv)
    Y_inv = np.concatenate(Y_inv)

    # ordinary least squares:  λ̂ = (HᵀH)⁻¹ HᵀY
    lam0, *_ = np.linalg.lstsq(H_inv[:, None], Y_inv, rcond=None)
    lam0 = lam0[0]

    residual = Y_inv - H_inv * lam0
    res_var = np.sum(residual ** 2) / (len(residual) - 1)

    return lam0, res_var


# ── lambda2alphabeta.m ───────────────────────────────────────────────────────

def lambda_to_alpha_beta(time0, time_train, p_vector_train, demand_train,
                         lam):
    """Estimate α, β from the demand regression equation for a *fixed* λ.

    Equation (13b):
      dⱼ = (α − βp) / λ · [e^{−λ(tⱼ₋₁−t₀)} − e^{−λ(tⱼ−t₀)}]

    Parameters
    ----------
    time0         : float               Order-arrival time.
    time_train    : list of 1-D arrays   Sampling times per cycle.
    p_vector_train: (m,) array           Price for each cycle.
    demand_train  : list of 1-D arrays   Observed demands per cycle.
    lam           : float               Current λ estimate.

    Returns
    -------
    alpha    : float
    beta     : float
    res_var  : float   Residual variance σ̂²_d.
    """
    E_vec, P_vec, Y_dem = [], [], []

    for i in range(len(time_train)):
        time_i = np.concatenate([[time0], time_train[i]])
        demand_i = demand_train[i]
        t_j = time_i[1:]       # t_{i,j}
        t_j1 = time_i[:-1]     # t_{i,j-1}

        Ei = (1.0 / lam) * (np.exp(-lam * (t_j1 - time0))
                             - np.exp(-lam * (t_j - time0)))
        Pi = -p_vector_train[i] * Ei

        E_vec.append(Ei)
        P_vec.append(Pi)
        Y_dem.append(demand_i)

    E_vec = np.concatenate(E_vec)
    P_vec = np.concatenate(P_vec)
    Y_dem = np.concatenate(Y_dem)

    H = np.column_stack([E_vec, P_vec])
    est, *_ = np.linalg.lstsq(H, Y_dem, rcond=None)
    alpha, beta = est[0], est[1]

    residual = Y_dem - H @ est
    res_var = np.sum(residual ** 2) / (len(residual) - 1)

    return alpha, beta, res_var


# ── lambda_inventoryres.m ────────────────────────────────────────────────────

def lambda_inventory_residual_var(time0, time_train, demand_train,
                                  level_diff_train, level_train, lam):
    """Residual variance of the inventory regression equation given λ.

    Identical regressor construction to lambda_initial, but uses the
    *current* λ to compute residuals.

    Returns
    -------
    res_var : float   σ̂²_I.
    """
    H_inv, Y_inv = [], []

    for i in range(len(time_train)):
        time_i = np.concatenate([[time0], time_train[i]])
        dt = np.diff(time_i)
        demand_i = demand_train[i]
        level_diff_i = level_diff_train[i]
        level_i = level_train[i]

        inv_i = -0.5 * (level_i[1:] + level_i[:-1]) * dt
        H_inv.append(inv_i)
        Y_inv.append(level_diff_i + demand_i)

    H_inv = np.concatenate(H_inv)
    Y_inv = np.concatenate(Y_inv)

    residual = Y_inv - H_inv * lam
    res_var = np.sum(residual ** 2) / (len(residual) - 1)
    return res_var


# ── lambda_residual.m ────────────────────────────────────────────────────────

def lambda_residual(lam, time0, p_vector_train, time_train, demand_train,
                    level_diff_train, level_train, weight):
    """Weighted residual vector for the joint regression model.

    This is the objective function minimised by `least_squares(…, method='lm')`.

    Parameters
    ----------
    lam    : float          Current λ (scalar — the optimisation variable).
    weight : (2,) array     [1/σ̂²_d, 1/σ̂²_I] — inverse variances.

    Returns
    -------
    residual : (N_d + N_I,) ndarray  Concatenated weighted residuals.
    """
    E_vec, P_vec, Y_dem = [], [], []
    H_inv, Y_inv = [], []

    for i in range(len(time_train)):
        time_i = np.concatenate([[time0], time_train[i]])
        dt = np.diff(time_i)
        demand_i = demand_train[i]
        level_diff_i = level_diff_train[i]
        level_i = level_train[i]
        t_j = time_i[1:]
        t_j1 = time_i[:-1]

        # --- demand equation (13b) ---
        Ei = (1.0 / lam) * (np.exp(-lam * (t_j1 - time0))
                             - np.exp(-lam * (t_j - time0)))
        Pi = -p_vector_train[i] * Ei
        E_vec.append(Ei)
        P_vec.append(Pi)
        Y_dem.append(demand_i)

        # --- inventory equation (13a) ---
        inv_i = -0.5 * (level_i[1:] + level_i[:-1]) * dt
        H_inv.append(inv_i)
        Y_inv.append(level_diff_i + demand_i)

    # demand part
    E_vec = np.concatenate(E_vec)
    P_vec = np.concatenate(P_vec)
    Y_dem = np.concatenate(Y_dem)
    H_dem = np.column_stack([E_vec, P_vec])
    est_ab, *_ = np.linalg.lstsq(H_dem, Y_dem, rcond=None)
    demand_res = weight[0] * (Y_dem - H_dem @ est_ab)

    # inventory part
    H_inv = np.concatenate(H_inv)
    Y_inv = np.concatenate(Y_inv)
    inv_res = weight[1] * (Y_inv - H_inv * lam)

    return np.concatenate([demand_res, inv_res])


# ── IRLS.m ───────────────────────────────────────────────────────────────────

def irls(time0, p_vector_train, time_train, demand_train, level_diff_train,
         level_train, weight_initial, lam_initial, max_iter=10, tol=1e-10):
    """Iteratively Reweighted Least Squares for λ, α, β.

    Alternates between:
      1. Updating σ̂²_I, σ̂²_d from current λ, α, β.
      2. Re-estimating λ via Levenberg–Marquardt on the weighted joint residual.

    Parameters
    ----------
    time0           : float               Order-arrival time.
    p_vector_train  : (m,) array           Price per cycle.
    time_train      : list of 1-D arrays   Sampling times per cycle.
    demand_train    : list of 1-D arrays   Observed demands per cycle.
    level_diff_train: list of 1-D arrays   Observed inventory changes per cycle.
    level_train     : list of 1-D arrays   AGO-reconstructed inventory levels.
    weight_initial  : (2,) array           [1/σ̂²_d, 1/σ̂²_I] initial weights.
    lam_initial     : float               Initial λ.
    max_iter        : int                 Max IRLS iterations.
    tol             : float               Convergence tolerance on λ.

    Returns
    -------
    lam_est  : float    Final λ estimate.
    history  : ndarray  λ history across iterations (length ≤ max_iter).
    """
    lam_prev = lam_initial
    history = np.full(max_iter, np.nan)
    history[0] = lam_initial
    weight = weight_initial.copy()

    # Trust-Region Reflective (supports bounds, matches MATLAB lsqnonlin LM)
    trf_opts = dict(
        method='trf',
        max_nfev=10,
        ftol=1e-10,
        xtol=1e-6,
    )

    for it in range(1, max_iter):
        # Step 1 — re-estimate λ
        res = least_squares(
            lambda lam: lambda_residual(
                lam, time0, p_vector_train, time_train, demand_train,
                level_diff_train, level_train, weight),
            lam_prev, bounds=(0.0, 0.2), **trf_opts
        )
        lam_next = res.x[0]

        # Step 2 — update variance estimates
        inv_var = lambda_inventory_residual_var(
            time0, time_train, demand_train,
            level_diff_train, level_train, lam_next)
        _, _, dem_var = lambda_to_alpha_beta(
            time0, time_train, p_vector_train, demand_train, lam_next)

        # Step 3 — update weights
        weight = np.array([1.0 / dem_var, 1.0 / inv_var])

        # Step 4 — convergence check
        if np.abs(lam_next - lam_prev) < tol:
            history[it] = lam_next
            break

        history[it] = lam_next
        lam_prev = lam_next

    return lam_next, history[:it + 1]
