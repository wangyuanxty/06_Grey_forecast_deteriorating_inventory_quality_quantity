"""Inventory system simulation — data generation layer.

Translations of:
  demand_rate.m         → demand_rate()
  levelattime.m         → level_at_time()
  inventory_level.m     → inventory_level()
  inventory_level_simulation.m → inventory_level_simulation()
"""

import numpy as np


# ── demand_rate.m ────────────────────────────────────────────────────────────

def demand_rate(alpha, beta, p, lam, time, time0):
    """Demand rate d(t,p) = (α − βp) · exp(−λ (t − t₀)).

    Parameters
    ----------
    alpha : float       Potential (zero-price) demand.
    beta  : float       Price-sensitivity coefficient.
    p     : float       Selling price.
    lam   : float       Deterioration rate λ.
    time  : ndarray     Evaluation time(s).
    time0 : float       Order-arrival time (start of cycle).

    Returns
    -------
    ndarray  Demand rate at each time in `time`.
    """
    Dp = alpha - beta * p                     # price effect
    Rt = np.exp(-lam * (np.asarray(time) - time0))  # quality effect
    return Dp * Rt


# ── levelattime.m ────────────────────────────────────────────────────────────

def level_at_time(alpha, beta, p, lam, time, time0, Q):
    """Inventory level at a single time point  I(t) = (α−βp)·e^{−λ(t−t₀)}·(T−t).

    Parameters
    ----------
    alpha, beta, p, lam, time, time0 : see demand_rate.
    Q : float   Order quantity (initial inventory).

    Returns
    -------
    float  Inventory level at `time`.
    """
    T = Q / (alpha - beta * p)               # cycle length
    t_end = time0 + T                         # depletion time
    return (alpha - beta * p) * np.exp(-lam * (time - time0)) * (t_end - time)


# ── inventory_level.m ────────────────────────────────────────────────────────

def inventory_level(alpha, beta, p, lam, time0, delta_t, Q):
    """Exact (noise-free) inventory trajectory for one replenishment cycle.

    Parameters
    ----------
    alpha, beta, p, lam, time0 : see demand_rate.
    delta_t : float  Time resolution (e.g. 1 day).
    Q       : float  Order quantity.

    Returns
    -------
    time        : (n,) ndarray  Sampling times.
    demand      : (n,) ndarray  Exact demand at each sample.
    level_diff  : (n,) ndarray  Inventory change at each sample (diff of I).
    level       : (n,) ndarray  Inventory level at each sample.
    """
    T = Q / (alpha - beta * p)
    t_end = time0 + T
    time = np.arange(time0 + delta_t, t_end + 1e-12, delta_t)

    demand = (alpha - beta * p) * np.exp(-lam * (time - time0))
    level = (alpha - beta * p) * np.exp(-lam * (time - time0)) * (t_end - time)
    level_diff = -np.diff(np.concatenate([[Q], level]))

    return time, demand, level_diff, level


# ── inventory_level_simulation.m ─────────────────────────────────────────────

def inventory_level_simulation(alpha, beta, p, std_dev, lam,
                               time0, delta_t, Q, rng=None):
    """Noisy inventory simulation for one replenishment cycle.

    Uses Poisson noise for deterioration loss (preserving monotonic decay)
    and Gaussian noise for demand observation.

    Parameters
    ----------
    alpha, beta, p, lam, time0, delta_t, Q : see inventory_level.
    std_dev : float  Std. deviation of demand observation noise  σ_d.
    rng     : numpy.random.Generator or None  Random state for reproducibility.

    Returns
    -------
    time_simu        : (n,) ndarray  Simulated sampling times.
    demand_simu      : (n,) ndarray  Simulated (noisy) demand.
    level_diff_simu  : (n,) ndarray  Simulated (noisy) inventory changes.
    level_simu       : (n,) ndarray  Simulated inventory levels.
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- initialisation -------------------------------------------------------
    time_list, demand_list, level_diff_list, level_list = [], [], [], []
    level_remain = Q
    time_k1 = time0
    time_k = time_k1 + delta_t

    # First step: demand with Gaussian noise (enforce positivity)
    demand = (1.0 / lam) * (alpha - beta * p) * (
        np.exp(-lam * (time_k1 - time0)) - np.exp(-lam * (time_k - time0))
    )
    demand_err = demand + std_dev * rng.standard_normal()
    while demand_err < 0:
        demand_err = demand + std_dev * rng.standard_normal()

    # Deterioration loss as Poisson draw (clamp: deterioration ≥ 0)
    deterioration = max(0.0, lam * (
        0.5 * level_at_time(alpha, beta, p, lam, time_k, time0, Q)
        + 0.5 * level_at_time(alpha, beta, p, lam, time_k1, time0, Q)
    ))
    deterioration_pois = rng.poisson(delta_t * deterioration)
    reduction = deterioration_pois + demand_err
    level_remain -= reduction

    # --- loop until depletion ------------------------------------------------
    while level_remain > 0:
        time_list.append(time_k)
        demand_list.append(demand_err)
        level_list.append(level_remain)
        level_diff_list.append(-reduction)

        # advance time
        time_k1 = time_k
        time_k = time_k + delta_t

        # demand (trapezoid-averaged instantaneous rate + noise)
        demand = 0.5 * demand_rate(alpha, beta, p, lam, time_k, time0) \
               + 0.5 * demand_rate(alpha, beta, p, lam, time_k1, time0)
        demand_err = demand + std_dev * rng.standard_normal()
        while demand_err < 0:
            demand_err = demand + std_dev * rng.standard_normal()

        # deterioration (Poisson — clamp: deterioration ≥ 0)
        deterioration = max(0.0, lam * (
            0.5 * level_at_time(alpha, beta, p, lam, time_k, time0, Q)
            + 0.5 * level_at_time(alpha, beta, p, lam, time_k1, time0, Q)
        ))
        deterioration_pois = rng.poisson(delta_t * deterioration)
        reduction = deterioration_pois + demand_err
        level_remain -= reduction

    return (np.array(time_list), np.array(demand_list),
            np.array(level_diff_list), np.array(level_list))
