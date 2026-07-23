"""Profit functions for the inventory optimisation model.

Translations of:
  profit.m       → profit()
  profit_appro.m → profit_approx()
"""

import numpy as np


def profit(alpha, beta, p, lam, c, h, K, T):
    """Exact profit per unit time  P(p, T) — equation (19).

    P = (α−βp)(pλ+h)/(λ²T) · [1−e^{−λT}] − (α−βp)(cλ+h)/λ − K/T

    Parameters
    ----------
    alpha, beta : float    Demand parameters.
    p           : float    Selling price.
    lam         : float    Deterioration rate λ.
    c           : float    Unit purchase cost.
    h           : float    Holding cost per unit per day.
    K           : float    Fixed ordering cost.
    T           : ndarray  Ordering cycle(s) to evaluate.

    Returns
    -------
    ndarray  Profit for each T.
    """
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)

    Dp = alpha - beta * p                          # price effect
    par1 = Dp * (p * lam + h) / (lam ** 2)
    par2 = Dp * (c * lam + h) / lam

    return par1 * (1.0 - np.exp(-lam * T)) / T - par2 - K / T


def profit_approx(alpha, beta, p, lam, c, h, K, T):
    """Approximate profit  P̃(p, T) — equation (21), Taylor expansion.

    P̃ = (p−c)(α−βp) − (pλ+h)(α−βp)·T/2 − K/T

    Valid when  λT < 0.1  (fresh products, short cycles).
    """
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)

    Dp = alpha - beta * p
    par1 = (p - c) * Dp
    par2 = (p * lam + h) * Dp * T / 2.0

    return par1 - par2 - K / T
