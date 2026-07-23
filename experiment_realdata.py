#!/usr/bin/env python3
"""
Real-data experiment: validate the paper's grey forecasting model on
FreshRetailNet-50K (Dingdong Inc. fresh retail data).

Pipeline:
  1. Load local parquet data
  2. From hours_stock_status, carve out replenishment cycles
  3. Map discount → price proxy, sale_amount → demand d(t)
  4. Fit the demand equation (13b) to estimate lambda, alpha, beta
  5. Evaluate prediction accuracy and report results
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Load data
# ═══════════════════════════════════════════════════════════════════════════════

DATADIR = os.path.join(os.path.dirname(__file__), "FreshRetailNet-50K")
OUTDIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTDIR, exist_ok=True)

print("Loading data ...")
train = pd.read_parquet(os.path.join(DATADIR, "train.parquet"))
eval_df = pd.read_parquet(os.path.join(DATADIR, "eval.parquet"))
df = pd.concat([train, eval_df], ignore_index=True)
df["dt"] = pd.to_datetime(df["dt"])
print(f"  {len(df):,} rows,  {df['dt'].min().date()} ~ {df['dt'].max().date()},  "
      f"{df['product_id'].nunique()} products x {df['store_id'].nunique()} stores")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Pre-processing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def daily_stockout_rate(status_array):
    """Fraction of hours in stockout (1=stockout, 0=in-stock)."""
    if status_array is None or len(status_array) == 0:
        return 1.0
    return float(np.mean(status_array))


def extract_cycles(product_df):
    """Split a productxstore time series into replenishment cycles.

    A day is 'in stock' if stockout_rate < 0.5.
    A cycle = contiguous block of in-stock days.

    Returns list of dicts: [{p, d_array, t0, days}, ...]
    """
    product_df = product_df.sort_values("dt").reset_index(drop=True)

    # mark in-stock days
    stock_rates = product_df["hours_stock_status"].apply(daily_stockout_rate)
    in_stock = stock_rates < 0.5

    cycles = []
    i = 0
    n = len(product_df)
    while i < n:
        if not in_stock.iloc[i]:
            i += 1
            continue
        # start of cycle
        j = i
        while j < n and in_stock.iloc[j]:
            j += 1
        # cycle = rows [i, j)
        block = product_df.iloc[i:j]
        if len(block) >= 3:  # need >=3 days for a meaningful fit
            # use discount as price proxy: p = discount (higher = pricier)
            p_mean = float(block["discount"].mean())
            cycles.append(dict(
                t0=float(i),                       # relative cycle index
                days=(block["dt"] - block["dt"].iloc[0]).dt.days.values.astype(float),
                d=block["sale_amount"].values.astype(float),
                p=p_mean,
                discount=block["discount"].values.astype(float),
                stock_cnt=block["stock_hour6_22_cnt"].values.astype(float),
            ))
        i = j
    return cycles


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Demand-equation parameter estimation
# ═══════════════════════════════════════════════════════════════════════════════

def demand_residual(theta, cycles):
    """Residual vector for nonlinear least-squares on demand equation (13b).

    theta = [lam, alpha, beta]
    Equation:  d_j ≈ (alpha−betap)/lambda · [e^{−lambda·t_{j-1}} − e^{−lambda·t_j}]
    """
    lam, alpha, beta = theta
    if lam <= 0 or alpha <= 0 or beta <= 0:
        return np.full(10000, 1e6)

    residuals = []
    for cyc in cycles:
        p = cyc["p"]
        t = cyc["days"]                     # t_j (relative to cycle start, t0=0)
        t_prev = np.concatenate([[0.0], t[:-1]])
        d_obs = cyc["d"]

        Dp = alpha - beta * p
        if Dp <= 0:
            residuals.extend([1e6] * len(d_obs))
            continue

        exp_prev = np.exp(-lam * t_prev)
        exp_cur  = np.exp(-lam * t)
        d_pred = (Dp / lam) * (exp_prev - exp_cur)

        residuals.extend((d_obs - d_pred).tolist())

    return np.array(residuals)


def estimate_params(cycles, lam0=0.05):
    """Estimate lambda, alpha, beta from demand equation (13b) via nonlinear LS."""
    # initial guess
    total_d = np.sum([np.sum(c["d"]) for c in cycles])
    n_obs = sum(len(c["d"]) for c in cycles)
    alpha0 = total_d / n_obs * 2.0
    beta0 = alpha0 * 0.1
    x0 = [lam0, alpha0, beta0]

    res = least_squares(
        demand_residual, x0,
        args=(cycles,),
        bounds=([1e-6, 1e-6, 1e-6], [1.0, np.inf, np.inf]),
        method="trf", max_nfev=500, ftol=1e-8, xtol=1e-8,
    )
    return res.x, res.cost


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Main experiment loop
# ═══════════════════════════════════════════════════════════════════════════════

# --- 4a. Select candidate productxstore combos --------------------------------
print("\nSelecting candidate combos ...")

combo_stats = df.groupby(["product_id", "store_id"]).agg(
    n_days=("dt", "nunique"),
    total_sales=("sale_amount", "sum"),
    mean_sales=("sale_amount", "mean"),
    stockout_rate=("hours_stock_status", lambda x: np.mean(
        [daily_stockout_rate(s) for s in x])),
    discount_std=("discount", "std"),
    discount_nunique=("discount", "nunique"),
).reset_index()

# Filters:
#   >= 30 days of data
#   mean daily sales > 3 (enough signal)
#   discount varies (std > 0.02 AND at least 2 distinct values)
#   not always stockout (< 0.8)
good = combo_stats[
    (combo_stats["n_days"] >= 30)
    & (combo_stats["mean_sales"] > 3)
    & (combo_stats["discount_std"] > 0.02)
    & (combo_stats["discount_nunique"] >= 2)
    & (combo_stats["stockout_rate"] < 0.8)
]
print(f"  Candidates: {len(good)} combos  (from {len(combo_stats)} total)")

# Sample up to 200 combos (balance coverage vs runtime)
SAMPLE_N = min(200, len(good))
rng = np.random.default_rng(42)
sampled = good.sample(SAMPLE_N, random_state=rng).reset_index(drop=True)

# --- 4b. Run estimation on each combo -----------------------------------------
print(f"\nRunning estimation on {len(sampled)} combos ...")
t_start = time.time()

results = []
for idx, row in sampled.iterrows():
    pid, sid = int(row["product_id"]), int(row["store_id"])
    sub = df[(df["product_id"] == pid) & (df["store_id"] == sid)]
    cycles = extract_cycles(sub)

    if len(cycles) < 5:          # need multiple cycles with price variation
        continue

    # skip if all cycles have near-identical price (beta unidentifiable)
    prices = np.array([c["p"] for c in cycles])
    if np.std(prices) < 1e-4:
        continue

    try:
        theta, cost = estimate_params(cycles)
        lam_hat, alpha_hat, beta_hat = theta

        # compute R^2 on predicted vs observed demand
        all_pred, all_obs = [], []
        for cyc in cycles:
            p = cyc["p"]
            t = cyc["days"]
            t_prev = np.concatenate([[0.0], t[:-1]])
            Dp = max(alpha_hat - beta_hat * p, 0.01)
            d_pred = (Dp / lam_hat) * (
                np.exp(-lam_hat * t_prev) - np.exp(-lam_hat * t))
            all_pred.extend(d_pred.tolist())
            all_obs.extend(cyc["d"].tolist())

        all_pred = np.array(all_pred)
        all_obs = np.array(all_obs)
        ss_res = np.sum((all_obs - all_pred) ** 2)
        ss_tot = np.sum((all_obs - np.mean(all_obs)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        results.append(dict(
            product_id=pid, store_id=sid,
            n_cycles=len(cycles),
            n_days=row["n_days"],
            lam_hat=lam_hat, alpha_hat=alpha_hat, beta_hat=beta_hat,
            r2=r2, cost=cost,
            n_obs=len(all_obs),
        ))
    except Exception as e:
        continue

    if (len(results) + 1) % 20 == 0:
        elapsed = time.time() - t_start
        print(f"  {len(results)}/{len(sampled)} done  ({elapsed:.0f}s)")

elapsed = time.time() - t_start
print(f"\n  Finished: {len(results)} successful fits in {elapsed:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Results summary
# ═══════════════════════════════════════════════════════════════════════════════

if len(results) == 0:
    print("ERROR: no successful fits — check data/criteria")
    sys.exit(1)

res_df = pd.DataFrame(results)
print(f"\n{'='*70}")
print(f"RESULTS SUMMARY  (n = {len(res_df)} productxstore combos)")
print(f"{'='*70}")

print(f"\n--- lambda (deterioration rate) ---")
lam = res_df["lam_hat"]
print(f"  Mean: {lam.mean():.4f}   Median: {lam.median():.4f}")
print(f"  Std:  {lam.std():.4f}    Range: [{lam.min():.4f}, {lam.max():.4f}]")
print(f"  Paper simulation lambda = 0.05")
print(f"  Real-data: fresh products decay ~{lam.median()*100:.1f}%/day")

print(f"\n--- alpha (potential demand) ---")
alp = res_df["alpha_hat"]
print(f"  Mean: {alp.mean():.2f}   Median: {alp.median():.2f}")

print(f"\n--- beta (price sensitivity) ---")
bet = res_df["beta_hat"]
print(f"  Mean: {bet.mean():.2f}   Median: {bet.median():.2f}")

print(f"\n--- R2 (fit quality) ---")
r2 = res_df["r2"]
print(f"  Mean: {r2.mean():.4f}   Median: {r2.median():.4f}")
print(f"  R2 > 0:   {(r2 > 0).mean()*100:.0f}%")
print(f"  R2 > 0.3: {(r2 > 0.3).mean()*100:.0f}%")
print(f"  R2 > 0.5: {(r2 > 0.5).mean()*100:.0f}%")

print(f"\n--- Cycles per combo ---")
print(f"  Mean: {res_df['n_cycles'].mean():.1f}   Median: {res_df['n_cycles'].median():.0f}")

# correlation: lambda vs discount variation
if res_df["n_cycles"].nunique() > 2:
    price_var = []
    for _, r in res_df.iterrows():
        sub = df[(df["product_id"] == int(r["product_id"]))
                 & (df["store_id"] == int(r["store_id"]))]
        price_var.append(sub["discount"].std())
    res_df["price_var"] = price_var
    corr, pval = spearmanr(res_df["price_var"], res_df["lam_hat"])
    print(f"\n--- Price variation vs lambda ---")
    print(f"  Spearman rho = {corr:.3f}  (p = {pval:.4f})")

# --- 5b. Top / bottom fits ---------------------------------------------------
print(f"\n--- Best 5 fits ---")
best = res_df.nlargest(5, "r2")
for _, r in best.iterrows():
    print(f"  product={int(r['product_id']):4d}  store={int(r['store_id']):3d}  "
          f"cycles={int(r['n_cycles']):2d}  lambda={r['lam_hat']:.4f}  "
          f"alpha={r['alpha_hat']:.1f}  beta={r['beta_hat']:.2f}  R^2={r['r2']:.3f}")

print(f"\n--- Worst 5 fits ---")
worst = res_df.nsmallest(5, "r2")
for _, r in worst.iterrows():
    print(f"  product={int(r['product_id']):4d}  store={int(r['store_id']):3d}  "
          f"cycles={int(r['n_cycles']):2d}  lambda={r['lam_hat']:.4f}  "
          f"alpha={r['alpha_hat']:.1f}  beta={r['beta_hat']:.2f}  R^2={r['r2']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Save
# ═══════════════════════════════════════════════════════════════════════════════

res_df.to_csv(os.path.join(OUTDIR, "realdata_results.csv"), index=False)
print(f"\nResults saved → output/realdata_results.csv")

# --- 6b. Plot best-fit example ------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

best_row = best.iloc[0]
pid_best, sid_best = int(best_row["product_id"]), int(best_row["store_id"])
sub_best = df[(df["product_id"] == pid_best) & (df["store_id"] == sid_best)]
cycles_best = extract_cycles(sub_best)
lam_b, alpha_b, beta_b = best_row["lam_hat"], best_row["alpha_hat"], best_row["beta_hat"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i in range(min(6, len(cycles_best))):
    ax = axes[i]
    cyc = cycles_best[i]
    t = np.concatenate([[0.0], cyc["days"]])
    d_obs = cyc["d"]
    p = cyc["p"]
    Dp = max(alpha_b - beta_b * p, 0.01)
    t_fine = np.linspace(0, t[-1], 100)
    d_pred_cont = Dp * np.exp(-lam_b * t_fine)
    t_prev = np.concatenate([[0.0], cyc["days"][:-1]])
    d_pred_disc = (Dp / lam_b) * (np.exp(-lam_b * t_prev) - np.exp(-lam_b * cyc["days"]))

    ax.plot(t_fine, d_pred_cont, "b-", lw=1.5, alpha=0.5, label="fitted d(t)")
    ax.stem(cyc["days"], d_obs, linefmt="r-", markerfmt="ro", basefmt="k-",
            label="observed sales")
    ax.stem(cyc["days"], d_pred_disc, linefmt="b--", markerfmt="bs", basefmt="k-",
            label="predicted")
    ax.set_title(f"Cycle {i+1}  (p={p:.3f})", fontsize=11)
    ax.set_xlabel("Day"); ax.set_ylabel("Demand")
    if i == 0:
        ax.legend(fontsize=8)

fig.suptitle(f"Best fit: product={pid_best} store={sid_best}  "
             f"lambda={lam_b:.4f} alpha={alpha_b:.1f} beta={beta_b:.2f}  R^2={best_row['r2']:.3f}",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "realdata_best_fit.png"), dpi=150)
fig.savefig(os.path.join(OUTDIR, "realdata_best_fit.pdf"))
plt.close(fig)
print(f"Best-fit plot saved → output/realdata_best_fit.pdf")

# --- 6c. lambda distribution histogram ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.hist(lam, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
ax2.axvline(lam.median(), color="red", lw=2, ls="--", label=f"median lambda = {lam.median():.4f}")
ax2.axvline(0.05, color="green", lw=2, ls=":", label="paper simulation lambda = 0.05")
ax2.set_xlabel("Estimated lambda (daily deterioration rate)", fontsize=13)
ax2.set_ylabel("Count", fontsize=13)
ax2.set_title(f"Distribution of lambda across {len(res_df)} productxstore combos", fontsize=14)
ax2.legend(fontsize=11)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTDIR, "realdata_lambda_dist.png"), dpi=150)
fig2.savefig(os.path.join(OUTDIR, "realdata_lambda_dist.pdf"))
plt.close(fig2)
print(f"Lambda distribution plot saved → output/realdata_lambda_dist.pdf")

print("\nDone.")