#!/usr/bin/env python3
"""
Reconstruct I(t) from hourly stock_status + hourly sales in FreshRetailNet-50K,
then run the paper's full dual-equation model.

Facts: Standalone script. Reads FreshRetailNet-50K/{train,eval}.parquet.
Writes output/dingdong_full_model.csv (pid, sid, n_cycles, lam_full, alpha_full,
beta_full, r2_full_val, r2_full_train, lam_demand, alpha_demand, beta_demand,
r2_demand_val, Q_mean, I_mean).
"""

import os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DATADIR = os.path.join(os.path.dirname(__file__), "FreshRetailNet-50K")
OUTDIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTDIR, exist_ok=True)


# ===== 1. I(t) reconstruction from hourly data =====

def reconstruct_inventory_from_hourly(cycle_blocks):
    """
    For each cycle, find the exact stockout HOUR and reconstruct daily
    opening inventory backwards:
      I(opening day T)   = sales before stockout hour on day T
      I(opening day T-1) = I(opening day T) + sales on day T-1
      ...
      Q = I(opening day 1)
    """
    cycles = []
    for block in cycle_blocks:
        n = len(block)
        if n < 2:
            continue

        hours_sale   = block["hours_sale"].values
        hours_status = block["hours_stock_status"].values
        discount     = block["discount"].values.astype(float)
        d_daily      = block["sale_amount"].values.astype(float)

        # find stockout hour per day (first hour with status > 0.5)
        stockout_hour = np.full(n, 24, dtype=int)
        for k in range(n):
            st = np.array(hours_status[k], dtype=float)
            idx = np.where(st > 0.5)[0]
            if len(idx) > 0:
                stockout_hour[k] = idx[0]

        # sales before stockout hour on each day
        sales_before_so = np.zeros(n)
        for k in range(n):
            hs = np.array(hours_sale[k], dtype=float)
            h = stockout_hour[k]
            sales_before_so[k] = float(np.sum(hs[:h]))

        # reconstruct backwards
        I_open = np.zeros(n)
        I_open[-1] = sales_before_so[-1]

        for k in range(n - 2, -1, -1):
            if stockout_hour[k] < 24:
                I_open[k] = sales_before_so[k]
            else:
                I_open[k] = I_open[k + 1] + d_daily[k]

        Q = I_open[0]

        # inventory change l(t) (positive = inventory decreased)
        l = np.zeros(n)
        I_close = np.maximum(I_open - d_daily, 0.0)
        for k in range(n - 1):
            l[k] = max(0.0, I_open[k] - I_close[k])
        l[-1] = I_open[-1]

        cycles.append(dict(
            n=n, t=np.arange(n, dtype=np.float32),
            I=I_open, d=d_daily, l=l,
            p=float(np.mean(discount)), Q=Q,
        ))
    return cycles


# ===== 2. Paper's estimation =====

def estimate_lambda_from_I(cycles):
    """Eq (13a): l_j + d_j = -lambda * 0.5*(I_j + I_{j-1})*dt"""
    H, Y = [], []
    for cyc in cycles:
        I, l, d = cyc["I"], cyc["l"], cyc["d"]
        for k in range(1, cyc["n"]):
            X = -0.5 * (I[k] + I[k - 1]) * 1.0
            y = l[k] + d[k]
            if abs(X) > 1e-6 and abs(y) > 1e-6:
                H.append(X); Y.append(y)
    if len(H) < 3:
        return 0.001, len(H)
    H, Y = np.array(H), np.array(Y)
    lam = np.dot(H, Y) / np.dot(H, H)
    return max(float(lam), 1e-8), len(H)


def estimate_from_demand_only(cycles):
    """Eq (13b) only — no I(t) needed (baseline)."""
    def residual(theta):
        lam, alpha, beta = theta
        if lam <= 0 or alpha <= 0: return np.full(2000, 1e6)
        res = []
        for cyc in cycles:
            p, d = cyc["p"], cyc["d"]
            Dp = max(alpha - beta * p, 0.0)
            for k in range(cyc["n"]):
                tp = max(k - 0.5, 0.0); tk = k + 0.5
                dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * tk))
                res.append(float(d[k]) - dp)
        res.append(0.0)
        return np.array(res)

    td = sum(c["d"].sum() for c in cycles)
    no = sum(c["n"] for c in cycles)
    x0 = [0.01, td / no * 2, td / no * 0.1]
    sol = least_squares(residual, x0,
                        bounds=([1e-8, 1e-6, 0], [1.0, 1e6, 1e6]),
                        method="trf", max_nfev=500, ftol=1e-8)
    return sol.x[0], sol.x[1], sol.x[2]


def fit_full_model(cycles):
    """lambda from I(t) regression, alpha/beta from demand regression."""
    lam, n_pts = estimate_lambda_from_I(cycles)

    def residual(theta):
        alpha, beta = theta
        if alpha <= 0: return np.full(1000, 1e6)
        res = []
        for cyc in cycles:
            p, d = cyc["p"], cyc["d"]
            Dp = max(alpha - beta * p, 0.0)
            for k in range(cyc["n"]):
                tp = max(k - 0.5, 0.0); tk = k + 0.5
                dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * tk))
                res.append(float(d[k]) - dp)
        res.append(0.0)
        return np.array(res)

    td = sum(c["d"].sum() for c in cycles)
    no = sum(c["n"] for c in cycles)
    x0 = [td / no * 2, td / no * 0.1]
    try:
        sol = least_squares(residual, x0,
                            bounds=([1e-6, 0], [1e6, 1e6]),
                            method="trf", max_nfev=500, ftol=1e-8)
        alpha, beta = sol.x
    except Exception:
        alpha, beta = td / no * 2, 0.0
    return lam, alpha, beta, n_pts


def compute_r2(cycles, lam, alpha, beta):
    obs_all, pred_all = [], []
    for cyc in cycles:
        p, d = cyc["p"], cyc["d"]
        Dp = max(alpha - beta * p, 0.0)
        for k in range(cyc["n"]):
            tp = max(k - 0.5, 0.0); tk = k + 0.5
            dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * tk))
            obs_all.append(float(d[k])); pred_all.append(float(dp))
    obs, pred = np.array(obs_all), np.array(pred_all)
    ssr = np.sum((obs - pred) ** 2); sst = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - ssr / sst if sst > 0 else 0.0


# ===== 3. Data loading =====

def load_data():
    train_df = pd.read_parquet(os.path.join(DATADIR, "train.parquet"))
    eval_df  = pd.read_parquet(os.path.join(DATADIR, "eval.parquet"))
    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def stockout_rate(arr):
    if arr is None or len(arr) == 0: return 1.0
    return float(np.mean(arr))


def extract_cycles(product_df):
    product_df = product_df.sort_values("dt").reset_index(drop=True)
    rates = product_df["hours_stock_status"].apply(stockout_rate)
    in_stock = rates < 0.5
    cycles = []
    i, n = 0, len(product_df)
    while i < n:
        if not in_stock.iloc[i]: i += 1; continue
        j = i
        while j < n and in_stock.iloc[j]: j += 1
        block = product_df.iloc[i:j]
        if len(block) >= 3: cycles.append(block)
        i = j
    return cycles


# ===== 4. Main =====

def main():
    print("Loading data ...")
    df = load_data()

    combo_stats = df.groupby(["product_id", "store_id"]).agg(
        n_days=("dt", "nunique"),
        mean_sales=("sale_amount", "mean"),
        stockout_rate=("hours_stock_status",
                       lambda x: np.mean([stockout_rate(s) for s in x])),
        discount_std=("discount", "std"),
        discount_nunique=("discount", "nunique"),
    ).reset_index()

    good = combo_stats[
        (combo_stats["n_days"] >= 50)
        & (combo_stats["mean_sales"] > 3)
        & (combo_stats["stockout_rate"] > 0.1)
        & (combo_stats["stockout_rate"] < 0.6)
        & (combo_stats["discount_std"] > 0.02)
        & (combo_stats["discount_nunique"] >= 2)
    ]
    print(f"Candidates: {len(good)} combos")

    cycle_info = []
    for _, row in good.iterrows():
        pid, sid = int(row["product_id"]), int(row["store_id"])
        sub = df[(df["product_id"] == pid) & (df["store_id"] == sid)]
        blocks = extract_cycles(sub)
        if len(blocks) >= 4:
            cycle_info.append((pid, sid, len(blocks), row["mean_sales"],
                               row["discount_std"], row["stockout_rate"]))
    cycle_df = pd.DataFrame(cycle_info, columns=[
        "pid", "sid", "n_cycles", "mean_sales", "discount_std", "stockout_rate"])
    chosen = cycle_df.nlargest(6, "n_cycles")
    print(f"Testing {len(chosen)} combos\n")

    results = []
    for combo_idx, (_, crow) in enumerate(chosen.iterrows()):
        pid, sid = int(crow["pid"]), int(crow["sid"])
        sub = df[(df["product_id"] == pid) & (df["store_id"] == sid)]
        blocks = extract_cycles(sub)
        cycles = reconstruct_inventory_from_hourly(blocks)
        if len(cycles) < 4:
            continue

        n_train = max(int(len(cycles) * 0.8), 3)
        train_c, val_c = cycles[:n_train], cycles[n_train:]

        Qs = [c["Q"] for c in cycles]
        Is = np.concatenate([c["I"] for c in cycles])

        print(f"--- {combo_idx+1}: product={pid}, store={sid}")
        print(f"    cycles={len(cycles)}, Q=[{min(Qs):.0f},{max(Qs):.0f}], "
              f"I=[{Is.min():.0f},{Is.max():.0f}]")

        # Full model (lambda from I(t))
        lam1, alpha1, beta1, n_pts = fit_full_model(train_c)
        r2v1 = compute_r2(val_c, lam1, alpha1, beta1)
        r2t1 = compute_r2(train_c, lam1, alpha1, beta1)

        # Baseline (demand-only, no I(t))
        lam2, alpha2, beta2 = estimate_from_demand_only(train_c)
        r2v2 = compute_r2(val_c, lam2, alpha2, beta2)
        r2t2 = compute_r2(train_c, lam2, alpha2, beta2)

        print(f"    [Full model]   lam={lam1:.5f}  alpha={alpha1:.1f}  "
              f"beta={beta1:.2f}  R^2_val={r2v1:.4f}  (I-pts={n_pts})")
        print(f"    [Demand only]  lam={lam2:.5f}  alpha={alpha2:.1f}  "
              f"beta={beta2:.2f}  R^2_val={r2v2:.4f}\n")

        results.append(dict(
            pid=pid, sid=sid, n_cycles=len(cycles),
            lam_full=lam1, alpha_full=alpha1, beta_full=beta1,
            r2_full_val=r2v1, r2_full_train=r2t1, n_I_pts=n_pts,
            lam_demand=lam2, alpha_demand=alpha2, beta_demand=beta2,
            r2_demand_val=r2v2, r2_demand_train=r2t2,
            Q_mean=np.mean(Qs), I_mean=Is.mean(),
        ))

        # Plot first combo
        if combo_idx == 0:
            fig, axes = plt.subplots(2, 3, figsize=(20, 11))
            for i in range(min(6, len(cycles))):
                ax = axes.flatten()[i]
                cyc = cycles[i]
                ax.bar(cyc["t"], cyc["d"], alpha=0.5, label="sales d(t)")
                ax.plot(cyc["t"], cyc["I"], "r-o", lw=2, ms=5, label="I(t)")
                ax.set_title(f"Cycle {i+1}  Q={cyc['Q']:.0f}")
                if i == 0: ax.legend(fontsize=7)
            fig.suptitle(f"product={pid} store={sid}  "
                         f"lam={lam1:.5f}  R^2={r2v1:.3f}",
                         fontweight="bold", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, "dingdong_reconstructed_I.png"), dpi=150)
            fig.savefig(os.path.join(OUTDIR, "dingdong_reconstructed_I.pdf"))
            plt.close(fig)
            print(f"    Plot -> output/dingdong_reconstructed_I.pdf\n")

    # Summary
    rdf = pd.DataFrame(results)
    print(f"{'='*60}\nSUMMARY ({len(rdf)} combos)\n{'='*60}")
    for label, lk, rk in [
        ("Full model   ", "lam_full", "r2_full_val"),
        ("Demand only  ", "lam_demand", "r2_demand_val"),
    ]:
        lams = rdf[lk].values; r2s = rdf[rk].values
        print(f"  {label}: lam mean={np.mean(lams):.5f} med={np.median(lams):.5f}  |  "
              f"R^2 mean={np.mean(r2s):.4f} med={np.median(r2s):.4f}")
    rdf.to_csv(os.path.join(OUTDIR, "dingdong_full_model.csv"), index=False)
    print(f"\nResults -> output/dingdong_full_model.csv")


if __name__ == "__main__":
    main()
