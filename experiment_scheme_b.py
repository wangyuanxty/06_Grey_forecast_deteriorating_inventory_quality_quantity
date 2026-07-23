#!/usr/bin/env python3
"""
Scheme B "Original" — paper's exact pipeline, only demand function swapped to NN.

Paper's pipeline preserved:
  1. Inventory regression (13a):  l + d = -lambda * mean(L) * dt  → OLS for lambda
  2. Demand model:                d(t) = NN(features) * exp(-lambda*t)
  3. IRLS-style iteration:        lambda ← OLS  →  NN ← Adam  →  loop

Uses stock_hour6_22_cnt as I(t) proxy (it's already a level, no AGO needed).
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATADIR = os.path.join(os.path.dirname(__file__), "FreshRetailNet-50K")
OUTDIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTDIR, exist_ok=True)
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# 1.  Data
# ═══════════════════════════════════════════════════════════════

def load_data():
    train_df = pd.read_parquet(os.path.join(DATADIR, "train.parquet"))
    eval_df  = pd.read_parquet(os.path.join(DATADIR, "eval.parquet"))
    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    df["day_of_week"] = df["dt"].dt.dayofweek.values.astype(np.float32)
    return df


def stockout_rate(arr):
    if arr is None or len(arr) == 0: return 1.0
    return float(np.mean(arr))


def extract_cycles(product_df):
    """Split into cycles with stock_hour6_22_cnt as I(t) proxy."""
    product_df = product_df.sort_values("dt").reset_index(drop=True)
    rates = product_df["hours_stock_status"].apply(stockout_rate)
    in_stock = rates < 0.5
    cycles = []
    i, n = 0, len(product_df)
    while i < n:
        if not in_stock.iloc[i]:
            i += 1; continue
        j = i
        while j < n and in_stock.iloc[j]:
            j += 1
        block = product_df.iloc[i:j]
        if len(block) >= 3:
            cycles.append(block)
        i = j
    return cycles


def build_cycle(block):
    """Extract I, l, d, t, features from one cycle."""
    m = len(block)
    I = block["stock_hour6_22_cnt"].values.astype(np.float32)
    d = block["sale_amount"].values.astype(np.float32)

    # l(t): inventory decrease from previous day
    l = np.zeros(m, dtype=np.float32)
    for k in range(1, m):
        l[k] = max(0.0, I[k - 1] - I[k])

    t = np.arange(m, dtype=np.float32)

    # features
    feats = np.column_stack([
        block["discount"].values,
        block["day_of_week"].values / 6.0,
        block["holiday_flag"].values.astype(np.float32),
        block["activity_flag"].values.astype(np.float32),
        block["avg_temperature"].values / 40.0,
        block["avg_humidity"].values / 100.0,
        block["precpt"].values / 50.0,
    ])
    lag1 = np.concatenate([[d[0]], d[:-1]])
    lag2 = np.concatenate([[d[0]], [d[0]], d[:-2]])
    feats = np.column_stack([feats, lag1 / 10.0, lag2 / 10.0])

    return dict(m=m, t=t, I=I, l=l, d=d, f_raw=feats.copy())


# ═══════════════════════════════════════════════════════════════
# 2.  Paper equation (13a): OLS for lambda
# ═══════════════════════════════════════════════════════════════

def estimate_lambda(cycles, use_pred_d=True):
    """
    l_j + d_j = -lambda * 0.5 * (I_j + I_{j-1}) * dt

    If use_pred_d=True, use d_pred (from NN) instead of d_obs.
    """
    H, Y = [], []
    for cyc in cycles:
        I, l, d = cyc["I"], cyc["l"], cyc.get("d_pred", cyc["d"])
        for k in range(1, cyc["m"]):
            X = -0.5 * (I[k] + I[k - 1]) * 1.0
            if abs(X) > 1e-8:
                H.append(X)
                Y.append(l[k] + d[k])
    if len(H) < 3:
        return 0.01
    H, Y = np.array(H), np.array(Y)
    lam = np.dot(H, Y) / np.dot(H, H)
    return max(float(lam), 1e-6)


# ═══════════════════════════════════════════════════════════════
# 3.  NN demand model
# ═══════════════════════════════════════════════════════════════

class DemandNN(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8),  nn.ReLU(),
            nn.Linear(8, 1),   nn.Softplus(),
        )

    def forward(self, feat, lam, t):
        return self.net(feat).squeeze(-1) * torch.exp(-lam * t)


# ═══════════════════════════════════════════════════════════════
# 4.  IRLS training:  lambda <-> NN
# ═══════════════════════════════════════════════════════════════

def train_irls(train_cycles, val_cycles, scaler, n_irls=5, nn_epochs=150):
    n_feat = train_cycles[0]["f_raw"].shape[1]

    # Step 0: initial lambda from observed demand
    lam = estimate_lambda(train_cycles, use_pred_d=False)
    lam_hist = [lam]
    print(f"  lambda init: {lam:.4f}")

    model = DemandNN(n_feat).to(DEVICE)

    for irls_iter in range(n_irls):
        # --- fix lambda, train NN ---
        lam_t = torch.tensor(lam, dtype=torch.float32, device=DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.5)

        for ep in range(nn_epochs):
            model.train()
            total = 0.0
            for cyc in train_cycles:
                t = torch.tensor(cyc["t"], dtype=torch.float32, device=DEVICE)
                f = torch.tensor(cyc["f_scl"], dtype=torch.float32, device=DEVICE)
                d = torch.tensor(cyc["d"], dtype=torch.float32, device=DEVICE)
                loss = nn.MSELoss()(model(f, lam_t, t), d)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            sched.step(total)

        # --- update d_pred, re-estimate lambda ---
        model.eval()
        with torch.no_grad():
            for cyc in train_cycles:
                t = torch.tensor(cyc["t"], dtype=torch.float32, device=DEVICE)
                f = torch.tensor(cyc["f_scl"], dtype=torch.float32, device=DEVICE)
                cyc["d_pred"] = model(f, lam_t, t).cpu().numpy()

        lam_new = estimate_lambda(train_cycles, use_pred_d=True)
        lam = 0.7 * lam + 0.3 * lam_new  # smooth
        lam_hist.append(lam)

        if abs(lam_new - lam) < 1e-6:
            break

    return model, lam, lam_hist


def evaluate(model, lam, cycles):
    model.eval()
    obs_all, pred_all = [], []
    lam_t = torch.tensor(lam, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for cyc in cycles:
            t = torch.tensor(cyc["t"], dtype=torch.float32, device=DEVICE)
            f = torch.tensor(cyc["f_scl"], dtype=torch.float32, device=DEVICE)
            p = model(f, lam_t, t).cpu().numpy()
            obs_all.extend(cyc["d"].tolist())
            pred_all.extend(p.tolist())
    obs, pred = np.array(obs_all), np.array(pred_all)
    ssr = np.sum((obs - pred) ** 2)
    sst = np.sum((obs - obs.mean()) ** 2)
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    return r2, obs, pred


# ═══════════════════════════════════════════════════════════════
# 5.  Paper baseline
# ═══════════════════════════════════════════════════════════════

def fit_paper_baseline(cycles):
    from scipy.optimize import least_squares

    def residual(theta):
        lam, alpha, beta = theta
        if lam <= 0 or alpha <= 0: return np.full(1000, 1e6)
        res = []
        for cyc in cycles:
            p = float(np.mean(cyc["f_raw"][:, 0]))
            Dp = max(alpha - beta * p, 0.0)
            t = cyc["t"]; d = cyc["d"]
            for k in range(cyc["m"]):
                tp = 0.0 if k == 0 else t[k - 1]
                dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * t[k]))
                res.append(d[k] - dp)
        return np.array(res)

    total_d = sum(c["d"].sum() for c in cycles)
    n_obs = sum(c["m"] for c in cycles)
    lam0 = estimate_lambda(cycles, use_pred_d=False)
    x0 = [lam0, total_d / n_obs * 2, total_d / n_obs * 0.1]
    res = least_squares(residual, x0, bounds=([1e-6, 1e-6, 0], [1.0, 1e6, 1e6]),
                        method="trf", max_nfev=500, ftol=1e-8)
    lam, alpha, beta = res.x

    obs_all, pred_all = [], []
    for cyc in cycles:
        p = float(np.mean(cyc["f_raw"][:, 0]))
        Dp = max(alpha - beta * p, 0.0)
        t = cyc["t"]
        for k in range(cyc["m"]):
            tp = 0.0 if k == 0 else t[k - 1]
            dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * t[k]))
            obs_all.append(cyc["d"][k])
            pred_all.append(dp)
    obs, pred = np.array(obs_all), np.array(pred_all)
    ssr = np.sum((obs - pred) ** 2)
    sst = np.sum((obs - obs.mean()) ** 2)
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    return lam, alpha, beta, r2


def predict_paper(lam, alpha, beta, cycles):
    obs_all, pred_all = [], []
    for cyc in cycles:
        p = float(np.mean(cyc["f_raw"][:, 0]))
        Dp = max(alpha - beta * p, 0.0)
        t = cyc["t"]
        for k in range(cyc["m"]):
            tp = 0.0 if k == 0 else t[k - 1]
            dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * t[k]))
            obs_all.append(cyc["d"][k])
            pred_all.append(dp)
    obs, pred = np.array(obs_all), np.array(pred_all)
    ssr = np.sum((obs - pred) ** 2)
    sst = np.sum((obs - obs.mean()) ** 2)
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    return r2, obs, pred


# ═══════════════════════════════════════════════════════════════
# 6.  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading data ...")
    df = load_data()

    combo_stats = df.groupby(["product_id", "store_id"]).agg(
        n_days=("dt", "nunique"),
        mean_sales=("sale_amount", "mean"),
        stock_nonzero=("stock_hour6_22_cnt", lambda x: (x > 0).mean()),
        discount_std=("discount", "std"),
    ).reset_index()

    good = combo_stats[
        (combo_stats["n_days"] >= 50)
        & (combo_stats["mean_sales"] > 3)
        & (combo_stats["stock_nonzero"] > 0.2)
        & (combo_stats["discount_std"] > 0.02)
    ]
    print(f"Candidates: {len(good)} combos")

    good["score"] = good["stock_nonzero"] + good["mean_sales"] / 20
    chosen = good.nlargest(4, "score")

    results = []
    for combo_idx, (_, row) in enumerate(chosen.iterrows()):
        pid, sid = int(row["product_id"]), int(row["store_id"])
        sub = df[(df["product_id"] == pid) & (df["store_id"] == sid)]
        cycles_df = extract_cycles(sub)
        if len(cycles_df) < 5:
            continue

        cycles = [build_cycle(c) for c in cycles_df]

        # scale features
        all_f = np.concatenate([c["f_raw"] for c in cycles], axis=0)
        scaler = StandardScaler().fit(all_f)
        for cyc in cycles:
            cyc["f_scl"] = scaler.transform(cyc["f_raw"])

        n_train = max(int(len(cycles) * 0.8), 3)
        train_c = cycles[:n_train]
        val_c   = cycles[n_train:]

        print(f"\n{'='*60}")
        print(f"Combo {combo_idx+1}: product={pid}, store={sid}")
        print(f"  cycles: {len(cycles)} ({n_train}/{len(val_c)}),  "
              f"sales: {row['mean_sales']:.1f},  stock>0: {row['stock_nonzero']*100:.0f}%")

        # --- Paper baseline ---
        t0 = time.time()
        lam_p, alpha_p, beta_p, r2_pt = fit_paper_baseline(train_c)
        r2_pv, obs_pv, pred_pv = predict_paper(lam_p, alpha_p, beta_p, val_c)
        print(f"  [Paper]  train R^2={r2_pt:.4f}  val R^2={r2_pv:.4f}  "
              f"lam={lam_p:.4f}  alpha={alpha_p:.1f}  beta={beta_p:.1f}  ({time.time()-t0:.1f}s)")

        # --- Scheme B ---
        t0 = time.time()
        model_b, lam_b, lam_hist = train_irls(train_c, val_c, scaler)
        r2_b, obs_b, pred_b = evaluate(model_b, lam_b, val_c)
        print(f"  [B-NN]   val R^2={r2_b:.4f}  lam={lam_b:.4f}  ({time.time()-t0:.1f}s)")
        print(f"    lam history: {[f'{l:.4f}' for l in lam_hist]}")

        results.append(dict(
            product_id=pid, store_id=sid, n_cycles=len(cycles),
            r2_paper_train=r2_pt, r2_paper_val=r2_pv, lam_paper=lam_p,
            r2_nn=r2_b, lam_nn=lam_b,
        ))

        # plot first combo
        if combo_idx == 0 and len(val_c) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, obs, pred, title in [
                (axes[0], obs_pv, pred_pv,
                 f"Paper baseline  (val R^2={r2_pv:.3f}, lam={lam_p:.4f})"),
                (axes[1], obs_b, pred_b,
                 f"Scheme B NN  (val R^2={r2_b:.3f}, lam={lam_b:.4f})"),
            ]:
                ax.scatter(obs, pred, alpha=0.5, s=15)
                ax.plot([0, obs.max()], [0, obs.max()], "r--", lw=1)
                ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
                ax.set_title(title, fontsize=11)
            fig.suptitle(f"product={pid} store={sid}", fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, "scheme_b_comparison.png"), dpi=150)
            fig.savefig(os.path.join(OUTDIR, "scheme_b_comparison.pdf"))
            plt.close(fig)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    rdf = pd.DataFrame(results)
    for label, r2k, lamk in [
        ("Paper baseline", "r2_paper_val", "lam_paper"),
        ("Scheme B NN",   "r2_nn",         "lam_nn"),
    ]:
        r2s = rdf[r2k].values
        lams = rdf[lamk].values
        print(f"  {label:16s}:  R^2 mean={np.mean(r2s):.4f}  median={np.median(r2s):.4f}  |  "
              f"lam mean={np.mean(lams):.4f}  median={np.median(lams):.4f}")

    rdf.to_csv(os.path.join(OUTDIR, "scheme_b_results.csv"), index=False)
    print(f"\nResults -> output/scheme_b_results.csv")


if __name__ == "__main__":
    main()
