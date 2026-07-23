#!/usr/bin/env python3
"""
Per-product experiment: single store, single product.
Lists each combo with store/product IDs and data volume.
Trains hybrid model with avg_wind_level added.

Facts: Standalone. Reads FreshRetailNet-50K/{train,eval}.parquet.
Writes output/per_product_results.csv.
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATADIR = os.path.join(os.path.dirname(__file__), "FreshRetailNet-50K")
OUTDIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTDIR, exist_ok=True)
print(f"Device: {DEVICE}")


# ===== 1. Data =====

def load_data():
    train_df = pd.read_parquet(os.path.join(DATADIR, "train.parquet"))
    eval_df  = pd.read_parquet(os.path.join(DATADIR, "eval.parquet"))
    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    df["day_of_week"] = df["dt"].dt.dayofweek.values.astype(np.float32)
    return df


def build_features(df):
    """Numeric features only (single product, no embedding needed)."""
    return np.column_stack([
        np.arange(len(df), dtype=np.float32),                       # 0: t_rel
        df["discount"].values.astype(np.float32),                    # 1: price
        df["day_of_week"].values.astype(np.float32) / 6.0,           # 2: weekday
        df["holiday_flag"].values.astype(np.float32),                # 3: holiday
        df["activity_flag"].values.astype(np.float32),               # 4: promo
        df["avg_temperature"].values.astype(np.float32) / 40.0,      # 5: temp
        df["avg_humidity"].values.astype(np.float32) / 100.0,        # 6: humidity
        df["precpt"].values.astype(np.float32) / 50.0,               # 7: precip
        df["avg_wind_level"].values.astype(np.float32) / 10.0,       # 8: wind
    ])


# ===== 2. Hybrid model (tiny = single product, few params) =====

class TinyHybrid(nn.Module):
    """d = (alpha - beta*p) * NN(features)  — tiny net for single product."""
    def __init__(self, n_feat):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(1.0))    # alpha ≈ 2.7
        self.log_beta  = nn.Parameter(torch.tensor(-1.0))   # beta ≈ 0.37
        self.net = nn.Sequential(
            nn.Linear(n_feat, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Softplus(),
        )

    @property
    def alpha(self): return torch.exp(self.log_alpha)

    @property
    def beta(self):  return torch.exp(self.log_beta)

    def forward(self, feat):
        Dp = self.alpha - self.beta * feat[:, 1]
        multiplier = self.net(feat).squeeze(-1)
        return Dp * multiplier


def train_tiny(model, X_tr, y_tr, X_vl, y_vl, epochs=300, lr=0.01):
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_vl, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_vl, dtype=torch.float32, device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50, factor=0.5)

    best_val = float("inf"); best_state = None

    for ep in range(epochs):
        model.train()
        pred = model(Xt)
        loss = nn.MSELoss()(pred, yt)
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.MSELoss()(model(Xv), yv).item()
        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def compute_r2(model, X, y):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        pred = model(Xt).cpu().numpy()
    obs = np.array(y)
    ssr = np.sum((obs - pred) ** 2); sst = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - ssr / sst if sst > 0 else 0.0, obs, pred


# ===== 3. Paper baseline =====

def fit_paper(X_tr, y_tr, X_vl, y_vl):
    def residual(theta):
        lam, alpha, beta = theta
        if lam <= 0 or alpha <= 0: return np.full(500, 1e6)
        res = []
        p = np.mean(X_tr[:, 1])  # avg discount
        Dp = max(alpha - beta * p, 0.0)
        for k in range(len(y_tr)):
            tp = max(k - 0.5, 0.0); tk = k + 0.5
            dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * tk))
            res.append(float(y_tr[k]) - dp)
        res.append(0.0)
        return np.array(res)

    x0 = [0.01, np.mean(y_tr) * 2, np.mean(y_tr) * 0.1]
    try:
        sol = least_squares(residual, x0,
                            bounds=([1e-8, 1e-6, 0], [1.0, 1e6, 1e6]),
                            method="trf", max_nfev=500, ftol=1e-8)
        lam, alpha, beta = sol.x
    except Exception:
        lam, alpha, beta = 0.01, np.mean(y_tr) * 2, 0.0

    p = np.mean(X_vl[:, 1])
    Dp = max(alpha - beta * p, 0.0)
    obs_v, pred_v = [], []
    for k in range(len(y_vl)):
        tp = max(k - 0.5, 0.0); tk = k + 0.5
        dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * tk))
        obs_v.append(float(y_vl[k])); pred_v.append(float(dp))
    obs, pred = np.array(obs_v), np.array(pred_v)
    ssr = np.sum((obs - pred) ** 2); sst = np.sum((obs - obs.mean()) ** 2)
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    return lam, alpha, beta, r2


# ===== 4. Main =====

def main():
    print("Loading data ...")
    df = load_data()

    # Find product-store combos with most data
    combo_stats = df.groupby(["product_id", "store_id"]).agg(
        n_days=("dt", "nunique"),
        mean_sales=("sale_amount", "mean"),
        discount_std=("discount", "std"),
        total_sales=("sale_amount", "sum"),
    ).reset_index()

    # Filter: >= 40 days, mean sales > 1, some price variation
    good = combo_stats[(combo_stats["n_days"] >= 40)
                       & (combo_stats["mean_sales"] > 1)
                       & (combo_stats["discount_std"] > 0.01)]
    good = good.sort_values("n_days", ascending=False)
    print(f"Eligible combos: {len(good)}")

    # Test top 20
    N = min(20, len(good))
    chosen = good.head(N).reset_index(drop=True)

    print(f"\n{'='*100}")
    print(f"{'#':3s} {'product':>8s} {'store':>6s}  {'n_days':>6s}  "
          f"{'mean_sales':>10s}  {'disc_std':>8s}  "
          f"{'paper_R2':>9s}  {'hybrid_R2':>9s}  {'alpha':>7s}  {'beta':>7s}")
    print(f"{'='*100}")

    results = []
    plot_data = []

    for idx, (_, row) in enumerate(chosen.iterrows()):
        pid = int(row["product_id"])
        sid = int(row["store_id"])

        sub = df[(df["product_id"] == pid) & (df["store_id"] == sid)].sort_values("dt")
        if len(sub) < 40:
            continue

        X = build_features(sub)
        y = sub["sale_amount"].values.astype(np.float32)

        # time split
        n_tr = int(len(sub) * 0.8)
        X_tr, X_vl = X[:n_tr], X[n_tr:]
        y_tr, y_vl = y[:n_tr], y[n_tr:]

        # scale
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_vl_s = scaler.transform(X_vl)

        # Paper baseline
        lam_p, alpha_p, beta_p, r2p = fit_paper(X_tr_s, y_tr, X_vl_s, y_vl)

        # Hybrid
        n_feat = X_tr_s.shape[1]
        model = TinyHybrid(n_feat).to(DEVICE)
        model = train_tiny(model, X_tr_s, y_tr, X_vl_s, y_vl)
        r2h, _, _ = compute_r2(model, X_vl_s, y_vl)
        alpha_h = float(model.alpha.detach().cpu())
        beta_h = float(model.beta.detach().cpu())

        print(f"{idx+1:3d} {pid:8d} {sid:6d}  {len(sub):6d}  "
              f"{row['mean_sales']:10.2f}  {row['discount_std']:8.3f}  "
              f"{r2p:9.4f}  {r2h:9.4f}  {alpha_h:7.2f}  {beta_h:7.3f}")

        results.append(dict(
            product_id=pid, store_id=sid, n_days=len(sub),
            mean_sales=row["mean_sales"], discount_std=row["discount_std"],
            paper_r2=r2p, paper_lam=lam_p,
            hybrid_r2=r2h, hybrid_alpha=alpha_h, hybrid_beta=beta_h,
        ))

        # keep first 4 for plotting
        if idx < 4:
            _, obs_v, pred_v = compute_r2(model, X_vl_s, y_vl)
            plot_data.append(dict(
                pid=pid, sid=sid, r2h=r2h, r2p=r2p,
                obs=obs_v, pred=pred_v, alpha=alpha_h, beta=beta_h,
                label=f"p={pid} s={sid}"
            ))

    # Summary
    rdf = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(rdf)} combos)")
    print(f"{'='*60}")
    print(f"  Paper R2:   mean={rdf['paper_r2'].mean():.4f}  "
          f"med={rdf['paper_r2'].median():.4f}  "
          f">0: {(rdf['paper_r2']>0).mean()*100:.0f}%")
    print(f"  Hybrid R2:  mean={rdf['hybrid_r2'].mean():.4f}  "
          f"med={rdf['hybrid_r2'].median():.4f}  "
          f">0: {(rdf['hybrid_r2']>0).mean()*100:.0f}%")

    # Plot top 4
    if plot_data:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        for i, pd_item in enumerate(plot_data):
            ax1 = axes[0, i]
            ax1.plot(pd_item["obs"], "b.-", ms=4, alpha=0.7, label="obs")
            ax1.plot(pd_item["pred"], "r.-", ms=4, alpha=0.7, label="pred")
            ax1.set_title(f"{pd_item['label']}\nHybrid R2={pd_item['r2h']:.3f}", fontsize=9)
            if i == 0: ax1.legend(fontsize=7)

            ax2 = axes[1, i]
            ax2.scatter(pd_item["obs"], pd_item["pred"], alpha=0.5, s=10)
            mx = max(pd_item["obs"].max(), pd_item["pred"].max())
            ax2.plot([0, mx], [0, mx], "r--", lw=1)
            ax2.set_xlabel("Observed"); ax2.set_ylabel("Predicted")
            ax2.set_title(f"Paper R2={pd_item['r2p']:.3f}  "
                          f"a={pd_item['alpha']:.1f} b={pd_item['beta']:.2f}", fontsize=9)

        fig.suptitle("Per-product Hybrid Model (single product x single store, "
                     "+avg_wind_level)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "per_product.png"), dpi=150)
        fig.savefig(os.path.join(OUTDIR, "per_product.pdf"))
        plt.close(fig)

    rdf.to_csv(os.path.join(OUTDIR, "per_product_results.csv"), index=False)
    print(f"\nResults -> output/per_product_results.csv")
    print(f"Plots -> output/per_product.pdf")


if __name__ == "__main__":
    main()
