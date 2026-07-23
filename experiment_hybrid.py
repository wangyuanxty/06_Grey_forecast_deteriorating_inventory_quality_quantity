#!/usr/bin/env python3
"""
Hybrid model:  d(t,p,X) = (alpha - beta*p) * NN(features)
  - alpha, beta retain paper's economic interpretation
  - NN replaces e^{-lambda*t} with a universal demand multiplier
  - After training, alpha, beta feed directly into the paper's profit function

Facts: Standalone. Reads FreshRetailNet-50K/{train,eval}.parquet.
Writes output/hybrid_results.csv.
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

# ===== 1. Data =====

def load_data():
    train_df = pd.read_parquet(os.path.join(DATADIR, "train.parquet"))
    eval_df  = pd.read_parquet(os.path.join(DATADIR, "eval.parquet"))
    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    df["day_of_week"] = df["dt"].dt.dayofweek.values.astype(np.float32)
    return df


def extract_cycles(product_df):
    """Split into replenishment cycles (contiguous in-stock blocks)."""
    def stockout_rate(arr):
        if arr is None or len(arr) == 0: return 1.0
        return float(np.mean(arr))

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
        if len(block) >= 3:
            cycles.append(block)
        i = j
    return cycles


def build_features(block):
    """Feature matrix for one cycle. Returns (n_days, n_features)."""
    n = len(block)
    t_rel = np.arange(n, dtype=np.float32)[:, None]          # days since cycle start
    feats = np.column_stack([
        t_rel[:, 0],                                           # 0: relative time
        block["discount"].values.astype(np.float32),            # 1: price proxy
        block["day_of_week"].values.astype(np.float32) / 6.0,   # 2: weekday
        block["holiday_flag"].values.astype(np.float32),        # 3: holiday
        block["activity_flag"].values.astype(np.float32),       # 4: activity/promo
        block["avg_temperature"].values.astype(np.float32)/40,  # 5: temperature
        block["avg_humidity"].values.astype(np.float32)/100,    # 6: humidity
        block["precpt"].values.astype(np.float32)/50,           # 7: precipitation
    ])
    # lag sales
    s = block["sale_amount"].values.astype(np.float32)
    lag1 = np.concatenate([[s[0]], s[:-1]])[:, None]
    lag2 = np.concatenate([[s[0]], [s[0]], s[:-2]])[:, None]
    feats = np.column_stack([feats, lag1[:, 0] / 10, lag2[:, 0] / 10])
    return feats


def build_cycle_dataset(cycles):
    """Convert cycles to flat (X, p, y) arrays for training."""
    X_all, p_all, y_all = [], [], []
    for cyc in cycles:
        f = build_features(cyc)
        p_vals = cyc["discount"].values.astype(np.float32)
        s_vals = cyc["sale_amount"].values.astype(np.float32)
        X_all.append(f)
        p_all.append(p_vals)
        y_all.append(s_vals)
    return X_all, p_all, y_all


# ===== 2. Model =====

class HybridDemandModel(nn.Module):
    """d_pred = (alpha - beta * p) * NN(features)"""
    def __init__(self, n_feat):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(1.0))       # log(alpha)
        self.log_beta  = nn.Parameter(torch.tensor(-1.0))      # log(beta)
        self.net = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8),  nn.ReLU(),
            nn.Linear(8, 1),   nn.Softplus(),                   # > 0 multiplier
        )

    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def beta(self):  return torch.exp(self.log_beta)

    def forward(self, feat, p):
        """feat: (batch, n_feat), p: (batch,)"""
        Dp = self.alpha - self.beta * p                         # price effect
        multiplier = self.net(feat).squeeze(-1)                 # NN multiplier
        return Dp * multiplier


# ===== 3. Training =====

def train_model(model, X_train, p_train, y_train, X_val, p_val, y_val,
                epochs=300, lr=0.005):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=40, factor=0.5)

    best_val_loss = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for X, p_vec, y in zip(X_train, p_train, y_train):
            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            p_t = torch.tensor(p_vec, dtype=torch.float32, device=DEVICE)
            y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
            pred = model(X_t, p_t)
            loss = nn.MSELoss()(pred, y_t)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, p_vec, y in zip(X_val, p_val, y_val):
                X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
                p_t = torch.tensor(p_vec, dtype=torch.float32, device=DEVICE)
                y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
                pred = model(X_t, p_t)
                val_loss += nn.MSELoss()(pred, y_t).item()
        sched.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_loss


def compute_r2(model, X_list, p_list, y_list):
    model.eval()
    all_obs, all_pred = [], []
    with torch.no_grad():
        for X, p_vec, y in zip(X_list, p_list, y_list):
            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            p_t = torch.tensor(p_vec, dtype=torch.float32, device=DEVICE)
            pred = model(X_t, p_t).cpu().numpy()
            all_obs.extend(y.tolist())
            all_pred.extend(pred.tolist())
    obs, pred = np.array(all_obs), np.array(all_pred)
    ssr = np.sum((obs - pred) ** 2); sst = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - ssr / sst if sst > 0 else 0.0, obs, pred


# ===== 4. Paper baseline (for comparison) =====

from scipy.optimize import least_squares

def fit_paper_baseline(X_list, p_list, y_list):
    """Original paper: d = (alpha - beta*p) * exp(-lambda*t)"""
    def residual(theta):
        lam, alpha, beta = theta
        if lam <= 0 or alpha <= 0: return np.full(1000, 1e6)
        res = []
        for X, p_vec, y in zip(X_list, p_list, y_list):
            t_rel = X[:, 0]  # feature 0 = days since cycle start
            Dp = max(alpha - beta * np.mean(p_vec), 0.0)
            for k in range(len(y)):
                tp = max(t_rel[k] - 0.5, 0.0); tk = t_rel[k] + 0.5
                dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * tk))
                res.append(float(y[k]) - dp)
        res.append(0.0)
        return np.array(res)

    td = sum(y.sum() for y in y_list)
    no = sum(len(y) for y in y_list)
    x0 = [0.01, td / no * 2, td / no * 0.1]
    sol = least_squares(residual, x0,
                        bounds=([1e-8, 1e-6, 0], [1.0, 1e6, 1e6]),
                        method="trf", max_nfev=500, ftol=1e-8)
    lam, alpha, beta = sol.x
    # compute R2
    obs_all, pred_all = [], []
    for X, p_vec, y in zip(X_list, p_list, y_list):
        t_rel = X[:, 0]
        Dp = max(alpha - beta * np.mean(p_vec), 0.0)
        for k in range(len(y)):
            tp = max(t_rel[k] - 0.5, 0.0); tk = t_rel[k] + 0.5
            dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * tk))
            obs_all.append(float(y[k])); pred_all.append(float(dp))
    obs, pred = np.array(obs_all), np.array(pred_all)
    ssr = np.sum((obs - pred) ** 2); sst = np.sum((obs - obs.mean()) ** 2)
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    return lam, alpha, beta, r2


# ===== 5. Main =====

def main():
    print("Loading data ...")
    df = load_data()

    combo_stats = df.groupby(["product_id", "store_id"]).agg(
        n_days=("dt", "nunique"),
        mean_sales=("sale_amount", "mean"),
        discount_std=("discount", "std"),
        discount_nunique=("discount", "nunique"),
    ).reset_index()

    good = combo_stats[
        (combo_stats["n_days"] >= 50)
        & (combo_stats["mean_sales"] > 3)
        & (combo_stats["discount_std"] > 0.02)
        & (combo_stats["discount_nunique"] >= 2)
    ]
    print(f"Candidates: {len(good)} combos")

    # find combos with many cycles
    cycle_info = []
    for _, row in good.iterrows():
        pid, sid = int(row["product_id"]), int(row["store_id"])
        sub = df[(df["product_id"] == pid) & (df["store_id"] == sid)]
        cycles = extract_cycles(sub)
        if len(cycles) >= 6:
            cycle_info.append((pid, sid, len(cycles), row["mean_sales"]))
    cycle_df = pd.DataFrame(cycle_info, columns=["pid", "sid", "n_cycles", "mean_sales"])
    chosen = cycle_df.nlargest(5, "n_cycles")
    print(f"Testing {len(chosen)} combos\n")

    results = []
    for combo_idx, (_, crow) in enumerate(chosen.iterrows()):
        pid, sid = int(crow["pid"]), int(crow["sid"])
        sub = df[(df["product_id"] == pid) & (df["store_id"] == sid)]
        cycles = extract_cycles(sub)
        X_all, p_all, y_all = build_cycle_dataset(cycles)

        # train/val split
        n_train = max(int(len(cycles) * 0.8), 4)
        X_tr, p_tr, y_tr = X_all[:n_train], p_all[:n_train], y_all[:n_train]
        X_vl, p_vl, y_vl = X_all[n_train:], p_all[n_train:], y_all[n_train:]

        # scale features
        all_X = np.concatenate(X_tr, axis=0)
        scaler = StandardScaler().fit(all_X)
        X_tr_s = [scaler.transform(x) for x in X_tr]
        X_vl_s = [scaler.transform(x) for x in X_vl]

        n_feat = all_X.shape[1]

        print(f"--- {combo_idx+1}: product={pid}, store={sid}")
        print(f"    cycles={len(cycles)} ({len(X_tr)}t/{len(X_vl)}v), "
              f"sales mean={crow['mean_sales']:.1f}/day")

        # Paper baseline
        t0 = time.time()
        lam_p, alpha_p, beta_p, r2p = fit_paper_baseline(X_tr, p_tr, y_tr)
        # paper R2 on validation
        obs_pv, pred_pv = [], []
        for X, p_vec, y in zip(X_vl, p_vl, y_vl):
            t_rel = X[:, 0]
            Dp = max(alpha_p - beta_p * np.mean(p_vec), 0.0)
            for k in range(len(y)):
                tp = max(t_rel[k] - 0.5, 0.0); tk = t_rel[k] + 0.5
                dp = (Dp / max(lam_p, 1e-8)) * (np.exp(-lam_p * tp) - np.exp(-lam_p * tk))
                obs_pv.append(float(y[k])); pred_pv.append(float(dp))
        obs_a, pred_a = np.array(obs_pv), np.array(pred_pv)
        ssr = np.sum((obs_a - pred_a)**2); sst = np.sum((obs_a - obs_a.mean())**2)
        r2pv = 1.0 - ssr / sst if sst > 0 else 0.0

        print(f"    [Paper]  lam={lam_p:.5f}  alpha={alpha_p:.1f}  beta={beta_p:.1f}  "
              f"R2_val={r2pv:.4f}  ({time.time()-t0:.1f}s)")

        # Hybrid model
        t0 = time.time()
        model = HybridDemandModel(n_feat).to(DEVICE)
        model, best_loss = train_model(model, X_tr_s, p_tr, y_tr, X_vl_s, p_vl, y_vl)
        r2h, obs_h, pred_h = compute_r2(model, X_vl_s, p_vl, y_vl)
        alpha_h = float(model.alpha.detach().cpu())
        beta_h = float(model.beta.detach().cpu())
        print(f"    [Hybrid] alpha={alpha_h:.1f}  beta={beta_h:.1f}  "
              f"R2_val={r2h:.4f}  ({time.time()-t0:.1f}s)\n")

        results.append(dict(
            pid=pid, sid=sid, n_cycles=len(cycles),
            paper_r2=r2pv, paper_lam=lam_p, paper_alpha=alpha_p, paper_beta=beta_p,
            hybrid_r2=r2h, hybrid_alpha=alpha_h, hybrid_beta=beta_h,
        ))

        # plot first combo
        if combo_idx == 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, obs, pred, title in [
                (axes[0], obs_a, pred_a,
                 f"Paper baseline  (R2={r2pv:.3f}, alpha={alpha_p:.1f}, beta={beta_p:.1f})"),
                (axes[1], obs_h, pred_h,
                 f"Hybrid NN  (R2={r2h:.3f}, alpha={alpha_h:.1f}, beta={beta_h:.1f})"),
            ]:
                ax.scatter(obs, pred, alpha=0.5, s=15)
                mx = max(obs.max(), pred.max())
                ax.plot([0, mx], [0, mx], "r--", lw=1)
                ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
                ax.set_title(title, fontsize=11)
            fig.suptitle(f"product={pid} store={sid}", fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, "hybrid_comparison.png"), dpi=150)
            fig.savefig(os.path.join(OUTDIR, "hybrid_comparison.pdf"))
            plt.close(fig)
            print(f"    Plot -> output/hybrid_comparison.pdf\n")

    # Summary
    rdf = pd.DataFrame(results)
    print(f"{'='*60}\nSUMMARY ({len(rdf)} combos)\n{'='*60}")
    for label, r2k, ak, bk in [
        ("Paper  ", "paper_r2", "paper_alpha", "paper_beta"),
        ("Hybrid ", "hybrid_r2", "hybrid_alpha", "hybrid_beta"),
    ]:
        r2s = rdf[r2k].values; al = rdf[ak].values; be = rdf[bk].values
        print(f"  {label}:  R2 mean={np.mean(r2s):.4f} med={np.median(r2s):.4f}  |  "
              f"alpha={np.mean(al):.1f}  beta={np.mean(be):.1f}")
    rdf.to_csv(os.path.join(OUTDIR, "hybrid_results.csv"), index=False)
    print(f"\nResults -> output/hybrid_results.csv")


if __name__ == "__main__":
    main()
