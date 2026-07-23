#!/usr/bin/env python3
"""
Complete pipeline: hourly data → I(t) → lambda → alpha/beta → p*, T*, Q*.

Steps:
  1. Reconstruct I(t) from hours_stock_status + hours_sale
  2. OLS on inventory equation (13a) → lambda
  3. Hybrid model → alpha, beta
  4. Profit function (19) → optimal price, cycle, order quantity

Facts: Standalone. Reads FreshRetailNet-50K/{train,eval}.parquet.
Writes output/full_pipeline_results.csv.
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATADIR = os.path.join(os.path.dirname(__file__), "FreshRetailNet-50K")
OUTDIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTDIR, exist_ok=True)
print(f"Device: {DEVICE}")


# ===== 1. I(t) reconstruction =====

def reconstruct_inventory(product_df):
    """Reconstruct daily I(t) from hourly stock_status + sales."""
    product_df = product_df.sort_values("dt").reset_index(drop=True)
    n = len(product_df)
    if n < 3:
        return None

    hours_sale   = product_df["hours_sale"].values
    hours_status = product_df["hours_stock_status"].values
    discount     = product_df["discount"].values.astype(float)
    d_daily      = product_df["sale_amount"].values.astype(float)

    stockout_hour = np.full(n, 24, dtype=int)
    sales_before_so = np.zeros(n)

    for k in range(n):
        st = np.array(hours_status[k], dtype=float)
        hs = np.array(hours_sale[k], dtype=float)
        idx = np.where(st > 0.5)[0]
        if len(idx) > 0:
            stockout_hour[k] = idx[0]
        sales_before_so[k] = float(np.sum(hs[:stockout_hour[k]]))

    # backwards reconstruction
    I_open = np.zeros(n)
    I_open[-1] = max(sales_before_so[-1], 0.01)

    for k in range(n - 2, -1, -1):
        if stockout_hour[k] < 24:
            I_open[k] = max(sales_before_so[k], I_open[k + 1] + d_daily[k])
        else:
            I_open[k] = I_open[k + 1] + d_daily[k]

    # inventory change (positive = decreased)
    l = np.zeros(n)
    for k in range(n - 1):
        l[k] = max(0.0, I_open[k] - I_open[k + 1])
    l[-1] = I_open[-1]

    # correct target: remove mechanical discount
    disc_clipped = np.clip(discount, 0.1, 1.5)
    y_corrected = d_daily / disc_clipped

    Q = I_open[0]

    return dict(
        n=n,
        I=I_open,
        l=l,
        d_raw=d_daily,
        d_corrected=y_corrected,
        discount=disc_clipped,
        Q=Q,
    )


# ===== 2. Inventory equation OLS → lambda =====

def estimate_lambda(cycles):
    """Eq (13a): l_j + d_j = -lambda * 0.5*(I_j + I_{j-1})*dt"""
    H, Y = [], []
    for cyc in cycles:
        I, l, d = cyc["I"], cyc["l"], cyc["d_raw"]
        for k in range(1, cyc["n"]):
            X = -0.5 * (I[k] + I[k - 1]) * 1.0
            y = l[k] + d[k]
            if abs(X) > 1e-6:
                H.append(X)
                Y.append(y)
    if len(H) < 3:
        return 0.0, 0
    H, Y = np.array(H), np.array(Y)
    lam = max(float(np.dot(H, Y) / np.dot(H, H)), 0.0)
    return lam, len(H)


# ===== 3. Hybrid model (same as single-store) =====

class SingleStoreModel(nn.Module):
    def __init__(self, n_numeric, n_products, n_cats, emb_dim=8):
        super().__init__()
        self.log_alpha = nn.Embedding(n_products, 1)
        self.log_beta  = nn.Embedding(n_products, 1)
        self.prod_emb = nn.Embedding(n_products, emb_dim)
        self.cat_emb   = nn.Embedding(n_cats, emb_dim // 2)

        total_in = n_numeric + emb_dim + emb_dim // 2
        self.net = nn.Sequential(
            nn.Linear(total_in, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Softplus(),
        )
        nn.init.constant_(self.log_alpha.weight, 1.5)
        nn.init.constant_(self.log_beta.weight, -1.5)

    def forward(self, numeric, ids):
        prod_id = ids[:, 0]; cat_id = ids[:, 1]
        alpha = torch.exp(self.log_alpha(prod_id)).squeeze(-1)
        beta  = torch.exp(self.log_beta(prod_id)).squeeze(-1)
        price = numeric[:, 1]
        Dp = alpha - beta * price
        p_emb = self.prod_emb(prod_id)
        c_emb = self.cat_emb(cat_id)
        net_in = torch.cat([numeric, p_emb, c_emb], dim=-1)
        multiplier = self.net(net_in).squeeze(-1)
        return Dp * multiplier


# ===== 4. Profit optimization =====

def profit(p, T, lam, alpha, beta, c, h, K):
    """Paper eq (19): profit per unit time."""
    Dp = alpha - beta * p
    if Dp <= 0:
        return -1e6
    par1 = Dp * (p * lam + h) / (lam ** 2 + 1e-8)
    par2 = Dp * (c * lam + h) / (lam + 1e-8)
    exp_term = 1.0 - np.exp(-lam * T)
    return float(par1 * exp_term / T - par2 - K / T)


def find_optimal(lam, alpha, beta, c, h, K):
    """Find (p*, T*) that maximize profit."""
    def objective(x):
        p, T = x[0], x[1]
        return -profit(p, T, lam, alpha, beta, c, h, K)

    # bounds: c <= p <= alpha/beta, 0.5 <= T <= 30
    p_max = min(alpha / (beta + 1e-8), 50.0)
    bounds = [(c + 0.01, p_max), (0.5, 30.0)]

    # initial guess
    p0 = (c + p_max) / 2
    T0 = 4.0

    res = minimize(objective, [p0, T0], bounds=bounds, method="L-BFGS-B")
    p_opt, T_opt = res.x
    Q_opt = (alpha - beta * p_opt) * T_opt
    P_opt = -res.fun
    return p_opt, T_opt, Q_opt, P_opt


# ===== 5. Main =====

def main():
    print("Loading data ...")
    train_df = pd.read_parquet(os.path.join(DATADIR, "train.parquet"))
    eval_df  = pd.read_parquet(os.path.join(DATADIR, "eval.parquet"))
    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    df["day_of_week"] = df["dt"].dt.dayofweek.values.astype(np.float32)

    # pick best store
    store_sizes = df.groupby("store_id").size().sort_values(ascending=False)
    best_store = store_sizes.index[0]
    df = df[df["store_id"] == best_store].copy()
    print(f"Store: {best_store}, {len(df):,} rows")

    # filter products
    prod_stats = df.groupby("product_id").agg(
        n_days=("dt", "nunique"),
        mean_sales=("sale_amount", "mean"),
        discount_std=("discount", "std"),
    ).reset_index()
    keep = prod_stats[(prod_stats["n_days"] >= 30)
                      & (prod_stats["mean_sales"] > 1)
                      & (prod_stats["discount_std"] > 0.01)]["product_id"]
    df = df[df["product_id"].isin(keep)].sort_values(["product_id", "dt"]).reset_index(drop=True)
    print(f"Products: {df['product_id'].nunique()}, {len(df):,} rows")

    # ---- Step 1 & 2: reconstruct I(t) + estimate lambda per product ----
    print("\nStep 1-2: I(t) reconstruction + lambda estimation ...")
    inv_data = {}
    for pid in df["product_id"].unique():
        sub = df[df["product_id"] == pid]
        cyc = reconstruct_inventory(sub)
        if cyc is not None:
            lam, n_pts = estimate_lambda([cyc])
            inv_data[pid] = dict(cyc=cyc, lam=lam, n_pts=n_pts)

    products_with_I = list(inv_data.keys())
    print(f"  Products with valid I(t): {len(products_with_I)}")

    # lambda stats
    lams = [inv_data[p]["lam"] for p in products_with_I]
    print(f"  lambda: mean={np.mean(lams):.5f} med={np.median(lams):.5f} "
          f"P10={np.percentile(lams,10):.5f} P90={np.percentile(lams,90):.5f}")
    print(f"  lambda > 0: {sum(1 for l in lams if l > 1e-6)}/{len(lams)}")

    # ---- Step 3: Hybrid model for alpha, beta ----
    print("\nStep 3: Hybrid model for alpha, beta ...")

    # prepare dataset
    df_inv = df[df["product_id"].isin(products_with_I)].copy()
    df_inv["t_first"] = df_inv.groupby("product_id")["dt"].transform("min")
    df_inv["t_rel"] = (df_inv["dt"] - df_inv["t_first"]).dt.days.astype(np.float32)

    dates = sorted(df_inv["dt"].unique())
    split_date = dates[int(len(dates) * 0.8)]
    train_df2 = df_inv[df_inv["dt"] <= split_date].copy()
    val_df2   = df_inv[df_inv["dt"] > split_date].copy()

    def build_features_std(df_sub):
        numeric = np.column_stack([
            df_sub["t_rel"].values.astype(np.float32),
            df_sub["discount"].values.astype(np.float32),
            df_sub["day_of_week"].values.astype(np.float32) / 6.0,
            df_sub["holiday_flag"].values.astype(np.float32),
            df_sub["activity_flag"].values.astype(np.float32),
            df_sub["avg_temperature"].values.astype(np.float32) / 40.0,
            df_sub["avg_humidity"].values.astype(np.float32) / 100.0,
            df_sub["precpt"].values.astype(np.float32) / 50.0,
            df_sub["avg_wind_level"].values.astype(np.float32) / 10.0,
        ])
        ids = np.column_stack([
            df_sub["product_id"].values.astype(np.int64),
            df_sub["first_category_id"].values.astype(np.int64),
        ])
        return numeric, ids

    X_num_tr, X_id_tr = build_features_std(train_df2)
    X_num_vl, X_id_vl = build_features_std(val_df2)

    # corrected target
    disc_tr = np.clip(train_df2["discount"].values.astype(np.float32), 0.1, 1.5)
    disc_vl = np.clip(val_df2["discount"].values.astype(np.float32), 0.1, 1.5)
    y_tr = train_df2["sale_amount"].values.astype(np.float32) / disc_tr
    y_vl = val_df2["sale_amount"].values.astype(np.float32) / disc_vl

    scaler = StandardScaler().fit(X_num_tr)
    X_tr_s = scaler.transform(X_num_tr)
    X_vl_s = scaler.transform(X_num_vl)

    n_products = df_inv["product_id"].max() + 1
    n_cats     = df_inv["first_category_id"].max() + 1

    class RetailDS(Dataset):
        def __init__(self, X, ids, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.ids = torch.tensor(ids, dtype=torch.long)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.ids[i], self.y[i]

    tr_loader = DataLoader(RetailDS(X_tr_s, X_id_tr, y_tr), batch_size=2048, shuffle=True)
    vl_loader = DataLoader(RetailDS(X_vl_s, X_id_vl, y_vl), batch_size=2048)

    model = SingleStoreModel(9, n_products, n_cats).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    best_vl = float("inf"); best_state = None
    for ep in range(80):
        model.train()
        for X, ids, y in tr_loader:
            X, ids, y = X.to(DEVICE), ids.to(DEVICE), y.to(DEVICE)
            loss = nn.MSELoss()(model(X, ids), y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for X, ids, y in vl_loader:
                X, ids, y = X.to(DEVICE), ids.to(DEVICE), y.to(DEVICE)
                vl_loss += nn.MSELoss()(model(X, ids), y).item()
        sched.step(vl_loss)
        if vl_loss < best_vl:
            best_vl = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == 79:
            print(f"    epoch {ep:3d}: val_mse={vl_loss/len(vl_loader):.4f}")

    model.load_state_dict(best_state)

    # eval R2
    model.eval()
    obs_all, pred_all = [], []
    with torch.no_grad():
        for X, ids, y in vl_loader:
            X, ids = X.to(DEVICE), ids.to(DEVICE)
            pred = model(X, ids).cpu().numpy()
            obs_all.extend(y.numpy().tolist())
            pred_all.extend(pred.tolist())
    obs, pred = np.array(obs_all), np.array(pred_all)
    ssr = np.sum((obs - pred) ** 2); sst = np.sum((obs - obs.mean()) ** 2)
    r2_val = 1.0 - ssr / sst if sst > 0 else 0.0
    print(f"\n  R2 val: {r2_val:.4f}")

    # extract alpha, beta per product
    alpha_all = torch.exp(model.log_alpha.weight).detach().cpu().numpy().flatten()
    beta_all  = torch.exp(model.log_beta.weight).detach().cpu().numpy().flatten()

    # ---- Step 4: Profit optimization per product ----
    print("\nStep 4: Profit optimization ...")
    # cost parameters (paper's values, normalized to our scale)
    # Paper: c=6, h=0.02, K=100, alpha≈120, Q≈140
    # Our alpha≈4.4, so scale factor ≈ 120/4.4 ≈ 27
    # Use reasonable defaults
    c = 1.0        # unit cost (normalized)
    h = 0.005      # holding cost per unit per day
    K = 2.0        # fixed ordering cost

    print(f"  Cost params: c={c}, h={h}, K={K}")

    results = []
    for pid in products_with_I:
        lam = inv_data[pid]["lam"]
        alpha = float(alpha_all[pid])
        beta  = float(beta_all[pid])
        Q_recon = inv_data[pid]["cyc"]["Q"]
        n_pts   = inv_data[pid]["n_pts"]
        disc_mean = float(inv_data[pid]["cyc"]["discount"].mean())

        if lam < 1e-6:
            lam = 0.001  # minimum for profit function numerical stability

        p_opt, T_opt, Q_opt, P_opt = find_optimal(lam, alpha, beta, c, h, K)

        results.append(dict(
            product_id=pid,
            lam=lam,
            n_I_pts=n_pts,
            alpha=alpha,
            beta=beta,
            Q_reconstructed=Q_recon,
            discount_mean=disc_mean,
            p_opt=p_opt,
            T_opt=T_opt,
            Q_opt=Q_opt,
            profit=P_opt,
        ))

    # ---- Summary ----
    rdf = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"FULL PIPELINE RESULTS ({len(rdf)} products)")
    print(f"{'='*80}")

    cols = ["lam", "alpha", "beta", "Q_reconstructed", "p_opt", "T_opt", "Q_opt", "profit"]
    for c_name in cols:
        v = rdf[c_name].values
        print(f"  {c_name:20s}: mean={np.mean(v):8.4f}  med={np.median(v):8.4f}  "
              f"[{np.min(v):.4f}, {np.max(v):.4f}]")

    print(f"\n  Products with profit > 0: {(rdf['profit']>0).sum()}/{len(rdf)}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # lambda distribution
    axes[0,0].hist(rdf["lam"], bins=30, color="steelblue", edgecolor="white")
    axes[0,0].axvline(rdf["lam"].median(), color="red", lw=2, ls="--")
    axes[0,0].set_title(f"lambda (med={rdf['lam'].median():.4f})")

    # alpha vs beta
    axes[0,1].scatter(rdf["alpha"], rdf["beta"], alpha=0.5, s=15)
    axes[0,1].set_xlabel("alpha"); axes[0,1].set_ylabel("beta")
    axes[0,1].set_title("alpha vs beta per product")

    # Q reconstructed vs Q optimal
    axes[0,2].scatter(rdf["Q_reconstructed"], rdf["Q_opt"], alpha=0.5, s=15)
    axes[0,2].plot([0, rdf["Q_reconstructed"].max()], [0, rdf["Q_reconstructed"].max()],
                   "r--", lw=1)
    axes[0,2].set_xlabel("Q reconstructed"); axes[0,2].set_ylabel("Q optimal")
    axes[0,2].set_title("Reconstructed vs Optimal Q")

    # price optimization
    axes[1,0].hist(rdf["p_opt"], bins=20, color="darkorange", edgecolor="white")
    axes[1,0].axvline(rdf["p_opt"].median(), color="red", lw=2, ls="--")
    axes[1,0].set_title(f"Optimal price (med={rdf['p_opt'].median():.2f})")

    # cycle optimization
    axes[1,1].hist(rdf["T_opt"], bins=20, color="green", edgecolor="white", alpha=0.7)
    axes[1,1].axvline(rdf["T_opt"].median(), color="red", lw=2, ls="--")
    axes[1,1].set_title(f"Optimal cycle (med={rdf['T_opt'].median():.1f} days)")

    # profit distribution
    axes[1,2].hist(rdf["profit"], bins=20, color="purple", edgecolor="white", alpha=0.7)
    axes[1,2].axvline(0, color="black", lw=1)
    axes[1,2].set_title(f"Profit (med={rdf['profit'].median():.2f})")

    fig.suptitle(f"Full Pipeline: hourly data → lambda → alpha/beta → (p*, T*, Q*)\n"
                 f"Store {best_store}, {len(rdf)} products, R^2_val={r2_val:.3f}",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "full_pipeline.png"), dpi=150)
    fig.savefig(os.path.join(OUTDIR, "full_pipeline.pdf"))
    plt.close(fig)

    rdf.to_csv(os.path.join(OUTDIR, "full_pipeline_results.csv"), index=False)
    print(f"\nPlots -> output/full_pipeline.pdf")
    print(f"Results -> output/full_pipeline_results.csv")
    print("Done.")


if __name__ == "__main__":
    main()
