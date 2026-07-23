#!/usr/bin/env python3
"""
Hybrid v2:  d(t,p,X) = (alpha - beta*p) * NN(features)

Changes from v1:
  - Remove lag1, lag2
  - Add product_id, store_id, category_id, city_id as embeddings
  - Train on ALL combos together (时间切分，不再按 combo 独立)
  - alpha, beta per product (865个), learned via embedding lookup

Facts: Standalone. Reads FreshRetailNet-50K/{train,eval}.parquet.
Writes output/hybrid_v2_results.csv.
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# ===== 1. Data preparation =====

def load_and_prepare():
    train_df = pd.read_parquet(os.path.join(DATADIR, "train.parquet"))
    eval_df  = pd.read_parquet(os.path.join(DATADIR, "eval.parquet"))
    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    df["day_of_week"] = df["dt"].dt.dayofweek.values.astype(np.float32)

    # filter: reasonable sales, price variation, enough days per combo
    combo = df.groupby(["product_id", "store_id"]).agg(
        n_days=("dt", "nunique"),
        mean_sales=("sale_amount", "mean"),
        discount_std=("discount", "std"),
    ).reset_index()
    keep = combo[(combo["n_days"] >= 30) & (combo["mean_sales"] > 1)
                 & (combo["discount_std"] > 0.01)]
    df = df.merge(keep[["product_id", "store_id"]], on=["product_id", "store_id"])
    print(f"Filtered: {len(df):,} rows, "
          f"{df['product_id'].nunique()} products x {df['store_id'].nunique()} stores")

    # relative time within each combo (days since first appearance)
    df = df.sort_values(["product_id", "store_id", "dt"]).reset_index(drop=True)
    df["t_first"] = df.groupby(["product_id", "store_id"])["dt"].transform("min")
    df["t_rel"] = (df["dt"] - df["t_first"]).dt.days.astype(np.float32)

    return df


def build_features(df):
    """Build feature matrix. No lag features. IDs as raw integers for embedding."""
    numeric = np.column_stack([
        df["t_rel"].values.astype(np.float32),                    # 0: days since first obs
        df["discount"].values.astype(np.float32),                  # 1: price proxy
        df["day_of_week"].values.astype(np.float32) / 6.0,         # 2: weekday
        df["holiday_flag"].values.astype(np.float32),              # 3: holiday
        df["activity_flag"].values.astype(np.float32),             # 4: promo
        df["avg_temperature"].values.astype(np.float32) / 40.0,    # 5: temp
        df["avg_humidity"].values.astype(np.float32) / 100.0,      # 6: humidity
        df["precpt"].values.astype(np.float32) / 50.0,             # 7: precip
    ])
    ids = np.column_stack([
        df["product_id"].values.astype(np.int64),                  # 8
        df["store_id"].values.astype(np.int64),                    # 9
        df["first_category_id"].values.astype(np.int64),           # 10
        df["city_id"].values.astype(np.int64),                     # 11
    ])
    return numeric, ids


class RetailDataset(Dataset):
    def __init__(self, numeric, ids, prices, targets, scaler=None):
        if scaler is None:
            self.scaler = StandardScaler().fit(numeric)
        else:
            self.scaler = scaler
        self.numeric = torch.tensor(self.scaler.transform(numeric), dtype=torch.float32)
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.prices = torch.tensor(prices, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self): return len(self.numeric)

    def __getitem__(self, idx):
        return (self.numeric[idx], self.ids[idx], self.prices[idx], self.targets[idx])


# ===== 2. Model =====

class HybridV2(nn.Module):
    def __init__(self, n_numeric, n_products, n_stores, n_cats, n_cities,
                 emb_dim=8):
        super().__init__()
        # per-product alpha, beta (log-space for positivity)
        self.log_alpha = nn.Embedding(n_products, 1)
        self.log_beta  = nn.Embedding(n_products, 1)

        # ID embeddings
        self.prod_emb = nn.Embedding(n_products, emb_dim)
        self.store_emb = nn.Embedding(n_stores, emb_dim)
        self.cat_emb   = nn.Embedding(n_cats, emb_dim // 2)
        self.city_emb  = nn.Embedding(n_cities, emb_dim // 2)

        # NN: numeric + embeddings → demand multiplier
        total_in = n_numeric + 3 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(total_in, 128), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Softplus(),
        )

        # init
        nn.init.constant_(self.log_alpha.weight, 1.5)   # alpha ≈ 4.5
        nn.init.constant_(self.log_beta.weight, -1.5)    # beta ≈ 0.22

    def forward(self, numeric, ids):
        """ids: (batch, 4) = [product_id, store_id, cat_id, city_id]"""
        prod_id   = ids[:, 0]; store_id  = ids[:, 1]
        cat_id    = ids[:, 2]; city_id   = ids[:, 3]

        alpha = torch.exp(self.log_alpha(prod_id)).squeeze(-1)
        beta  = torch.exp(self.log_beta(prod_id)).squeeze(-1)
        price = numeric[:, 1]  # discount column

        Dp = alpha - beta * price

        # embeddings
        p_emb = self.prod_emb(prod_id)
        s_emb = self.store_emb(store_id)
        c_emb = self.cat_emb(cat_id)
        ct_emb = self.city_emb(city_id)

        net_in = torch.cat([numeric, p_emb, s_emb, c_emb, ct_emb], dim=-1)
        multiplier = self.net(net_in).squeeze(-1)

        return Dp * multiplier


# ===== 3. Training =====

def train_model(model, train_loader, val_loader, epochs=80, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=15, factor=0.5)

    best_val = float("inf"); best_state = None

    for ep in range(epochs):
        model.train(); train_loss = 0.0; n_batch = 0
        for num, ids, prices, targets in train_loader:
            num, ids, targets = num.to(DEVICE), ids.to(DEVICE), targets.to(DEVICE)
            pred = model(num, ids)
            loss = nn.MSELoss()(pred, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item(); n_batch += 1

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for num, ids, prices, targets in val_loader:
                num, ids, targets = num.to(DEVICE), ids.to(DEVICE), targets.to(DEVICE)
                pred = model(num, ids)
                val_loss += nn.MSELoss()(pred, targets).item()

        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}:  train_loss={train_loss/n_batch:.4f}  "
                  f"val_loss={val_loss/len(val_loader):.4f}")

    model.load_state_dict(best_state)
    return model, best_val


def compute_r2(model, loader):
    model.eval()
    all_obs, all_pred = [], []
    with torch.no_grad():
        for num, ids, prices, targets in loader:
            num, ids = num.to(DEVICE), ids.to(DEVICE)
            pred = model(num, ids).cpu().numpy()
            all_obs.extend(targets.numpy().tolist())
            all_pred.extend(pred.tolist())
    obs, pred = np.array(all_obs), np.array(all_pred)
    ssr = np.sum((obs - pred) ** 2); sst = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - ssr / sst if sst > 0 else 0.0, obs, pred


# ===== 4. Main =====

def main():
    print("Loading & preparing ...")
    df = load_and_prepare()

    # time-based split: first 80% days train, last 20% val
    dates = sorted(df["dt"].unique())
    split_date = dates[int(len(dates) * 0.8)]
    train_df = df[df["dt"] <= split_date].copy()
    val_df   = df[df["dt"] > split_date].copy()
    print(f"Train: {len(train_df):,} rows ({train_df['dt'].min().date()} ~ "
          f"{train_df['dt'].max().date()})")
    print(f"Val:   {len(val_df):,} rows ({val_df['dt'].min().date()} ~ "
          f"{val_df['dt'].max().date()})")

    # build features
    X_num_tr, X_id_tr = build_features(train_df)
    X_num_vl, X_id_vl = build_features(val_df)
    y_tr = train_df["sale_amount"].values.astype(np.float32)
    y_vl = val_df["sale_amount"].values.astype(np.float32)
    p_tr = train_df["discount"].values.astype(np.float32)
    p_vl = val_df["discount"].values.astype(np.float32)

    # ID ranges
    n_products = df["product_id"].max() + 1
    n_stores   = df["store_id"].max() + 1
    n_cats     = df["first_category_id"].max() + 1
    n_cities   = df["city_id"].max() + 1
    print(f"Embeddings: {n_products} products, {n_stores} stores, "
          f"{n_cats} cats, {n_cities} cities")
    print(f"Numeric features: {X_num_tr.shape[1]}")

    # datasets
    train_ds = RetailDataset(X_num_tr, X_id_tr, p_tr, y_tr)
    val_ds   = RetailDataset(X_num_vl, X_id_vl, p_vl, y_vl, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)

    # model
    model = HybridV2(
        n_numeric=X_num_tr.shape[1],
        n_products=n_products, n_stores=n_stores,
        n_cats=n_cats, n_cities=n_cities,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # train
    print("\nTraining ...")
    t0 = time.time()
    model, best_loss = train_model(model, train_loader, val_loader)
    print(f"Training time: {time.time()-t0:.0f}s")

    # evaluate
    r2_train, obs_tr, pred_tr = compute_r2(model, train_loader)
    r2_val, obs_vl, pred_vl = compute_r2(model, val_loader)
    print(f"\nR2 train: {r2_train:.4f}")
    print(f"R2 val:   {r2_val:.4f}")

    # alpha, beta stats
    alpha_vals = torch.exp(model.log_alpha.weight).detach().cpu().numpy().flatten()
    beta_vals  = torch.exp(model.log_beta.weight).detach().cpu().numpy().flatten()
    # only show products that appear in data
    active_products = np.unique(train_df["product_id"].values)
    alpha_a = alpha_vals[active_products]
    beta_a  = beta_vals[active_products]
    print(f"\nalpha: mean={np.mean(alpha_a):.2f} med={np.median(alpha_a):.2f} "
          f"p10={np.percentile(alpha_a,10):.2f} p90={np.percentile(alpha_a,90):.2f}")
    print(f"beta:  mean={np.mean(beta_a):.3f} med={np.median(beta_a):.3f} "
          f"p10={np.percentile(beta_a,10):.3f} p90={np.percentile(beta_a,90):.3f}")

    # ===== 5. Plots =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # scatter: observed vs predicted
    ax = axes[0]
    sample = np.random.choice(len(obs_vl), min(5000, len(obs_vl)), replace=False)
    ax.scatter(obs_vl[sample], pred_vl[sample], alpha=0.3, s=8)
    mx = max(obs_vl[sample].max(), pred_vl[sample].max())
    ax.plot([0, mx], [0, mx], "r--", lw=1.5)
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    ax.set_title(f"Validation  (R2={r2_val:.3f}, n={len(obs_vl):,})")

    # alpha distribution
    ax = axes[1]
    ax.hist(alpha_a, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(alpha_a), color="red", lw=2, ls="--",
               label=f"median={np.median(alpha_a):.2f}")
    ax.set_xlabel("alpha"); ax.set_title("Alpha distribution")
    ax.legend()

    # beta distribution
    ax = axes[2]
    ax.hist(beta_a, bins=50, color="darkorange", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(beta_a), color="red", lw=2, ls="--",
               label=f"median={np.median(beta_a):.3f}")
    ax.set_xlabel("beta"); ax.set_title("Beta distribution")
    ax.legend()

    fig.suptitle("Hybrid v2: NN demand with product/store/city embeddings\n"
                 f"{len(active_products)} products, {len(train_df):,} train rows, "
                 f"{len(val_df):,} val rows",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "hybrid_v2.png"), dpi=150)
    fig.savefig(os.path.join(OUTDIR, "hybrid_v2.pdf"))
    plt.close(fig)
    print(f"\nPlots -> output/hybrid_v2.pdf")

    # save
    pd.DataFrame(dict(alpha=alpha_a, beta=beta_a)).to_csv(
        os.path.join(OUTDIR, "hybrid_v2_alpha_beta.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
