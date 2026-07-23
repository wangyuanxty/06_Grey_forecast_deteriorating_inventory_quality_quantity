#!/usr/bin/env python3
"""
Single-store model: pick the store with most data, train on all its products.
No store_id needed — all products share the same customer base.

d(t,p,X) = (alpha - beta*p) * NN(t, features, product_emb, category_emb)

Facts: Standalone. Reads FreshRetailNet-50K/{train,eval}.parquet.
Writes output/single_store_results.csv.
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


# ===== 1. Data =====

def load_and_prepare():
    train_df = pd.read_parquet(os.path.join(DATADIR, "train.parquet"))
    eval_df  = pd.read_parquet(os.path.join(DATADIR, "eval.parquet"))
    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    df["day_of_week"] = df["dt"].dt.dayofweek.values.astype(np.float32)

    # pick the store with the most product-days
    store_sizes = df.groupby("store_id").size().sort_values(ascending=False)
    best_store = store_sizes.index[0]
    print(f"Best store: {best_store} ({store_sizes.iloc[0]:,} rows)")

    df = df[df["store_id"] == best_store].copy()

    # filter products with >= 30 days and some price variation
    prod_stats = df.groupby("product_id").agg(
        n_days=("dt", "nunique"),
        mean_sales=("sale_amount", "mean"),
        discount_std=("discount", "std"),
    ).reset_index()
    keep_prods = prod_stats[(prod_stats["n_days"] >= 30)
                            & (prod_stats["mean_sales"] > 1)
                            & (prod_stats["discount_std"] > 0.01)]["product_id"]
    df = df[df["product_id"].isin(keep_prods)]
    print(f"After filtering: {len(df):,} rows, "
          f"{df['product_id'].nunique()} products")

    df = df.sort_values(["product_id", "dt"]).reset_index(drop=True)
    df["t_first"] = df.groupby("product_id")["dt"].transform("min")
    df["t_rel"] = (df["dt"] - df["t_first"]).dt.days.astype(np.float32)

    return df, int(best_store)


def build_features(df):
    numeric = np.column_stack([
        df["t_rel"].values.astype(np.float32),
        df["discount"].values.astype(np.float32),
        df["day_of_week"].values.astype(np.float32) / 6.0,
        df["holiday_flag"].values.astype(np.float32),
        df["activity_flag"].values.astype(np.float32),
        df["avg_temperature"].values.astype(np.float32) / 40.0,
        df["avg_humidity"].values.astype(np.float32) / 100.0,
        df["precpt"].values.astype(np.float32) / 50.0,
    ])
    ids = np.column_stack([
        df["product_id"].values.astype(np.int64),
        df["first_category_id"].values.astype(np.int64),
    ])
    return numeric, ids


class RetailDataset(Dataset):
    def __init__(self, numeric, ids, targets, scaler=None):
        if scaler is None:
            self.scaler = StandardScaler().fit(numeric)
        else:
            self.scaler = scaler
        self.numeric = torch.tensor(self.scaler.transform(numeric), dtype=torch.float32)
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self): return len(self.numeric)

    def __getitem__(self, idx):
        return (self.numeric[idx], self.ids[idx], self.targets[idx])


# ===== 2. Model =====

class SingleStoreModel(nn.Module):
    def __init__(self, n_numeric, n_products, n_cats, emb_dim=8):
        super().__init__()
        # per-product alpha, beta
        self.log_alpha = nn.Embedding(n_products, 1)
        self.log_beta  = nn.Embedding(n_products, 1)

        # embeddings (no store_id)
        self.prod_emb = nn.Embedding(n_products, emb_dim)
        self.cat_emb   = nn.Embedding(n_cats, emb_dim // 2)

        total_in = n_numeric + emb_dim + emb_dim // 2
        self.net = nn.Sequential(
            nn.Linear(total_in, 128), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.1),
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


# ===== 3. Training =====

def train_model(model, train_loader, val_loader, epochs=80, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    best_val = float("inf"); best_state = None

    for ep in range(epochs):
        model.train(); train_loss = 0.0; n_batch = 0
        for num, ids, targets in train_loader:
            num, ids, targets = num.to(DEVICE), ids.to(DEVICE), targets.to(DEVICE)
            pred = model(num, ids)
            loss = nn.MSELoss()(pred, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item(); n_batch += 1

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for num, ids, targets in val_loader:
                num, ids, targets = num.to(DEVICE), ids.to(DEVICE), targets.to(DEVICE)
                pred = model(num, ids)
                val_loss += nn.MSELoss()(pred, targets).item()

        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}:  train_loss={train_loss/n_batch:.4f}  "
                  f"val_mse={val_loss/len(val_loader):.4f}")

    model.load_state_dict(best_state)
    return model, best_val


def compute_r2(model, loader):
    model.eval()
    all_obs, all_pred = [], []
    with torch.no_grad():
        for num, ids, targets in loader:
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
    df, store_id = load_and_prepare()

    dates = sorted(df["dt"].unique())
    split_date = dates[int(len(dates) * 0.8)]
    train_df = df[df["dt"] <= split_date].copy()
    val_df   = df[df["dt"] > split_date].copy()
    print(f"Train: {len(train_df):,} rows ({train_df['dt'].min().date()} ~ "
          f"{train_df['dt'].max().date()})")
    print(f"Val:   {len(val_df):,} rows ({val_df['dt'].min().date()} ~ "
          f"{val_df['dt'].max().date()})")

    X_num_tr, X_id_tr = build_features(train_df)
    X_num_vl, X_id_vl = build_features(val_df)
    # Corrected target: remove mechanical discount effect
    # sale_amount = units * base_price * discount
    # sale_amount / discount = units * base_price  (behavioral only)
    disc_tr = np.clip(train_df["discount"].values.astype(np.float32), 0.1, 1.5)
    disc_vl = np.clip(val_df["discount"].values.astype(np.float32), 0.1, 1.5)
    y_tr = (train_df["sale_amount"].values.astype(np.float32) / disc_tr)
    y_vl = (val_df["sale_amount"].values.astype(np.float32) / disc_vl)

    n_products = df["product_id"].max() + 1
    n_cats     = df["first_category_id"].max() + 1
    print(f"Products: {df['product_id'].nunique()}, Cats: {n_cats}, "
          f"Numeric: {X_num_tr.shape[1]}")

    train_ds = RetailDataset(X_num_tr, X_id_tr, y_tr)
    val_ds   = RetailDataset(X_num_vl, X_id_vl, y_vl, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=2048, shuffle=False)

    model = SingleStoreModel(X_num_tr.shape[1], n_products, n_cats).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining ...")
    t0 = time.time()
    model, _ = train_model(model, train_loader, val_loader)
    print(f"Time: {time.time()-t0:.0f}s")

    r2_tr, _, _ = compute_r2(model, train_loader)
    r2_vl, obs_v, pred_v = compute_r2(model, val_loader)
    print(f"\nR2 train: {r2_tr:.4f}")
    print(f"R2 val:   {r2_vl:.4f}")

    # alpha, beta stats
    active = np.unique(train_df["product_id"].values)
    alpha_vals = torch.exp(model.log_alpha.weight).detach().cpu().numpy().flatten()
    beta_vals  = torch.exp(model.log_beta.weight).detach().cpu().numpy().flatten()
    aa, ba = alpha_vals[active], beta_vals[active]
    print(f"alpha: mean={np.mean(aa):.2f} med={np.median(aa):.2f} "
          f"p10={np.percentile(aa,10):.2f} p90={np.percentile(aa,90):.2f}")
    print(f"beta:  mean={np.mean(ba):.3f} med={np.median(ba):.3f} "
          f"p10={np.percentile(ba,10):.3f} p90={np.percentile(ba,90):.3f}")

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    samp = np.random.choice(len(obs_v), min(5000, len(obs_v)), replace=False)
    axes[0].scatter(obs_v[samp], pred_v[samp], alpha=0.3, s=8)
    mx = max(obs_v[samp].max(), pred_v[samp].max())
    axes[0].plot([0, mx], [0, mx], "r--", lw=1.5)
    axes[0].set_xlabel("Observed"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Store {store_id}, val R2={r2_vl:.3f}")

    axes[1].hist(aa, bins=40, color="steelblue", edgecolor="white")
    axes[1].axvline(np.median(aa), color="red", lw=2, ls="--",
                    label=f"median={np.median(aa):.2f}")
    axes[1].set_xlabel("alpha"); axes[1].set_title("Alpha by product"); axes[1].legend()

    axes[2].hist(ba, bins=40, color="darkorange", edgecolor="white")
    axes[2].axvline(np.median(ba), color="red", lw=2, ls="--",
                    label=f"median={np.median(ba):.3f}")
    axes[2].set_xlabel("beta"); axes[2].set_title("Beta by product"); axes[2].legend()

    fig.suptitle(f"Single-store model (store {store_id}): "
                 f"{df['product_id'].nunique()} products, no store_id needed",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "single_store.png"), dpi=150)
    fig.savefig(os.path.join(OUTDIR, "single_store.pdf"))
    plt.close(fig)
    print(f"Plots -> output/single_store.pdf")

    pd.DataFrame(dict(product_id=active, alpha=aa, beta=ba)).to_csv(
        os.path.join(OUTDIR, "single_store_alpha_beta.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
