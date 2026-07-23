#!/usr/bin/env python3
"""
Three NN-enhanced grey forecasting models for FreshRetailNet-50K.

  A: Residual correction — keep paper's (alpha-beta*p)*e^{-lambda*t}, NN learns the rest.
  B: NN demand intensity — d(t) = NN(features) * e^{-lambda*t}.
  C: PINN — d(t) from NN, ODE constraint in the loss, lambda & Q jointly learned.
"""

import os, time, warnings, json
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
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Shared data pipeline
# ═══════════════════════════════════════════════════════════════════════════════

DATADIR = os.path.join(os.path.dirname(__file__), "FreshRetailNet-50K")
OUTDIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTDIR, exist_ok=True)


def load_data():
    train_df = pd.read_parquet(os.path.join(DATADIR, "train.parquet"))
    eval_df  = pd.read_parquet(os.path.join(DATADIR, "eval.parquet"))
    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    df["day_of_week"] = df["dt"].dt.dayofweek.values.astype(np.float32)
    return df


def stockout_rate(arr):
    if arr is None or len(arr) == 0:
        return 1.0
    return float(np.mean(arr))


def extract_cycles(product_df):
    """Split a product-store time series into replenishment cycles."""
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


def build_features(block):
    """Build feature matrix for one cycle."""
    feats = np.column_stack([
        block["discount"].values,
        block["day_of_week"].values / 6.0,
        block["holiday_flag"].values.astype(np.float32),
        block["activity_flag"].values.astype(np.float32),
        block["avg_temperature"].values / 40.0,
        block["avg_humidity"].values / 100.0,
        block["precpt"].values / 50.0,
    ])
    # lag features: pad with first value for day 0
    sales = block["sale_amount"].values.astype(np.float32)
    lag1 = np.concatenate([[sales[0]], sales[:-1]])
    lag2 = np.concatenate([[sales[0]], [sales[0]], sales[:-2]])
    feats = np.column_stack([feats, lag1 / 10.0, lag2 / 10.0])
    return feats


class CycleDataset(Dataset):
    """Turns a list of cycle DataFrames into (t, features, sales, stock_status) tensors."""

    def __init__(self, cycles, scaler=None):
        all_t, all_f, all_s, all_status = [], [], [], []
        for cyc in cycles:
            t = np.arange(len(cyc), dtype=np.float32)[:, None]
            f = build_features(cyc)
            s = cyc["sale_amount"].values.astype(np.float32)
            st = np.array([stockout_rate(x) for x in cyc["hours_stock_status"]], dtype=np.float32)
            all_t.append(t); all_f.append(f); all_s.append(s); all_status.append(st)

        self.t_list = all_t
        self.f_list = all_f
        self.s_list = all_s
        self.status_list = all_status
        self.n_feat = all_f[0].shape[1]

        if scaler is None:
            all_f_flat = np.concatenate(all_f, axis=0)
            self.scaler = StandardScaler().fit(all_f_flat)
        else:
            self.scaler = scaler
        for i in range(len(all_f)):
            self.f_list[i] = self.scaler.transform(self.f_list[i])

    def __len__(self):
        return len(self.t_list)

    def __getitem__(self, idx):
        return (torch.tensor(self.t_list[idx], dtype=torch.float32),
                torch.tensor(self.f_list[idx], dtype=torch.float32),
                torch.tensor(self.s_list[idx], dtype=torch.float32),
                torch.tensor(self.status_list[idx], dtype=torch.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Model A: Residual Correction
# ═══════════════════════════════════════════════════════════════════════════════

class ModelA_Residual(nn.Module):
    """d_pred = (alpha - beta*p) * exp(-lambda*t) + NN_residual(features)"""

    def __init__(self, n_feat, alpha_init=10.0, beta_init=5.0, lam_init=0.01):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha_init)))
        self.log_beta  = nn.Parameter(torch.tensor(np.log(beta_init)))
        self.log_lam   = nn.Parameter(torch.tensor(np.log(lam_init)))
        self.net = nn.Sequential(
            nn.Linear(n_feat, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def beta(self):  return torch.exp(self.log_beta)
    @property
    def lam(self):   return torch.exp(self.log_lam)

    def forward(self, t, feat, price_col=0):
        Dp = self.alpha - self.beta * feat[:, price_col]    # (batch,) scalar per row
        d_grey = Dp * torch.exp(-self.lam * t.squeeze(-1))  # (batch,)
        residual = self.net(feat).squeeze(-1)               # (batch,)
        return d_grey + residual

    def grey_part(self, t, price):
        """Return just the grey-model component for diagnostic."""
        return (self.alpha - self.beta * price) * torch.exp(-self.lam * t)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Model B: NN Demand Intensity (hybrid)
# ═══════════════════════════════════════════════════════════════════════════════

class ModelB_NNDemand(nn.Module):
    """d_pred(t) = NN_intensity(features) * exp(-lambda*t)"""

    def __init__(self, n_feat, lam_init=0.01):
        super().__init__()
        self.log_lam = nn.Parameter(torch.tensor(np.log(lam_init)))
        self.intensity_net = nn.Sequential(
            nn.Linear(n_feat, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),                    # enforce positivity
        )

    @property
    def lam(self): return torch.exp(self.log_lam)

    def forward(self, t, feat, price_col=0):
        intensity = self.intensity_net(feat).squeeze(-1)    # (batch,)
        return intensity * torch.exp(-self.lam * t.squeeze(-1))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Model C: PINN
# ═══════════════════════════════════════════════════════════════════════════════

class ModelC_PINN(nn.Module):
    """ODE: dI/dt = -lambda*I - d_NN(t).  Jointly learn lambda, Q, and NN weights."""

    def __init__(self, n_feat, n_cycles, lam_init=0.01):
        super().__init__()
        self.log_lam = nn.Parameter(torch.tensor(np.log(lam_init)))
        self.log_Q = nn.Parameter(torch.ones(n_cycles) * np.log(20.0))  # one Q per cycle
        self.demand_net = nn.Sequential(
            nn.Linear(n_feat + 1, 32), nn.ReLU(),   # +1 for t
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    @property
    def lam(self):  return torch.exp(self.log_lam)
    @property
    def Q(self):    return torch.exp(self.log_Q)

    def demand(self, t, feat):
        """d_pred = NN(t, features). Softplus output ensures positivity."""
        inp = torch.cat([t, feat], dim=-1)
        return self.demand_net(inp).squeeze(-1)

    def integrate_euler(self, t, feat, cycle_idx):
        """Euler-integrate the ODE to get inventory trajectory I(t)."""
        Q = self.Q[cycle_idx]                                       # scalar
        lam = self.lam
        n_steps = len(t)
        dt_val = float(t[1, 0] - t[0, 0]) if n_steps > 1 else 1.0

        d_pred_list, I_traj_list = [], []
        Ik = Q                                                      # scalar at t=0

        for k in range(n_steps):
            dk = self.demand(t[k:k + 1], feat[k:k + 1]).squeeze()  # scalar
            d_pred_list.append(dk)
            I_traj_list.append(Ik)

            # Euler step: I_{k+1} = I_k + dt * (-lam * I_k - d_k)
            I_next = Ik + dt_val * (-lam * Ik - dk)
            Ik = torch.clamp(I_next, min=0.0)

        I_traj = torch.stack(I_traj_list)   # (n_steps,)
        d_pred = torch.stack(d_pred_list)   # (n_steps,)
        return I_traj, d_pred


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Training utilities
# ═══════════════════════════════════════════════════════════════════════════════

def compute_r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def train_model_a(model, dataset, epochs=200, lr=0.01, lam_reg=0.1):
    """Train residual-correction model."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.5)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for t, f, s, _ in dataset:
            t, f, s = t.to(DEVICE), f.to(DEVICE), s.to(DEVICE)
            pred = model(t, f)
            loss = nn.MSELoss()(pred, s)
            # regularize: don't let lambda stray too far from 0.01
            loss = loss + lam_reg * (model.lam - 0.01) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
    return model


def train_model_b(model, dataset, epochs=200, lr=0.01, lam_reg=0.1):
    """Train NN-demand hybrid model."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.5)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for t, f, s, _ in dataset:
            t, f, s = t.to(DEVICE), f.to(DEVICE), s.to(DEVICE)
            pred = model(t, f)
            loss = nn.MSELoss()(pred, s) + lam_reg * (model.lam - 0.01) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
    return model


def train_model_c(model, dataset, epochs=300, lr=0.005, w_phys=0.5, w_bnd=1.0, w_nonneg=0.3):
    """Train PINN model with physics + boundary + non-negativity losses."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50, factor=0.5)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for cycle_idx, (t, f, s, status) in enumerate(dataset):
            t, f, s, status = t.to(DEVICE), f.to(DEVICE), s.to(DEVICE), status.to(DEVICE)

            I_traj, d_pred = model.integrate_euler(t, f, cycle_idx)

            # Data loss: demand prediction
            L_data = nn.MSELoss()(d_pred, s)

            # Physics loss: ODE residual  dI/dt + lambda*I + d = 0
            if len(t) > 1:
                dI_dt = (I_traj[1:] - I_traj[:-1]) / (t[1:] - t[:-1]).squeeze()
                ode_res = dI_dt + model.lam * I_traj[:-1].detach() + d_pred[:-1]
                L_phys = (ode_res ** 2).mean()
            else:
                L_phys = torch.tensor(0.0, device=DEVICE)

            # Boundary loss: I(T) ≈ 0 at cycle end
            L_bnd = (I_traj[-1] ** 2).squeeze()

            # Non-negativity: I(t) >= 0 everywhere
            L_nonneg = (torch.clamp(-I_traj, min=0) ** 2).mean()

            # Stockout-hour boundary: where status > 0.8, I should be near 0
            stockout_mask = (status > 0.8)
            if stockout_mask.any():
                L_stockout = (I_traj[stockout_mask] ** 2).mean()
                L_bnd = L_bnd + L_stockout

            loss = L_data + w_phys * L_phys + w_bnd * L_bnd + w_nonneg * L_nonneg
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
    return model


def evaluate(model, dataset, model_type, price_col=0):
    """Compute R^2 and collect predictions for a model over all cycles."""
    model.eval()
    all_s, all_p = [], []
    with torch.no_grad():
        for idx, (t, f, s, status) in enumerate(dataset):
            t, f, s = t.to(DEVICE), f.to(DEVICE), s.to(DEVICE)
            if model_type == "C":
                _, pred = model.integrate_euler(t, f, idx)
            else:
                pred = model(t, f)
            # flatten to 1-D list (cycles have different lengths)
            all_s.extend(s.cpu().tolist())
            all_p.extend(pred.cpu().tolist())
    ys = torch.tensor(all_s)
    yp = torch.tensor(all_p)
    return float(compute_r2(ys, yp)), ys.numpy(), yp.numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data ...")
    df = load_data()

    # Select combos with strong signal
    combo_stats = df.groupby(["product_id", "store_id"]).agg(
        n_days=("dt", "nunique"),
        mean_sales=("sale_amount", "mean"),
        discount_std=("discount", "std"),
        discount_nunique=("discount", "nunique"),
        stockout_rate=("hours_stock_status", lambda x: np.mean([stockout_rate(s) for s in x])),
    ).reset_index()

    good = combo_stats[
        (combo_stats["n_days"] >= 50)
        & (combo_stats["mean_sales"] > 3)
        & (combo_stats["discount_std"] > 0.03)
        & (combo_stats["discount_nunique"] >= 2)
        & (combo_stats["stockout_rate"] > 0.1)
        & (combo_stats["stockout_rate"] < 0.6)
    ]
    print(f"Candidates: {len(good)} combos")

    # Pick top 4 by number of cycles (more cycles = richer signal)
    # First compute actual cycle counts to rank
    cycle_counts = []
    for _, row in good.iterrows():
        pid, sid = int(row["product_id"]), int(row["store_id"])
        sub = df[(df["product_id"] == pid) & (df["store_id"] == sid)]
        cycles = extract_cycles(sub)
        prices = [float(c["discount"].mean()) for c in cycles]
        p_std = np.std(prices) if len(prices) > 1 else 0
        if len(cycles) >= 3 and p_std > 0.01:
            cycle_counts.append((pid, sid, len(cycles), row["mean_sales"], row["discount_std"], p_std))

    cycle_df = pd.DataFrame(cycle_counts, columns=["pid", "sid", "n_cycles", "mean_sales", "discount_std", "p_std"])
    chosen = cycle_df.nlargest(4, "n_cycles")
    results = []

    for row_idx, (_, crow) in enumerate(chosen.iterrows()):
        pid, sid = int(crow["pid"]), int(crow["sid"])
        sub = df[(df["product_id"] == pid) & (df["store_id"] == sid)]
        cycles = extract_cycles(sub)

        if len(cycles) < 3:
            continue

        # Train/val split: first 80% cycles for training
        n_train = max(int(len(cycles) * 0.8), 4)
        train_cycles = cycles[:n_train]
        val_cycles = cycles[n_train:]

        train_ds = CycleDataset(train_cycles)
        val_ds = CycleDataset(val_cycles, scaler=train_ds.scaler)

        n_feat = train_ds.n_feat
        print(f"\n{'='*60}")
        print(f"Combo {row_idx+1}: product={pid}, store={sid}")
        print(f"  cycles: {len(cycles)} ({n_train} train / {len(val_cycles)} val)")
        print(f"  mean sales: {crow['mean_sales']:.1f}, discount std: {crow['discount_std']:.3f}")

        # --- Model A ---
        t0 = time.time()
        model_a = ModelA_Residual(n_feat).to(DEVICE)
        model_a = train_model_a(model_a, train_ds)
        r2_a, ya, ypa = evaluate(model_a, val_ds, "A")
        lam_a = float(model_a.lam.detach().cpu())
        t_a = time.time() - t0
        print(f"  [A] Residual:   val R^2={r2_a:.4f},  lambda={lam_a:.4f}  ({t_a:.1f}s)")

        # --- Model B ---
        t0 = time.time()
        model_b = ModelB_NNDemand(n_feat).to(DEVICE)
        model_b = train_model_b(model_b, train_ds)
        r2_b, yb, ypb = evaluate(model_b, val_ds, "B")
        lam_b = float(model_b.lam.detach().cpu())
        t_b = time.time() - t0
        print(f"  [B] NN-demand:  val R^2={r2_b:.4f},  lambda={lam_b:.4f}  ({t_b:.1f}s)")

        # --- Model C ---
        t0 = time.time()
        model_c = ModelC_PINN(n_feat, n_train).to(DEVICE)
        model_c = train_model_c(model_c, train_ds)
        r2_c, yc, ypc = evaluate(model_c, val_ds, "C")
        lam_c = float(model_c.lam.detach().cpu())
        t_c = time.time() - t0
        print(f"  [C] PINN:       val R^2={r2_c:.4f},  lambda={lam_c:.4f}  ({t_c:.1f}s)")

        results.append(dict(
            product_id=pid, store_id=sid, n_cycles=len(cycles),
            r2_a=r2_a, lam_a=lam_a,
            r2_b=r2_b, lam_b=lam_b,
            r2_c=r2_c, lam_c=lam_c,
        ))

        # --- Plot comparison for first combo ---
        if row_idx == 0 and len(val_cycles) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            ts_val, ss_val = [], []
            for cyc_idx, (t, f, s, _) in enumerate(val_ds):
                ts_val.extend(t[:, 0].tolist())
                ss_val.extend(s.tolist())
            ss_val = np.array(ss_val)

            for ax, yp, title in [
                (axes[0], ypa, f"Model A (Residual)\nR^2={r2_a:.3f}  lambda={lam_a:.4f}"),
                (axes[1], ypb, f"Model B (NN-demand)\nR^2={r2_b:.3f}  lambda={lam_b:.4f}"),
                (axes[2], ypc, f"Model C (PINN)\nR^2={r2_c:.3f}  lambda={lam_c:.4f}"),
            ]:
                ax.scatter(ss_val, yp, alpha=0.5, s=12)
                ax.plot([0, max(ss_val)], [0, max(ss_val)], "r--", lw=1)
                ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
                ax.set_title(title, fontsize=11)
            fig.suptitle(f"product={pid} store={sid} (validation cycles)", fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, "nn_comparison_scatter.png"), dpi=150)
            fig.savefig(os.path.join(OUTDIR, "nn_comparison_scatter.pdf"))
            plt.close(fig)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for m, key_r2, key_lam in [("A-Residual", "r2_a", "lam_a"),
                                 ("B-NN-demand", "r2_b", "lam_b"),
                                 ("C-PINN", "r2_c", "lam_c")]:
        r2s = [r[key_r2] for r in results]
        lams = [r[key_lam] for r in results]
        print(f"  {m:14s}:  R^2 mean={np.mean(r2s):.4f}  median={np.median(r2s):.4f}  |  "
              f"lambda mean={np.mean(lams):.4f}  median={np.median(lams):.4f}")

    # Save
    pd.DataFrame(results).to_csv(os.path.join(OUTDIR, "nn_results.csv"), index=False)
    print(f"\nResults -> output/nn_results.csv")


if __name__ == "__main__":
    main()
