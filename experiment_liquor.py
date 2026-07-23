#!/usr/bin/env python3
"""
Full paper-model validation on Retail Inventory 2024 (liquor retail).

Reconstructs daily I(t) from begin_inventory + purchases - sales,
extracts replenishment cycles, and fits the paper's dual-equation model.
"""

import os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DATADIR = os.path.join(os.path.dirname(__file__), "Retail_Inventory_2024")
OUTDIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTDIR, exist_ok=True)


def load_inventory_reconstruction(inv_id, store_id):
    """Reconstruct daily I(t) = begin_inv + cumulative_purchases - cumulative_sales."""
    beg = pd.read_csv(os.path.join(DATADIR, "begin_inventory.csv"))
    beg_row = beg[(beg["InventoryId"] == inv_id) & (beg["Store"] == store_id)]
    if len(beg_row) == 0:
        return None
    I0 = float(beg_row["onHand"].iloc[0])

    pur = pd.read_csv(os.path.join(DATADIR, "purchases.csv"))
    pur_sub = pur[(pur["InventoryId"] == inv_id) & (pur["Store"] == store_id)].copy()
    pur_sub["ReceivingDate"] = pd.to_datetime(pur_sub["ReceivingDate"])
    pur_sub = pur_sub.sort_values("ReceivingDate")

    # Read ALL sales, filter for this product only (memory-efficient with chunks)
    sales_chunks = []
    for chunk in pd.read_csv(os.path.join(DATADIR, "sales.csv"), chunksize=200000):
        sub = chunk[(chunk["InventoryId"] == inv_id) & (chunk["Store"] == store_id)]
        if len(sub) > 0:
            sales_chunks.append(sub)
    if not sales_chunks:
        return None
    sales_sub = pd.concat(sales_chunks, ignore_index=True)
    sales_sub["SalesDate"] = pd.to_datetime(sales_sub["SalesDate"])
    sales_sub = sales_sub.sort_values("SalesDate")

    start_date = sales_sub["SalesDate"].min()
    end_date = sales_sub["SalesDate"].max()
    date_range = pd.date_range(start_date, end_date, freq="D")
    daily = pd.DataFrame({"date": date_range})
    daily["d"] = 0.0; daily["Q_purchase"] = 0.0

    s_agg = sales_sub.groupby("SalesDate").agg(
        d=("SalesQuantity", "sum"), p=("SalesPrice", "mean")).reset_index()
    s_agg["SalesDate"] = pd.to_datetime(s_agg["SalesDate"])
    for _, row in s_agg.iterrows():
        mask = daily["date"] == row["SalesDate"]
        daily.loc[mask, "d"] = row["d"]; daily.loc[mask, "p"] = row["p"]

    for _, row in pur_sub.iterrows():
        mask = daily["date"] == row["ReceivingDate"]
        if mask.any():
            daily.loc[mask, "Q_purchase"] += float(row["Quantity"])

    I = np.zeros(len(daily)); I[0] = I0
    for k in range(1, len(daily)):
        I[k] = max(0.0, I[k - 1] + daily.iloc[k]["Q_purchase"] - daily.iloc[k]["d"])
    daily["I"] = I

    daily["l"] = 0.0
    for k in range(1, len(daily)):
        daily.loc[daily.index[k], "l"] = max(0.0,
            daily.iloc[k - 1]["I"] - daily.iloc[k]["I"] + daily.iloc[k]["Q_purchase"])

    return daily, inv_id, store_id


def extract_replenishment_cycles(daily):
    """Split into cycles, each starting at a purchase event."""
    daily = daily.reset_index(drop=True)
    purchase_days = daily.index[daily["Q_purchase"] > 0].tolist()
    if not purchase_days:
        return []
    cycles = []
    for pi in range(len(purchase_days)):
        start = purchase_days[pi]
        end = purchase_days[pi + 1] if pi + 1 < len(purchase_days) else len(daily)
        block = daily.iloc[start:end]
        if len(block) >= 3:
            cycles.append(block)
    return cycles


def estimate_lambda_from_I(cycles):
    """Eq (13a): l_j + d_j = -lambda * 0.5*(I_j + I_{j-1})*dt"""
    H, Y, W = [], [], []
    for cyc in cycles:
        I = cyc["I"].values.astype(float)
        l = cyc["l"].values.astype(float)
        d = cyc["d"].values.astype(float)
        for k in range(1, len(cyc)):
            X = -0.5 * (I[k] + I[k - 1]) * 1.0
            y = l[k] + d[k]
            w = 1.0 / max(I[k] + I[k - 1], 1.0)
            if abs(X) > 1e-8:
                H.append(X); Y.append(y); W.append(w)
    if len(H) < 3:
        return 0.001
    H, Y, W = np.array(H), np.array(Y), np.array(W)
    return max(float(np.sum(W * H * Y) / np.sum(W * H * H)), 1e-8)


def fit_paper_model(cycles):
    """Joint nonlinear LS for lambda, alpha, beta."""
    lam0 = estimate_lambda_from_I(cycles)

    def residual(theta):
        lam, alpha, beta = theta
        if lam <= 0 or alpha <= 0:
            return np.full(1000, 1e6)
        res = []
        for cyc in cycles:
            p = cyc["p"].mean()
            Dp = max(alpha - beta * p, 0.0)
            d = cyc["d"].values.astype(float)
            for k in range(len(cyc)):
                tp = max(k - 0.5, 0.0); tk = k + 0.5
                dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * tk))
                res.append(float(d[k]) - dp)
        res.append(0.0)
        return np.array(res)

    total_d = sum(cyc["d"].sum() for cyc in cycles)
    n_obs = sum(len(cyc) for cyc in cycles)
    x0 = [lam0, total_d / n_obs * 2, total_d / n_obs * 0.1]
    try:
        sol = least_squares(residual, x0,
                            bounds=([1e-8, 1e-6, 0], [1.0, 1e6, 1e6]),
                            method="trf", max_nfev=500, ftol=1e-8)
        lam, alpha, beta = sol.x
    except Exception:
        lam, alpha, beta = lam0, total_d / n_obs * 2, 0.0
    return lam, alpha, beta


def compute_r2(cycles, lam, alpha, beta):
    obs_all, pred_all = [], []
    for cyc in cycles:
        p = cyc["p"].mean()
        Dp = max(alpha - beta * p, 0.0)
        d = cyc["d"].values.astype(float)
        for k in range(len(cyc)):
            tp = max(k - 0.5, 0.0); tk = k + 0.5
            dp = (Dp / max(lam, 1e-8)) * (np.exp(-lam * tp) - np.exp(-lam * tk))
            obs_all.append(float(d[k])); pred_all.append(float(dp))
    obs, pred = np.array(obs_all), np.array(pred_all)
    ssr = np.sum((obs - pred) ** 2); sst = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - ssr / sst if sst > 0 else 0.0


def main():
    print("Loading data ...")
    beg = pd.read_csv(os.path.join(DATADIR, "begin_inventory.csv"))
    pur = pd.read_csv(os.path.join(DATADIR, "purchases.csv"), nrows=300000)
    pur_cnt = pur.groupby(["InventoryId", "Store"]).size().reset_index(name="n_pur")
    good_pur = pur_cnt[(pur_cnt["n_pur"] >= 5) & (pur_cnt["n_pur"] <= 50)]
    good = good_pur.merge(beg[["InventoryId", "Store", "onHand", "Price"]],
                          on=["InventoryId", "Store"], how="inner")
    good = good[(good["onHand"] > 20) & (good["Price"] > 1)]
    good["score"] = good["n_pur"] + good["onHand"] / 50
    chosen = good.nlargest(5, "score")
    print(f"Selected {len(chosen)} products\n")

    results = []
    for idx, (_, row) in enumerate(chosen.iterrows()):
        inv_id = row["InventoryId"]
        store_id = int(row["Store"])
        print(f"--- {idx+1}: {inv_id}, Store {store_id} ---")
        result = load_inventory_reconstruction(inv_id, store_id)
        if result is None:
            print("  SKIP\n"); continue
        daily, _, _ = result
        cycles = extract_replenishment_cycles(daily)
        if len(cycles) < 4:
            print(f"  SKIP: {len(cycles)} cycles\n"); continue
        n_train = max(int(len(cycles) * 0.8), 3)
        train_c, val_c = cycles[:n_train], cycles[n_train:]
        t0 = time.time()
        lam, alpha, beta = fit_paper_model(train_c)
        r2t = compute_r2(train_c, lam, alpha, beta)
        r2v = compute_r2(val_c, lam, alpha, beta)
        print(f"  lambda={lam:.6f}  alpha={alpha:.2f}  beta={beta:.2f}")
        print(f"  R^2 train={r2t:.4f}  val={r2v:.4f}  "
              f"({time.time()-t0:.1f}s)\n")
        results.append(dict(inv_id=inv_id, store=store_id,
                            n_days=len(daily), n_cycles=len(cycles),
                            lam=lam, alpha=alpha, beta=beta,
                            r2_train=r2t, r2_val=r2v))

    # Summary
    rdf = pd.DataFrame(results)
    print(f"{'='*60}\nSUMMARY ({len(rdf)} products)\n{'='*60}")
    for k in ["lam", "r2_train", "r2_val"]:
        v = rdf[k].values
        print(f"  {k:12s}: mean={np.mean(v):.4f}  median={np.median(v):.4f}  "
              f"min={np.min(v):.4f}  max={np.max(v):.4f}")
    rdf.to_csv(os.path.join(OUTDIR, "liquor_results.csv"), index=False)
    print(f"\nResults -> output/liquor_results.csv")


if __name__ == "__main__":
    main()
