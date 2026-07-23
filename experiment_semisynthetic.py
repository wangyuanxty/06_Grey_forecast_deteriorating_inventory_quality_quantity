#!/usr/bin/env python3
"""
Semi-synthetic experiment: inject known lambda into real data,
test whether paper model and NN model can recover it.

Two datasets: Dingdong (store 18), Liquor (top products).
Three true lambda values: 0.01, 0.05, 0.10.
Two models: Paper (grid search) vs Full NN.

Facts: Standalone. Reads FreshRetailNet-50K/{train,eval}.parquet
  and Retail_Inventory_2024/{begin_inventory,sales,purchases}.csv.
Writes output/semisynthetic_results.csv.
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTDIR, exist_ok=True)
print(f"Device: {DEVICE}")


# ===== 1. Dingdong data =====

def prepare_dingdong():
    DATADIR = os.path.join(os.path.dirname(__file__), "FreshRetailNet-50K")
    train_df = pd.read_parquet(os.path.join(DATADIR, "train.parquet"))
    eval_df  = pd.read_parquet(os.path.join(DATADIR, "eval.parquet"))
    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    df["day_of_week"] = df["dt"].dt.dayofweek.values.astype(np.float32)

    # store 18
    df = df[df["store_id"] == 18].copy()

    # filter products
    ps = df.groupby("product_id").agg(
        n=("dt","nunique"), ms=("sale_amount","mean"),
        ds=("discount","std")).reset_index()
    keep = ps[(ps["n"]>=40)&(ps["ms"]>1)&(ps["ds"]>0.01)]["product_id"]
    df = df[df["product_id"].isin(keep)].sort_values(["product_id","dt"]).reset_index(drop=True)
    df["t_rel"] = df.groupby("product_id").cumcount().astype(np.float32)

    # features
    X = np.column_stack([
        df["t_rel"].values,
        df["discount"].values,
        df["day_of_week"].values / 6.0,
        df["holiday_flag"].values.astype(np.float32),
        df["activity_flag"].values.astype(np.float32),
        df["avg_temperature"].values.astype(np.float32) / 40.0,
        df["avg_humidity"].values.astype(np.float32) / 100.0,
        df["precpt"].values.astype(np.float32) / 50.0,
        df["avg_wind_level"].values.astype(np.float32) / 10.0,
    ])
    y = df["sale_amount"].values.astype(np.float32)
    disc = np.clip(df["discount"].values.astype(np.float32), 0.1, 1.5)
    y_corrected = y / disc
    t_rel = df["t_rel"].values.astype(np.float32)
    pid = df["product_id"].values.astype(np.int64)

    return X, y_corrected, t_rel, disc, pid, df


# ===== 2. Liquor data =====

def prepare_liquor():
    DATADIR = os.path.join(os.path.dirname(__file__), "Retail_Inventory_2024")
    beg = pd.read_csv(os.path.join(DATADIR, "begin_inventory.csv"))

    # pick products with purchases + good sales
    pur = pd.read_csv(os.path.join(DATADIR, "purchases.csv"), nrows=300000)
    pur["ReceivingDate"] = pd.to_datetime(pur["ReceivingDate"])
    pur_cnt = pur.groupby(["InventoryId","Store"]).size().reset_index(name="np")
    good_p = pur_cnt[(pur_cnt["np"]>=6)&(pur_cnt["np"]<=30)]
    good_p = good_p.merge(beg[["InventoryId","Store","onHand","Price"]],
                           on=["InventoryId","Store"], how="inner")
    good_p = good_p[(good_p["onHand"]>30)&(good_p["Price"]>2)]

    all_data = []
    for _, row in good_p.nlargest(6, "np").iterrows():
        inv_id = row["InventoryId"]; sid = int(row["Store"])
        br = beg[(beg["InventoryId"]==inv_id)&(beg["Store"]==sid)]
        if len(br)==0: continue
        I0 = float(br["onHand"].iloc[0])

        pur_sub = pur[(pur["InventoryId"]==inv_id)&(pur["Store"]==sid)].copy()

        sales_chunks = []
        for chunk in pd.read_csv(os.path.join(DATADIR,"sales.csv"), chunksize=200000):
            sub = chunk[(chunk["InventoryId"]==inv_id)&(chunk["Store"]==sid)]
            if len(sub)>0: sales_chunks.append(sub)
        if not sales_chunks: continue
        sales_sub = pd.concat(sales_chunks, ignore_index=True)
        sales_sub["SalesDate"] = pd.to_datetime(sales_sub["SalesDate"])
        sales_sub = sales_sub.sort_values("SalesDate")

        start = sales_sub["SalesDate"].min(); end = sales_sub["SalesDate"].max()
        dr = pd.date_range(start, end, freq="D")
        daily = pd.DataFrame({"date": dr}); daily["d"]=0.0; daily["Q"]=0.0
        s_agg = sales_sub.groupby("SalesDate").agg(
            d=("SalesQuantity","sum"), p=("SalesPrice","mean")).reset_index()
        s_agg["SalesDate"] = pd.to_datetime(s_agg["SalesDate"])
        for _, sr in s_agg.iterrows():
            m = daily["date"]==sr["SalesDate"]
            daily.loc[m,"d"]=sr["d"]; daily.loc[m,"p"]=sr["p"]
        for _, pr in pur_sub.iterrows():
            m = daily["date"]==pr["ReceivingDate"]
            if m.any(): daily.loc[m,"Q"]+=float(pr["Quantity"])

        I = np.zeros(len(daily)); I[0]=I0
        for k in range(1,len(daily)):
            I[k]=max(0.0, I[k-1]+daily.iloc[k]["Q"]-daily.iloc[k]["d"])
        daily["I"]=I
        daily["t_rel"]=np.arange(len(daily),dtype=np.float32)
        daily["pid"]=hash(inv_id)%10000
        all_data.append(daily)

    if not all_data: return None,None,None,None,None,None
    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all[df_all["d"]>0].copy()

    X = np.column_stack([
        df_all["t_rel"].values,
        df_all["p"].values / df_all["p"].mean(),  # normalize price
        np.zeros(len(df_all), dtype=np.float32),
        np.zeros(len(df_all), dtype=np.float32),
        np.zeros(len(df_all), dtype=np.float32),
        np.zeros(len(df_all), dtype=np.float32),
        np.zeros(len(df_all), dtype=np.float32),
        np.zeros(len(df_all), dtype=np.float32),
        np.zeros(len(df_all), dtype=np.float32),
    ])
    y = df_all["d"].values.astype(np.float32)
    t_rel = df_all["t_rel"].values.astype(np.float32)
    p_raw = df_all["p"].values.astype(np.float32)
    pid = df_all["pid"].values.astype(np.int64)

    return X, y, t_rel, p_raw, pid, df_all


# ===== 3. Paper model with grid search =====

def fit_paper_grid(X_tr, y_tr, p_tr, t_tr, pid_tr):
    """Grid search lambda, OLS for alpha,beta per product."""
    lams = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
    products = np.unique(X_tr[:, 0] * 0)  # placeholder — single product case
    best_lam = 0.01; best_rss = np.inf
    best_ab = {}

    # grouped by product (for liquor, product IDs are in last col)
    # For dingdong, pid is separate array
    for lam in lams:
        rss_total = 0.0
        for prod in np.unique(pid_tr):
            mask = pid_tr == prod
            if mask.sum() < 5: continue
            t_p = t_tr[mask]; y_p = y_tr[mask]; p_p = p_tr[mask]
            n = len(t_p)
            # build OLS matrix: d_j = alpha*A_j - beta*(p*A_j)
            A = np.zeros(n)
            for k in range(n):
                tp = max(t_p[k]-0.5, 0.0); tk = t_p[k]+0.5
                A[k] = (np.exp(-lam*tp)-np.exp(-lam*tk))/max(lam,1e-8)
            H = np.column_stack([A, -p_p*A])
            try:
                coef, _, _, _ = np.linalg.lstsq(H, y_p, rcond=None)
                alpha, beta = coef[0], coef[1]
                pred = H @ coef
                rss_total += np.sum((y_p-pred)**2)
            except: rss_total += 1e10
        if rss_total < best_rss:
            best_rss = rss_total; best_lam = lam

    return best_lam


def eval_paper(lam, X_ev, y_ev, p_ev, t_ev, pid_ev):
    obs_all, pred_all = [], []
    for prod in np.unique(pid_ev):
        mask = pid_ev == prod
        if mask.sum() < 2: continue
        t_p = t_ev[mask]; y_p = y_ev[mask]; p_p = p_ev[mask]
        n = len(t_p)
        A = np.zeros(n)
        for k in range(n):
            tp = max(t_p[k]-0.5,0.0); tk = t_p[k]+0.5
            A[k] = (np.exp(-lam*tp)-np.exp(-lam*tk))/max(lam,1e-8)
        H = np.column_stack([A, -p_p*A])
        coef, _, _, _ = np.linalg.lstsq(H, y_p, rcond=None)
        pred = H @ coef
        obs_all.extend(y_p.tolist()); pred_all.extend(pred.tolist())
    obs, pred = np.array(obs_all), np.array(pred_all)
    ssr = np.sum((obs-pred)**2); sst = np.sum((obs-obs.mean())**2)
    return 1.0-ssr/sst if sst>0 else 0.0


# ===== 4. NN model (full NN, no lambda) =====

class FullNN(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Softplus(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn(X_tr, y_tr, X_vl, y_vl, epochs=200):
    scaler = StandardScaler().fit(X_tr)
    Xt = torch.tensor(scaler.transform(X_tr), dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(scaler.transform(X_vl), dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_vl, dtype=torch.float32, device=DEVICE)

    model = FullNN(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    best_v = float("inf"); best_st = None
    for ep in range(epochs):
        model.train()
        loss = nn.MSELoss()(model(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(model(Xv), yv).item()
        if vl < best_v: best_v = vl; best_st = {k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        pred = model(Xv).cpu().numpy()
    obs = y_vl
    ssr = np.sum((obs-pred)**2); sst = np.sum((obs-obs.mean())**2)
    r2 = 1.0-ssr/sst if sst>0 else 0.0
    return r2, model, scaler


# ===== 5. Main experiment =====

def run_experiment(name, X, y_orig, t_rel, p_raw, pid_arr, df):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    true_lams = [0.01, 0.05, 0.10]
    results = []

    # time split
    if "dt" in df.columns:
        dates = sorted(df["dt"].unique())
        sd = dates[int(len(dates)*0.8)]
        train_mask = df["dt"].values <= pd.Timestamp(sd)
        val_mask   = df["dt"].values > pd.Timestamp(sd)
    else:
        n_tr = int(len(y_orig)*0.8)
        train_mask = np.zeros(len(y_orig), dtype=bool)
        train_mask[:n_tr] = True
        val_mask = ~train_mask

    for true_lam in true_lams:
        print(f"\n  --- true lambda = {true_lam} ---")

        # inject decay
        y_syn = y_orig * np.exp(-true_lam * t_rel)

        y_tr = y_syn[train_mask]; y_vl = y_syn[val_mask]
        X_tr = X[train_mask]; X_vl = X[val_mask]
        t_tr = t_rel[train_mask]; t_vl = t_rel[val_mask]
        p_tr = p_raw[train_mask]; p_vl = p_raw[val_mask]
        pid_tr = pid_arr[train_mask]; pid_vl = pid_arr[val_mask]

        # Paper model
        t0 = time.time()
        lam_hat = fit_paper_grid(X_tr, y_tr, p_tr, t_tr, pid_tr)
        r2_p = eval_paper(lam_hat, X_vl, y_vl, p_vl, t_vl, pid_vl)
        t_p = time.time()-t0

        # NN model
        t0 = time.time()
        r2_n, _, _ = train_nn(X_tr, y_tr, X_vl, y_vl)
        t_n = time.time()-t0

        err = abs(lam_hat - true_lam) / true_lam
        print(f"    Paper:  lambda_hat={lam_hat:.4f} (err={err*100:.1f}%)  R2={r2_p:.4f}  ({t_p:.1f}s)")
        print(f"    NN:     R2={r2_n:.4f}  ({t_n:.1f}s)")

        results.append(dict(
            dataset=name, true_lam=true_lam,
            paper_lam=lam_hat, paper_err=err, paper_r2=r2_p,
            nn_r2=r2_n,
        ))

    return results


# ===== 6. Main =====

def main():
    all_results = []

    # Dingdong
    X_dd, y_dd, t_dd, disc_dd, pid_dd, df_dd = prepare_dingdong()
    print(f"Dingdong: {len(y_dd):,} rows, {len(np.unique(pid_dd))} products")
    all_results.extend(run_experiment("Dingdong", X_dd, y_dd, t_dd, disc_dd, pid_dd, df_dd))

    # Liquor
    X_lq, y_lq, t_lq, p_lq, pid_lq, df_lq = prepare_liquor()
    if y_lq is not None:
        print(f"\nLiquor: {len(y_lq):,} rows, {len(np.unique(pid_lq))} products")
        all_results.extend(run_experiment("Liquor", X_lq, y_lq, t_lq, p_lq, pid_lq, df_lq))
    else:
        print("Liquor: no valid products found")

    # Summary
    rdf = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    for ds in rdf["dataset"].unique():
        sub = rdf[rdf["dataset"]==ds]
        print(f"\n  {ds}:")
        for _, row in sub.iterrows():
            print(f"    true_lambda={row['true_lam']:.2f}:  "
                  f"paper_lam_hat={row['paper_lam']:.4f} (err={row['paper_err']*100:.0f}%)  "
                  f"paper_R2={row['paper_r2']:.4f}  nn_R2={row['nn_r2']:.4f}")

    rdf.to_csv(os.path.join(OUTDIR, "semisynthetic_results.csv"), index=False)
    print(f"\nResults -> output/semisynthetic_results.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ds in zip(axes, rdf["dataset"].unique()):
        sub = rdf[rdf["dataset"]==ds]
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x-w/2, sub["paper_r2"], w, label="Paper R2", color="steelblue")
        ax.bar(x+w/2, sub["nn_r2"], w, label="NN R2", color="darkorange")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"lambda={t:.2f}" for t in sub["true_lam"]])
        ax.set_ylabel("R2 val"); ax.set_title(ds); ax.legend()
    fig.suptitle("Semi-synthetic: Paper vs Full NN", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "semisynthetic.png"), dpi=150)
    fig.savefig(os.path.join(OUTDIR, "semisynthetic.pdf"))
    plt.close(fig)
    print(f"Plots -> output/semisynthetic.pdf")


if __name__ == "__main__":
    main()
