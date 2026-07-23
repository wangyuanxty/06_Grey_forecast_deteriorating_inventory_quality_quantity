#!/usr/bin/env python3
"""Inventory-level injection: test equation (13a) on Liquor + Dingdong.

Inject lambda via I_{k+1}=I_k - lambda*I_k*dt - d_k.
Recover via OLS: l + d = -lambda * 0.5*(I_j+I_{j-1})*dt.

Facts: Standalone. Reads Retail_Inventory_2024/{begin_inventory,sales,purchases}.csv
  and FreshRetailNet-50K/{train,eval}.parquet.
Writes output/inventory_injection_results.csv.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
LIQ = os.path.join(os.path.dirname(__file__), "Retail_Inventory_2024")
DD  = os.path.join(os.path.dirname(__file__), "FreshRetailNet-50K")
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


# ---- Liquor ----

def liq_I(inv_id, store_id):
    beg = pd.read_csv(os.path.join(LIQ, "begin_inventory.csv"))
    br = beg[(beg["InventoryId"]==inv_id)&(beg["Store"]==store_id)]
    if len(br)==0: return None
    I0 = float(br["onHand"].iloc[0])
    pur = pd.read_csv(os.path.join(LIQ, "purchases.csv"))
    pur["ReceivingDate"] = pd.to_datetime(pur["ReceivingDate"])
    ps = pur[(pur["InventoryId"]==inv_id)&(pur["Store"]==store_id)]
    chunks=[]
    for c in pd.read_csv(os.path.join(LIQ,"sales.csv"),chunksize=200000):
        s=c[(c["InventoryId"]==inv_id)&(c["Store"]==store_id)]
        if len(s)>0: chunks.append(s)
    if not chunks: return None
    ss=pd.concat(chunks); ss["SalesDate"]=pd.to_datetime(ss["SalesDate"])
    dr=pd.date_range(ss["SalesDate"].min(),ss["SalesDate"].max(),freq="D")
    daily=pd.DataFrame({"date":dr}); daily["d"]=0.0; daily["Q"]=0.0
    for dt,v in ss.groupby("SalesDate")["SalesQuantity"].sum().items():
        daily.loc[daily["date"]==pd.Timestamp(dt),"d"]=v
    for _,r in ps.iterrows():
        m=daily["date"]==r["ReceivingDate"]
        if m.any(): daily.loc[m,"Q"]+=float(r["Quantity"])
    I=np.zeros(len(daily)); I[0]=I0
    for k in range(1,len(daily)):
        I[k]=max(0.0,I[k-1]+daily.iloc[k]["Q"]-daily.iloc[k]["d"])
    daily["I"]=I
    return daily

def liq_cycles(daily):
    pd_=daily.index[daily["Q"]>0].tolist()
    if not pd_: return []
    out=[]
    for pi in range(len(pd_)):
        s=pd_[pi]; e=pd_[pi+1] if pi+1<len(pd_) else len(daily)
        b=daily.iloc[s:e]
        if len(b)>=3: out.append(b)
    return out


# ---- Dingdong ----

def dd_I(product_df):
    df=product_df.sort_values("dt").reset_index(drop=True); n=len(df)
    d_daily=df["sale_amount"].values.astype(float)
    soh=np.full(n,24,dtype=int); sbs=np.zeros(n)
    for k in range(n):
        st=np.array(df.iloc[k]["hours_stock_status"],dtype=float)
        hs=np.array(df.iloc[k]["hours_sale"],dtype=float)
        idx=np.where(st>0.5)[0]
        if len(idx)>0: soh[k]=idx[0]
        sbs[k]=float(np.sum(hs[:soh[k]]))
    Io=np.zeros(n); Io[-1]=max(sbs[-1],0.01)
    for k in range(n-2,-1,-1):
        if soh[k]<24: Io[k]=max(sbs[k],Io[k+1]+d_daily[k])
        else: Io[k]=Io[k+1]+d_daily[k]
    return pd.DataFrame({"d":d_daily,"I":Io,"Q":np.zeros(n)})

def dd_cycles(daily):
    instock=daily["I"]>0.01; cycles=[]; i=0; n=len(daily)
    while i<n:
        if not instock.iloc[i]: i+=1; continue
        j=i
        while j<n and instock.iloc[j]: j+=1
        b=daily.iloc[i:j]
        if len(b)>=3: cycles.append(b)
        i=j
    return cycles


# ---- Inject & test ----

def inject_and_test(cycles, lam_true, dt=1.0):
    H,Y=[],[]; npt=0
    for cyc in cycles:
        n=len(cyc); Q=float(cyc.iloc[0]["I"]); d=cyc["d"].values.astype(float)
        Is=np.zeros(n); Is[0]=Q
        for k in range(n-1):
            Is[k+1]=max(0.0,Is[k]-lam_true*Is[k]*dt-d[k])
        for k in range(1,n):
            lk=Is[k]-Is[k-1]
            X=-0.5*(Is[k]+Is[k-1])*dt; y=lk+d[k]
            if abs(X)>1e-6: H.append(X); Y.append(y); npt+=1
    if npt<3: return np.nan,np.nan,np.nan,0
    H=np.array(H); Y=np.array(Y)
    lh=float(np.dot(H,Y)/np.dot(H,H))
    err=abs(lh-lam_true)/lam_true if lam_true>0 else 1.0
    pred=H*lh; ssr=np.sum((Y-pred)**2); sst=np.sum((Y-Y.mean())**2)
    r2=1.0-ssr/sst if sst>0 else 0.0
    return lh,err,r2,npt


# ---- Main ----

def main():
    true_lams=[0.01,0.05,0.10,0.20]
    results=[]

    # Liquor
    print("="*60+"\nLIQUOR\n"+"="*60)
    beg=pd.read_csv(os.path.join(LIQ,"begin_inventory.csv"))
    pur=pd.read_csv(os.path.join(LIQ,"purchases.csv"),nrows=300000)
    pur["ReceivingDate"]=pd.to_datetime(pur["ReceivingDate"])
    pc=pur.groupby(["InventoryId","Store"]).size().reset_index(name="np")
    g=pc[(pc["np"]>=6)&(pc["np"]<=30)]
    g=g.merge(beg[["InventoryId","Store","onHand","Price"]],on=["InventoryId","Store"],how="inner")
    g=g[(g["onHand"]>30)&(g["Price"]>2)]
    ch=g.nlargest(6,"np")
    for idx,(_,r) in enumerate(ch.iterrows()):
        daily=liq_I(r["InventoryId"],int(r["Store"]))
        if daily is None: continue
        cycles=liq_cycles(daily)
        if len(cycles)<3: continue
        print(f"  {idx+1}: {r['InventoryId'][:20]}, {len(daily)}d, {len(cycles)}c, d={daily['d'].mean():.1f}")
        for lam in true_lams:
            lh,err,r2,npt=inject_and_test(cycles,lam)
            s=f"lh={lh:.4f} err={err*100:.0f}% R2={r2:.3f}" if not np.isnan(lh) else "FAIL"
            print(f"    lam={lam:.2f}  {s}")
            results.append(dict(ds="Liquor",prod=r["InventoryId"],tl=lam,lh=lh,err=err,r2=r2,npt=npt))

    # Dingdong
    print("\n"+"="*60+"\nDINGDONG\n"+"="*60)
    tr=pd.read_parquet(os.path.join(DD,"train.parquet"))
    ev=pd.read_parquet(os.path.join(DD,"eval.parquet"))
    dd=pd.concat([tr,ev]); dd["dt"]=pd.to_datetime(dd["dt"])
    dd=dd[dd["store_id"]==18]
    top=dd.groupby("product_id").size().sort_values(ascending=False).head(6).index
    for idx,pid in enumerate(top):
        sub=dd[dd["product_id"]==pid]
        daily=dd_I(sub)
        if daily is None: continue
        cycles=dd_cycles(daily)
        if len(cycles)<1: continue
        print(f"  {idx+1}: pid={pid}, {len(daily)}d, {len(cycles)}c, d={daily['d'].mean():.2f}")
        for lam in true_lams:
            lh,err,r2,npt=inject_and_test(cycles,lam)
            s=f"lh={lh:.4f} err={err*100:.0f}% R2={r2:.3f}" if not np.isnan(lh) else "FAIL"
            print(f"    lam={lam:.2f}  {s}")
            results.append(dict(ds="Dingdong",prod=str(pid),tl=lam,lh=lh,err=err,r2=r2,npt=npt))

    # Summary
    rdf=pd.DataFrame(results); rdf=rdf[~rdf["lh"].isna()]
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for ds in ["Liquor","Dingdong"]:
        s=rdf[rdf["ds"]==ds]
        print(f"\n  {ds} ({s['prod'].nunique()} products):")
        for lam in true_lams:
            s2=s[s["tl"]==lam]
            if len(s2)==0: continue
            print(f"    lam={lam:.2f}: lh_mean={s2['lh'].mean():.4f} med={s2['lh'].median():.4f}  "
                  f"err={s2['err'].mean()*100:.0f}%  R2={s2['r2'].mean():.3f}")

    # Plot
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,ds in zip(axes,["Liquor","Dingdong"]):
        s=rdf[rdf["ds"]==ds]
        for lam in true_lams:
            s2=s[s["tl"]==lam]
            if len(s2)>0:
                ax.boxplot(s2["lh"].dropna().values,positions=[lam],widths=0.015,showfliers=False)
        ax.plot([0,0.22],[0,0.22],"r--",lw=1,label="perfect")
        ax.set_xlabel("true lambda"); ax.set_ylabel("estimated lambda")
        ax.set_title(ds); ax.legend()
    fig.suptitle("Inventory-Level Injection: I_{k+1}=I_k-lambda*I_k-d_k\nOLS recovery via l+d=-lambda*mean(I)",
                 fontweight="bold",fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"inventory_injection.png"),dpi=150)
    fig.savefig(os.path.join(OUT,"inventory_injection.pdf"))
    plt.close(fig)
    rdf.to_csv(os.path.join(OUT,"inventory_injection_results.csv"),index=False)
    print(f"\nPlots -> output/inventory_injection.pdf")


if __name__=="__main__":
    main()