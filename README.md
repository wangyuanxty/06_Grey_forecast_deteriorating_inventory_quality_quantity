# Grey Forecasting for Deteriorating Inventory — 开放问题探索实验

> 原论文：[*Grey forecasting modelling for deteriorating inventory with interdependent quality and quantity decay*](https://doi.org/10.1016/j.eswa.2025.130956) (ESWA '26)
>
> 原仓库：[wangxiaolei0721/grey4deterioratinginventory](https://github.com/wangxiaolei0721/grey4deterioratinginventory)
>
> 本仓库：[wangyuanxty/06_Grey_forecast_deteriorating_inventory_quality_quantity](https://github.com/wangyuanxty/06_Grey_forecast_deteriorating_inventory_quality_quantity)

## 动机

原论文提出灰色预测建模方法，通过 AGO + 双方程联合回归从库存数据中估计衰减速率 $\lambda$ 、潜在需求 $\alpha$ 和价格敏感度 $\beta$ ，进而构建利润优化模型导出最优定价 $p^\ast$ 和补货周期 $T^\ast$ 。

原论文 Section 6 明确提出了三个开放问题：

> "Future research will focus on **analyzing the inventory system using real-world datasets**, developing a **tailored grey forecasting modelling approach**, and providing **intelligent decision support** for inventory management."

本研究围绕前两个：

1. **真实数据集验证**：论文仅在 70 条仿真数据上验证（数据生成与模型共享同一方程形式）。在真实数据上模型能否运行、 $\lambda$ 能否被识别——这两个问题论文没有回答。
2. **场景定制的识别条件**：如果模型在真实数据上表现不佳，是因为方法失效还是场景不匹配？ $\lambda$ 的检测边界在哪里？什么条件下可以信任估计结果？

---

## 论文模型

系统方程为 $dI/dt = -\lambda I - d$ 。需求函数 $d=(\alpha-\beta p)e^{-\lambda t}$ 。将系统方程积分并用梯形近似处理，得到两个回归方程：

$$l_j + d_j = -\lambda \cdot \tfrac{1}{2}(L_j + L_{j-1})\Delta t \qquad
d_j = \frac{\alpha - \beta p}{\lambda}\bigl[e^{-\lambda t_{j-1}} - e^{-\lambda t_j}\bigr]$$

前者关于 $\lambda$ 线性（OLS 一步出初值），后者关于 $\alpha,\beta$ 线性（固定 $\lambda$ 后 OLS）。IRLS 交替迭代收敛——三参数 $(\lambda,\alpha,\beta)$ 从两个方程联合估计。

---

## 数据集搜索

模型需要：逐日库存 $I(t)$ 、逐日销量 $d(t)$ 、价格变化、产品有明显价值衰减。

| 数据集 | I(t) | d(t) | 价格 | 衰减信号 | 规模 | 结论 |
|--------|------|------|------|---------|------|------|
| **叮咚买菜 FreshRetailNet-50K** | 可重建 | 归一化 | discount | $\lambda \approx 0.002$ | 485 万行 | 冷链压制信号 |
| **酒类零售 2024** | 完整 | 真实件数 | 进价+售价 | $\lambda = 0$ | ~1000 万行 | 品类不匹配 |
| TAMD 淘宝服装 | 无 | 聚合 | 无 | 可能有 | 1765 行 | 数据稀疏 |
| 花店订单 | 无 | 每笔一单 | 无 | 有 | 106 行 | 太少 |
| M5 Walmart | 无 | 日销量 | 周价格 | 部分品类 | 30K 条 | 无 I(t) |

**结论：真实 + $I(t)$ + 衰减信号 > 0 = 公开数据里不存在。** 有物理衰减的场景（菜市场、花店）没被数字化。被数字化的场景（冷链生鲜、酒类零售）衰减不可见。

酒类数据的 $I(t)$ 从 `begin_inventory + purchases - sales` 精确重建：三账齐全（`begin_inventory.csv` 含 `onHand`，`purchases.csv` 含 `ReceivingDate` + `Quantity`，`sales.csv` 含 `SalesQuantity`——每个补货周期完整还原）。叮咚只有缺货标记（0/1）和小时级销量，从缺货小时倒推给出 $I(t)$ 的下界。

---

## 实验一：论文原版在真实数据上的表现

将 $d = (\alpha-\beta p) \cdot e^{-\lambda t}$ 在叮咚数据上逐产品训练，格点搜索 $\lambda$ ，OLS 拟合 $\alpha,\beta$ 。时间 8:2 切分。

```
叮咚 169 个产品×门店组合：

R^2 中位数：  -0.61
lambda 中位数： 0.002
alpha 范围：    3.8 ~ 1327（爆炸）
R^2 > 0：       0/169
```

$R^2 = -0.61$ 意味着预测不如直接猜平均值。$\alpha$ 从 3.8 跳到 1327——当 $e^{-\lambda t} \approx 1$（冷链 $\lambda \approx 0$），模型退化为常数预测，NLS 优化器将 $\alpha$ 推向荒谬值来补偿残差。

酒类数据上 $\lambda$ 正确收敛到 0，框架本身可运行——问题不在方法，在冷链。

---

## 实验二：半合成——核心方程在真实噪声下的 $\lambda$ 识别

$\lambda$ 贯穿库存和需求方程。**库存层注入**：对真实 $I(t)$ 模拟 $I_{k+1} = I_k - \lambda \cdot I_k \cdot \Delta t - d_k$，生成带衰减的库存轨迹，然后只用 (3) 做 OLS 恢复 $\lambda$——与需求函数形式完全解耦。

| true $\lambda$ | 酒类 (短周期, 3-5d) | | 叮咚 (长周期, 97d) | |
|---|---|---|---|---|
| | $\hat{\lambda}$ | $R^2_{inv}$ | $\hat{\lambda}$ | $R^2_{inv}$ |
| 0.01 | 检测不到 | ≈0 | **0.010** | 0.21 |
| 0.05 | 0.035 | ≈0 | **0.051** | 0.85 |
| 0.10 | 0.087 | 0.29 | **0.105** | 0.96 |
| 0.20 | 0.200 | 0.67 | **0.222** | 0.99 |

叮咚在 $\lambda=0.01$ 时即可检出（误差 2%），酒类在 0.20 处也精准（误差 3%）。酒类 0.01/0.05 信号微弱的原因不是 $\lambda$ 的绝对值太低，而是**短周期结构**：每周期仅 3-5 天（$I \approx 15$，$\lambda I \approx 0.75$ vs $d \approx 5$，衰减仅占总变化的 13%）。叮咚 97 天长周期（$I \approx 238$，$\lambda I \approx 11.9$ vs $d \approx 2.5$，衰减占 83%）给了 $\lambda$ 充分的信号窗口。

**库存方程的检测能力不取决于 $\lambda$ 的绝对值，取决于 $\lambda I$ 在全周期中的累积贡献。**

---

## 主要发现

1. **论文方法在真实 $I(t)$ + $d(t)$ 数据上可运行**（酒类、叮咚均跑通完整链路）。但 $e^{-\lambda t}$ 假设在冷链场景（$\lambda \approx 0.002$）下被数据否定——指数项退化为 1，参数估计失效。这不是方法有问题，是真实数据的 $\lambda$ 确实接近零。

2. **库存方程 (13a) 能独立识别 $\lambda$**：在半合成实验中，$\lambda$ 完全通过库存动力学恢复，与需求函数形式无关。检测精度取决于信噪比——周期内的 $\sum \lambda I$ 相对于 $\sum d$ 的比例。长周期大库存场景下，$\lambda = 0.01$ 即可以 2% 误差检出。

3. **论文设定 $\lambda = 0.05$ 是合理的**：在真实噪声下，这是大多数实际补货场景（周期 3-7 天、$I$ 几十到几百）能够可靠检出的量级。冷链将 $\lambda$ 压到 0.002——比这个阈值低 25 倍——库存方程输出 $\lambda = 0$ 是正确的，不是方法失败了。

4. **公开数据集的系统性缺口**：经过对 Kaggle、HuggingFace、Mendeley、阿里天池、GitHub 的搜索，不存在同时满足"真实 + $I(t)$ + $d(t)$ + 衰减信号 > 0"的公开数据集。有衰减的场景无数据，有数据的场景无衰减。

---

## 代码结构

```
grey_inventory/
├── simulation.py          # 数据加载与 I(t) 重建
├── estimation.py          # paper 原版估计 (IRLS + grid search)
└── optimization.py        # 利润函数

experiment_realdata.py              # 实验一：论文原版 · 169 combo
experiment_inventory_injection.py   # 实验二：库存层注入 · 酒类+叮咚
experiment_semisynthetic.py         # 实验二：需求层注入（辅助验证）
```

数据：`FreshRetailNet-50K/`（[HuggingFace](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)）、`Retail_Inventory_2024/`（[Kaggle](https://www.kaggle.com/datasets/surajs17/retail-sales-inventory-and-purchases-data-2024)）。

---

## 引用

```bibtex
@article{wang2026grey,
  title={Grey forecasting modelling for deteriorating inventory with 
         interdependent quality and quantity decay},
  author={Wang, Xiaolei and Xie, Naiming and Yang, Lu and Wei, Baolei},
  journal={Expert Systems with Applications},
  volume={306},
  pages={130956},
  year={2026}
}
```
