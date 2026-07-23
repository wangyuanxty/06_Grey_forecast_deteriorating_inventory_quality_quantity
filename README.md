# Grey Forecasting for Deteriorating Inventory — 真实数据验证与模型改进

> 原论文：*Grey forecasting modelling for deteriorating inventory with interdependent quality and quantity decay* (ESWA '26)
>
> 原仓库：[wangxiaolei0721/grey4deterioratinginventory](https://github.com/wangxiaolei0721/grey4deterioratinginventory)

## 动机

原论文提出了灰色预测建模方法，通过 AGO + 双方程联合回归从库存数据中估计衰减速率 λ、潜在需求 α 和价格敏感度 β，进而构建利润优化模型导出最优定价 p* 和补货周期 T*。论文在 70 条仿真数据上验证了参数估计精度（λ 误差 5.6%，α 误差 2.2%），但数据生成过程和模型共享同一个方程形式。**模型在真实数据上能否运行、核心参数 λ 能否被识别——这两个问题论文没有回答。**

本研究围绕三个问题展开：

1. **公开数据里是否存在适合验证该模型的真实数据集？**
2. **论文的 e^{-λt} 需求衰减假设在真实数据上是否成立？如果 λ≈0（冷链场景），如何调整模型结构？**
3. **论文的库存方程 (13a)——λ 估计的核心——在真实噪声下能否独立识别 λ？其检测边界在哪里？**

---

## 论文模型回顾

三个方程，三参数 (λ, α, β)：

$$\frac{dI}{dt} = -\lambda I(t) - d(t,p) \tag{微分方程}$$

$$d(t,p) = (\alpha - \beta p) \cdot e^{-\lambda t} \tag{需求函数}$$

$$l_j + d_j = -\lambda \cdot \frac{1}{2}(L_j + L_{j-1}) \cdot \Delta t \tag{库存方程 (13a)}$$

```math
d_j = \frac{\alpha - \beta p}{\lambda}[e^{-\lambda t_{j-1}} - e^{-\lambda t_j}] \tag{需求方程 (13b)}
```

需求方程取了对需求函数的积分形式，因为观测值是离散区间上的累加值。λ 同时出现在三个方程中：库存方程里的物理变质速率、需求方程里的品质衰减速率、利润函数里的成本折现——论文隐含假设三者是同一个 λ。AGO 将差分观测转化为水平值，梯形近似连接了离散数据和连续积分，库存方程 (13a) 给出 λ 的线性初值，IRLS 交替迭代收敛——从数据到方程再到决策决策。

四步流程：

```
AGO → 库存方程 OLS 得 λ₀ → 需求方程 NLS 得 α,β → IRLS 迭代 → 利润优化得 p*, T*
```

---

## 数据集搜索

模型需要：逐日库存 I(t)、逐日销量 d(t)、价格变化、产品有明显价值衰减。

| 数据集 | I(t) | d(t) | 价格 | 衰减信号 | 规模 | 结论 |
|--------|------|------|------|---------|------|------|
| **叮咚买菜 FreshRetailNet-50K** | 可重建 | 归一化金额 | discount | λ≈0.002 | 485 万行 | 冷链压制信号 |
| **酒类零售 2024** | 完整 | 真实件数 | 进价+售价 | λ=0 | ~1000 万行 | 品类不匹配 |
| **TAMD 淘宝服装** | 无 | 平台聚合 | 无 | 可能有 | 1765 行 | 数据太稀疏 |
| **花店订单** | 无 | 每笔一单 | 无 | 有（物理） | 106 行 | 太少 |
| **M5 Walmart** | 无 | 日销量 | 周价格 | 部分品类 | 3 万条 | 无 I(t) |

**结论：真实 + I(t) + 衰减信号 > 0 = 公开数据里不存在。** 有物理衰减的场景（菜市场、花店）没被数字化。被数字化的场景（冷链生鲜、酒类零售）衰减不可见。叮咚和酒类两个数据集互补——前者有品类 (生鲜) 但 λ≈0，后者有完整 I(t) 但 λ=0——各覆盖了论文模型所需的一半条件。

酒类数据的 I(t) 从 `begin_inventory + purchases − sales` 精确重建：`begin_inventory.csv` 和 `end_inventory.csv` 提供年度盘点数（onHand 字段），`purchases.csv` 记录每次采购的 ReceivingDate 和 Quantity，`sales.csv` 提供逐日 SalesQuantity——三账齐全，每个补货周期的起始库存和每日消耗可完全还原。叮咚只有缺货标记（0/1）和小时级销量，必须依赖缺货小时倒推——这给出了 I(t) 的下界（忽略了未销售的损耗），但对 λ≈0.002 的场景精度足够。

---

## 实验一：论文原版在真实数据上的表现

将论文原版 `d = (α-βp)·e^{-λt}` 在叮咚数据上逐产品训练，格点搜索 λ，OLS 拟合 α,β。叮咚的 discount（折扣率 1.0=原价）作为价格 p，归一化 sale_amount 作为标签。

训练/验证按时间 8:2 切分。

```
叮咚 169 个产品×门店组合：

R² 中位数：  −0.61
λ 中位数：   0.002
α 范围：     3.8 ~ 1327（爆炸）
R² > 0：     0/169
```

R² = −0.61 意味着模型预测不如直接猜平均值。α 波动范围从 3.8 到 1327——当 e^{-λt} ≈ 1（冷链 λ≈0），模型退化为常数预测，NLS 优化器将 α 推向荒谬值来补偿残差——参数完全失去经济含义。

酒类数据上 λ 正确收敛到 0（酒不会变质），α≈30~60（零价格日需求件数），β≈1.5~3（每涨价 1 美元少卖 2 件左右）。框架本身在真实数据上是可运行的——问题不在方法，在冷链。

---

## 实验二：Hybrid 模型——去掉 e^{-λt}

设计 Hybrid 替代模型：保留论文的 (α-βp) 价格因子和经济解释，将 e^{-λt} 替换为 NN 学习的需求倍率。NN 输入包括 t_rel（相对天数）、discount（价格）、day_of_week、holiday_flag、activity_flag、温度、湿度、降水、风力。无 lag 特征（不偷看昨天答案）。损失为 MSE(sale_amount/discount, pred)——用 discount 修正机械效应。

$$d(t,p,X) = (\alpha - \beta \cdot discount) \cdot NN(t, weekday, weather, holiday, ...)$$

| 模型 | R² val | 数据规模 | 说明 |
|------|--------|---------|------|
| 论文原版 | −0.61 | 169 combo × ~50 点 | 逐产品独立训练 |
| Hybrid v1 (小 NN，逐产品) | −0.02 | 97 点/combo | 40% 的 combo 拿到正 R² |
| Hybrid v2 (加 embedding，全量) | 0.68 | 895K 行，275 产品 × 866 门店 | product/store/city embedding 共享 |
| Hybrid 单门店 (去 store_id) | **0.59** | 4753 行，49 产品 | 门店 18，无 store_id，消除门店差异的标签泄露问题 |
| Hybrid 单门店 (y=sale_amount/discount) | 0.59 | 同上 | 修正机械效应后 R² 不变——α,β 捕捉的是行为效应 |

R² 从 −0.61 到 0.59 的关键改动：去掉 e^{-λt}（不再强制指数衰减），保留 (α-βp)（保留经济结构和利润闭式解），加 ID embedding（让模型知道是哪个产品、哪个品类），去 lag 特征（不偷看答案）。单门店版本自然消除了门店差异——所有产品共享同一批客流，α 自动吸收门店基线。

---

## 实验三：半合成验证——核心方程在真实噪声下的 λ 识别

论文的 λ 分布在库存和需求方程中。分开测试：

**库存层注入**：对真实 I(t) 执行微分方程模拟 `I_{k+1}=I_k−λ·I_k−d_k`，生成带衰减的库存轨迹。然后只用库存方程 (13a) 做 OLS 恢复 λ——λ 与需求函数形式完全解耦。

**需求层注入**（仅验证完整性）：对真实 d(t) 乘以 e^{-λt}，需求方程格点搜索估计 λ。

| true λ | 酒类 (短周期, 3-5d) | | 叮咚 (长周期, 97d) | |
|--------|---------|------|---------|------|
| | λ̂ | R²_inv | λ̂ | R²_inv |
| 0.01 | 检测不到 | ≈0 | **0.010** | 0.21 |
| 0.05 | 0.035 | ≈0 | **0.051** | 0.85 |
| 0.10 | 0.087 | 0.29 | **0.105** | 0.96 |
| 0.20 | 0.200 | 0.67 | **0.222** | 0.99 |

叮咚在 λ=0.01 时就能检出（误差 2%），而非之前认为的"0.05 是最低阈值"。先前结论的偏差源于酒类的短周期结构（每周期仅 3-5 天，I≈15，λI≈0.75 vs d≈5，衰减仅占总库存变化的 13%）。叮咚 97 天长周期（I≈238，λI≈11.9 vs d≈2.5，衰减占 83%）给了 λ 充分的信号窗口。**库存方程的检测能力不取决于 λ 的绝对值，取决于 λI 在全周期中的累积贡献。** 周期越长、初始库存越多，同一 λ 的信号越强。

需求层注入如预期表现完美（λ 误差 0%，R² 0.89~0.92）——注入的函数形式和估计方程同源——验证了论文的 NLS+OLS 框架在该函数形式下的收敛性。

---

## 代码结构

```
grey_inventory/
├── simulation.py          # 叮咚数据 I(t) 重建 + 酒类数据加载
├── estimation.py          # paper 原版 λ,O,A,B 估计 (IRLS + grid search)
├── optimization.py        # 利润函数精确版 + 泰勒近似
│
experiment_realdata.py          # 实验一：论文原版 · 169 combo · R²=−0.61
experiment_hybrid.py            # 实验二：Hybrid v1 逐产品小 NN
experiment_hybrid_v2.py         # 实验二：Hybrid v2 全量 embedding · R²=0.68
experiment_single_store.py      # 实验二：单门店 · 去 store_id · R²=0.59
experiment_full_pipeline.py     # 完整链路：小时数据 → I(t) → λ → α,β → p*
experiment_semisynthetic.py     # 实验三：需求层注入
experiment_inventory_injection.py  # 实验三：库存层注入 · 酒类+叮咚
```

数据：`FreshRetailNet-50K/`（HuggingFace, 485 万行）、`Retail_Inventory_2024/`（酒类, ~1000 万行）。

---

## 主要发现

1. **论文方法在真实 I(t)+d(t) 数据上可运行**（酒类、叮咚均跑通完整链路，利润全产品为正）。但 e^{-λt} 假设在冷链场景（λ≈0.002）下被数据否定，整个指数项退化为 1，参数估计爆炸或无效。

2. **(α-βp)·NN 替代 e^{-λt} 有效**：保留经济可解释性和利润函数的解析最优解，R² 从 −0.61 提升到 0.59。关键是不强制指数衰减函数形式，让 NN 从 weekday、天气、品类 embedding 等真实特征中学习需求模式。

3. **库存方程 (13a) 的 λ 识别有明确的信噪比条件**：在半合成注入实验中，λ 可被独立检出——前提是周期内初始库存 I₀ 足够大、周期长度足够长，使得累积衰减项 ΣλI 明显超过日销量波动 Σd。短周期 3-5 天信号被淹没；长周期 97 天 λ=0.01 即可检出（误差 2%）。这为实际部署中判断"能否从某个产品的库存数据里估计 λ"提供了定量标准。

4. **公开数据集的系统性缺口被确认**：经过对 Kaggle、HuggingFace、Mendeley、阿里天池、GitHub 的全面搜索，没有找到同时满足"真实、逐日库存 I(t)、逐日销量 d(t)、产品有明显价值衰减"的公开数据集。有衰减的场景无数据，有数据的场景无衰减。

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

@article{dingdong2025,
  title={FreshRetailNet-50K: A Benchmark for Censored Demand Estimation 
         in Fresh Retail},
  year={2025},
  note={HuggingFace: Dingdong-Inc/FreshRetailNet-50K}
}
```
