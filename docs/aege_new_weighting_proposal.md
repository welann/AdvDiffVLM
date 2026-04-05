# 针对 AEGE 的新梯度权重计算方案设想

## 1. 文档目的

这份文档提出一个可替代 AEGE 当前权重计算方式的新方案，目标不是简单“改个经验公式”，而是尽量从数学上回答下面这个问题：

> 当多个 surrogate 模型都在给我们提供目标梯度时，怎样给每个梯度分配权重，才能让最终聚合梯度更接近“真实可迁移目标梯度”？

我这里提出的方案叫：

**共识-稳定性逆方差加权**

英文可记作：

**Consensus-Stability Inverse-Variance Weighting, CSIVW**

它的核心思想是：

1. 把每个 surrogate 提供的梯度看成“真实目标梯度的一个带噪估计”；
2. 估计每个 surrogate 当前的“梯度不确定性”；
3. 用逆方差加权的方式自动降低高噪声 surrogate 的影响；
4. 让权重计算直接基于梯度几何关系，而不是只基于 loss 标量变化。

## 2. 为什么我认为当前 AEGE 还可以继续改

论文里的 AEGE，以及当前代码里的 AEGE，本质上都在做一件事：

- 根据每个 surrogate 的最近损失变化情况，动态调整权重。

这当然比简单平均更好，但它有一个结构性缺点：

> 它用来分配权重的是 loss 的变化，而最终真正被聚合的是 gradient。

这两者并不总是对齐。

### 2.1 当前 AEGE 的主要问题

如果我们只看 loss 变化，会忽略至少 4 类对 transferability 很关键的信息：

1. **方向一致性**
   一个 surrogate 的 loss 变化很快，不代表它的梯度方向和其他 surrogate 一致。

2. **时间稳定性**
   一个 surrogate 当前步的 loss 提升很大，不代表它在相邻时间步上的梯度稳定。

3. **梯度范数异常**
   某个 surrogate 可能只是梯度特别大，于是损失变化也大，但这可能是“爆炸梯度”而不是“好梯度”。

4. **目标真正需要的是“共识方向”**
   对 transfer-based attack 来说，最有可能迁移出去的，不是某个 surrogate 私有的方向，而是多个 surrogate 共同支持的方向。

所以从攻击机制上说，AEGE 更合理的升级方向应该是：

> 直接对“梯度估计的可靠性”建模，而不是只对“损失变化快不快”建模。

## 3. 新方案的基本建模思路

### 3.1 把 surrogate 梯度看成真实梯度的带噪观测

设在第 `t` 个 reverse step，有 `M` 个 surrogate 模型。第 `i` 个 surrogate 给出的梯度记为：

```math
g_i^{(t)} = \nabla_x L_i^{(t)}
```

其中：

- `x` 表示当前优化变量，具体可对应当前 latent `\tilde{x}_t`
- `L_i^{(t)}` 表示第 `i` 个 surrogate 的 targeted loss
- `g_i^{(t)}` 是该 surrogate 对当前样本给出的梯度

我们假设存在一个不可见的“理想可迁移梯度”：

```math
g_*^{(t)}
```

每个 surrogate 给出的梯度都只是它的一个带噪估计：

```math
g_i^{(t)} = g_*^{(t)} + \varepsilon_i^{(t)}
```

其中 `\varepsilon_i^{(t)}` 表示 surrogate 的估计误差。

如果某个 surrogate 当前误差更大，那么它就不应该拥有更大的权重。

### 3.2 我们真正要解的优化问题

我们希望构造一个加权聚合梯度：

```math
g_{\text{ens}}^{(t)} = \sum_{i=1}^{M} w_i^{(t)} g_i^{(t)}
```

满足：

```math
\sum_{i=1}^{M} w_i^{(t)} = 1, \quad w_i^{(t)} \ge 0
```

并且让它尽可能接近真实梯度 `g_*^{(t)}`。

最自然的目标就是最小化均方误差：

```math
\min_{w \in \Delta^{M-1}} \mathbb{E}\left[\left\| \sum_{i=1}^{M} w_i g_i - g_* \right\|_2^2 \right]
```

其中 `\Delta^{M-1}` 是概率单纯形。

## 4. 逆方差加权的数学推导

### 4.1 在独立同方差噪声近似下的最优权重

如果进一步假设：

1. 各 surrogate 的误差均值为 0：

```math
\mathbb{E}[\varepsilon_i] = 0
```

2. 不同 surrogate 的误差相互独立：

```math
\mathbb{E}[\varepsilon_i^\top \varepsilon_j] = 0, \quad i \ne j
```

3. 每个 surrogate 的误差协方差近似各向同性：

```math
\mathrm{Cov}(\varepsilon_i) \approx \sigma_i^2 I
```

那么有：

```math
\mathbb{E}\left[\left\| \sum_i w_i g_i - g_* \right\|_2^2 \right]
= \mathbb{E}\left[\left\| \sum_i w_i \varepsilon_i \right\|_2^2 \right]
= \sum_i w_i^2 \, d \sigma_i^2
```

这里 `d` 是梯度维度，和权重优化无关，所以等价于解：

```math
\min_{w} \sum_{i=1}^{M} w_i^2 \sigma_i^2
\quad
\text{s.t. } \sum_i w_i = 1,\; w_i \ge 0
```

写出拉格朗日函数：

```math
\mathcal{L}(w,\lambda)=\sum_i w_i^2 \sigma_i^2 + \lambda\left(\sum_i w_i - 1\right)
```

对 `w_i` 求导：

```math
\frac{\partial \mathcal{L}}{\partial w_i} = 2 w_i \sigma_i^2 + \lambda = 0
```

所以：

```math
w_i = -\frac{\lambda}{2\sigma_i^2}
```

代入约束 `\sum_i w_i = 1` 可得：

```math
w_i^* = \frac{\sigma_i^{-2}}{\sum_{j=1}^{M} \sigma_j^{-2}}
```

这就是经典的 **逆方差加权**。

### 4.2 这说明了什么

它说明如果我们能估计出每个 surrogate 当前梯度噪声有多大，那么最合理的做法不是：

- 看谁 loss 涨得快；

而是：

- 看谁当前的梯度不确定性更小。

所以整个问题就变成：

> 如何在线估计 surrogate 梯度的“方差”或“不确定性”？

## 5. 如何在线估计 surrogate 的梯度不确定性

真实方差 `\sigma_i^2` 我们当然拿不到，所以只能构造代理量。我的建议是用 4 个量来估计：

1. **跨 surrogate 的方向分歧**
2. **单个 surrogate 的时间抖动**
3. **梯度范数离群程度**
4. **loss 进展的有效性**

这 4 个量分别对应 4 种“不可靠”的情况。

---

### 5.1 方向分歧项：Consensus Disagreement

先定义归一化梯度方向：

```math
u_i^{(t)} = \frac{g_i^{(t)}}{\|g_i^{(t)}\|_2 + \epsilon}
```

然后定义第 `i` 个 surrogate 与其他 surrogate 的平均方向一致性：

```math
C_i^{(t)} =
\frac{1}{M-1}
\sum_{j \ne i}
\max\left(0, \langle u_i^{(t)}, u_j^{(t)} \rangle \right)
```

这里：

- `C_i^{(t)}` 越大，说明它和其他模型方向越一致；
- `max(0, \cdot)` 是为了避免一个完全反向的 surrogate 通过负值把统计量搞得太极端。

那么它的方向分歧可以定义为：

```math
D_i^{(t)} = 1 - C_i^{(t)}
```

#### 数学直觉

如果每个 surrogate 都是在真实方向附近带小角度噪声摆动，那么：

```math
\cos(\theta_{ij}) \approx 1 - \frac{1}{2}(\theta_i^2 + \theta_j^2)
```

因此，和其他 surrogate 的平均余弦越小，通常意味着该 surrogate 的角度噪声越大。

所以 `D_i^{(t)}` 可以看作 surrogate `i` 的 **角度方差代理项**。

---

### 5.2 时间抖动项：Temporal Instability

对同一个 surrogate，如果它相邻两步的梯度方向剧烈变化，那么它大概率不稳定。

定义时间一致性：

```math
S_i^{(t)} =
\max\left(0, \langle u_i^{(t)}, u_i^{(t-1)} \rangle \right)
```

于是时间抖动定义为：

```math
T_i^{(t)} = 1 - S_i^{(t)}
```

#### 数学直觉

如果某个 surrogate 真的在稳定地逼近目标方向，那么它在相邻 reverse steps 上的方向变化应该比较平滑；反之，如果它大起大落，则更像是局部噪声主导。

所以 `T_i^{(t)}` 可以看作 surrogate `i` 的 **时间方差代理项**。

---

### 5.3 范数离群项：Magnitude Outlier

有些 surrogate 会给出范数异常大的梯度，但“大”未必代表“对”。

为了避免某个 surrogate 单靠梯度爆炸抢权重，可以定义：

```math
n_i^{(t)} = \log(\|g_i^{(t)}\|_2 + \epsilon)
```

再定义同一步所有 surrogate 的中位数范数：

```math
\tilde{n}^{(t)} = \mathrm{median}\{n_1^{(t)}, \dots, n_M^{(t)}\}
```

则第 `i` 个 surrogate 的范数离群项为：

```math
M_i^{(t)} = | n_i^{(t)} - \tilde{n}^{(t)} |
```

#### 数学直觉

如果一个 surrogate 的梯度模长和其他 surrogate 差异特别大，那么它更像是 outlier estimator。用 log 范数是为了减小极端值影响。

---

### 5.4 有效进展项：Progress Failure

这个项保留 AEGE 里“loss 动态”这一思想，但不再把它作为唯一标准，只把它当作一个辅助不确定性指标。

令 targeted attack 下第 `i` 个 surrogate 的相对进展量为：

```math
\Delta L_i^{(t)} =
\frac{L_i^{(t)} - L_i^{(t-1)}}{|L_i^{(t-1)}| + \epsilon}
```

因为 targeted attack 里通常是在 **增大目标相似度**，所以 `\Delta L_i^{(t)}` 越大越好。

定义一个平滑的有效进展分数：

```math
P_i^{(t)} = \sigma(\kappa \Delta L_i^{(t)})
```

其中：

- `\sigma(\cdot)` 是 sigmoid
- `\kappa` 控制敏感度

则“进展不足”项为：

```math
F_i^{(t)} = 1 - P_i^{(t)}
```

#### 数学直觉

如果某个 surrogate 当前根本没有推动 targeted objective 前进，那么即使它和别人方向看起来还行，也不应该被过度信任。

## 6. 新的方差估计公式

综合上面 4 项，我建议把 surrogate `i` 在第 `t` 步的在线方差估计写成：

```math
\hat{\sigma}_i^2{}^{(t)}
=
\alpha D_i^{(t)}
 \beta T_i^{(t)}
 \gamma M_i^{(t)}
 \eta F_i^{(t)}
 \epsilon
```

其中：

- `\alpha`：控制跨模型方向分歧的重要性
- `\beta`：控制时间稳定性的重要性
- `\gamma`：控制范数离群惩罚
- `\eta`：控制 loss 有效进展约束
- `\epsilon`：数值稳定项

### 6.1 进一步加一个 EMA 平滑

为了避免单步剧烈波动，可以再对方差估计做指数滑动平均：

```math
\bar{\sigma}_i^2{}^{(t)}
=
\rho \bar{\sigma}_i^2{}^{(t-1)}
 (1-\rho)\hat{\sigma}_i^2{}^{(t)}
```

其中 `\rho \in [0,1)`。

这样得到的是一个更平滑的“历史不确定性估计”。

## 7. 最终权重公式

有了平滑后的方差估计，直接使用逆方差加权：

```math
w_i^{(t)} =
\frac{\left(\bar{\sigma}_i^2{}^{(t)}\right)^{-1}}
{\sum_{j=1}^{M}\left(\bar{\sigma}_j^2{}^{(t)}\right)^{-1}}
```

如果还希望保留温度超参，控制权重是否过于尖锐，可以写成温度化版本：

```math
w_i^{(t)} =
\frac{\exp\left(-\frac{\log \bar{\sigma}_i^2{}^{(t)}}{\tau}\right)}
{\sum_{j=1}^{M}\exp\left(-\frac{\log \bar{\sigma}_j^2{}^{(t)}}{\tau}\right)}
```

因为：

```math
\exp(-\log \sigma^2 / \tau) = (\sigma^2)^{-1/\tau}
```

所以它本质上是一个 **tempered inverse-variance weighting**：

- `\tau = 1`：标准逆方差加权
- `\tau > 1`：更平滑，不容易一家独大
- `\tau < 1`：更尖锐，更偏向少数高可靠 surrogate

## 8. 最终聚合梯度

新的聚合梯度可以写成：

```math
g_{\text{ens}}^{(t)} = \sum_{i=1}^{M} w_i^{(t)} \, \mathrm{clip}(g_i^{(t)}, -\delta, \delta)
```

然后再把它带回 score 修正项：

```math
\mathrm{score}^{(t)}

=
-\left(
\frac{\epsilon_\theta(\tilde{x}_t)}{\sqrt{1-\bar{\alpha}_t}}
 s \, g_{\text{ens}}^{(t)}
\right)
```

这里：

- `s` 是 adversarial gradient scale
- `\delta` 是梯度裁剪阈值

## 9. 这个新方案和当前 AEGE 的本质差别

### 9.1 当前 AEGE

当前 AEGE 更接近：

> 根据最近几步 loss 的相对变化，估计哪个 surrogate 更值得信任。

这是一个 **loss-dynamics-based weighting**。

### 9.2 新方案 CSIVW

CSIVW 更接近：

> 直接把 surrogate 梯度看作 noisy estimator，并根据其“方向分歧、时间稳定、范数离群、进展有效性”估计方差，再做逆方差融合。

这是一个 **gradient-reliability-based weighting**。

### 9.3 为什么我认为新方案更合理

因为最终真正被加到 score 里的量就是梯度，所以：

- 最好直接对梯度可靠性建模；
- 而不是只用 loss 变化去间接猜测梯度质量。

换句话说：

> AEGE 的当前版本更像“loss 层面的启发式动态加权”，而 CSIVW 更像“梯度层面的近似最优估计”。

## 10. 一个更直观的理解

假设 4 个 surrogate 给出的梯度情况如下：

1. 模型 A：方向和其他人都一致，且相邻两步很稳定
2. 模型 B：方向基本一致，但梯度模长偏大
3. 模型 C：loss 提升很快，但方向经常和其他人不一致
4. 模型 D：方向还行，但相邻时间步抖动很厉害

那么：

- 当前 AEGE 可能会偏向 C，因为它 loss 提升快
- CSIVW 会偏向 A，并适当保留 B，抑制 C 和 D

而从 transferability 角度看，A/B 通常更像“共识方向”，所以这种加权更合理。

## 11. 一个简化版本

如果不想一下子引入 4 个项，我建议先上一个最简版：

### 11.1 方差估计

```math
\hat{\sigma}_i^2{}^{(t)}
=
\alpha D_i^{(t)}
 \beta T_i^{(t)}
 \epsilon
```

其中：

- `D_i^{(t)}`：跨 surrogate 的方向分歧
- `T_i^{(t)}`：同一 surrogate 的时间抖动

### 11.2 权重

```math
w_i^{(t)}
=
\frac{(\hat{\sigma}_i^2{}^{(t)})^{-1}}
{\sum_j (\hat{\sigma}_j^2{}^{(t)})^{-1}}
```

这个版本已经比只看 loss ratio 更贴近梯度本身，而且实现代价不高。

## 12. 如何接到当前代码里

### 12.1 当前代码需要改的地方

当前 `ddim_main.py` 里，大致是：

1. 先算多个 surrogate 的 loss；
2. 用当前 `weights` 加权成总 loss；
3. 再对总 loss 求一次梯度。

如果要实现 CSIVW，需要改成：

1. 分别计算每个 surrogate 的 loss `L_i`
2. 分别计算每个 surrogate 的梯度 `g_i = ∇ L_i`
3. 根据 `g_i` 计算 `D_i, T_i, M_i, F_i`
4. 得到 `w_i`
5. 最后做
   `g_ens = Σ w_i g_i`

也就是说，和当前代码相比，最大的变化不是权重公式本身，而是：

> 权重计算需要访问每个 surrogate 的单独梯度，而不仅仅是单独的 loss。

### 12.2 伪代码

```python
for t in reverse_steps:
    grads = []
    losses = []
    for i in range(M):
        L_i = cosine_similarity(feature_i(x_t), target_i)
        g_i = grad(L_i, x_t)
        losses.append(L_i)
        grads.append(g_i)

    # 归一化方向
    u = [g / (norm(g) + eps) for g in grads]

    # 方向分歧
    D_i = 1 - mean_j max(0, cosine(u_i, u_j))

    # 时间抖动
    T_i = 1 - max(0, cosine(u_i_t, u_i_t_minus_1))

    # 范数离群
    M_i = abs(log(norm(g_i)) - median_j log(norm(g_j)))

    # 进展不足
    F_i = 1 - sigmoid(kappa * delta_loss_i)

    sigma2_i = alpha * D_i + beta * T_i + gamma * M_i + eta * F_i + eps
    sigma2_i_bar = rho * sigma2_i_bar_prev + (1-rho) * sigma2_i

    w_i = inv(sigma2_i_bar) / sum_j inv(sigma2_j_bar)
    g_ens = sum_i w_i * clip(g_i, -delta, delta)

    x_t = x_t + s * g_ens
```

## 13. 这个方案可能带来的优点

### 13.1 更贴近 transferability 的真实来源

可迁移攻击通常依赖多个 surrogate 共享的“共识方向”，不是某个 surrogate 单独的最优方向。CSIVW 直接把这个共识编码进了权重。

### 13.2 能主动抑制方向 outlier

如果某个 surrogate 给出的梯度和别人明显不一致，它会自动降权。

### 13.3 更稳

加上时间稳定项和 EMA 后，权重不会因为某一步 loss 突然变化就剧烈抖动。

### 13.4 有明确的估计论解释

它不只是经验上看起来合理，而是可以回到“多 noisy estimator 融合”的最小 MSE 原理来解释。

## 14. 这个方案可能的代价

### 14.1 计算量更大

要为每个 surrogate 分别计算梯度，而不是只算一个加权 loss 的梯度。

不过当前 `M=4`，而且 reverse process 后期步数有限，所以这个开销并不是不可接受。

### 14.2 需要存历史梯度方向

为了算时间稳定项 `T_i^{(t)}`，要为每个 surrogate 保存上一步方向。

### 14.3 超参数更多

需要设置：

- `\alpha, \beta, \gamma, \eta`
- `\rho`
- `\tau`
- `\kappa`

但这类超参大多可以通过消融实验逐步简化。

## 15. 如果让我实际落地，我会怎么做

我会按下面顺序推进，而不是一次把完整版本全上：

### 第一步：只上最简版

只用：

- 方向分歧 `D_i`
- 时间抖动 `T_i`

构造：

```math
\hat{\sigma}_i^2 = \alpha D_i + \beta T_i + \epsilon
```

### 第二步：加 EMA

如果发现权重太抖，再加：

```math
\bar{\sigma}_i^2 = \rho \bar{\sigma}_i^2_{\text{prev}} + (1-\rho)\hat{\sigma}_i^2
```

### 第三步：再补范数离群项

如果发现某些 surrogate 经常爆梯度，再加 `M_i`。

### 第四步：最后再考虑进展项

只有在实验中发现“方向一致但不推进目标”的情况明显存在时，再补 `F_i`。

## 16. 最终总结

如果只用一句话概括这个新方案，那就是：

> 不要再只根据 surrogate 的 loss 变化来分配权重，而应该把每个 surrogate 的梯度看成一个带噪估计器，根据它与其他 surrogate 的方向共识、时间稳定性和离群程度在线估计不确定性，再用逆方差加权得到更接近真实可迁移目标梯度的聚合方向。

从数学上看，这个方案的基础是：

1. 多估计器融合的最小均方误差原则；
2. 逆方差加权；
3. 用梯度几何统计量在线近似 surrogate 的噪声强度。

如果后续真的要把 AEGE 往更稳、更“像估计器融合”而不是“像启发式 loss 加权”的方向推进，我认为这条路线是值得优先尝试的。
