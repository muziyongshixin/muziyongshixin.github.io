---
title: '大模型面试手撕题全攻略：Attention、Transformer、归一化与损失函数'
date: 2026-04-22
permalink: /posts/2026/04/llm-interview-implementations/
categories:
  - blog
tags:
  - llm
  - transformer
  - attention
  - interview
  - machine learning
toc: true
---

> 大模型算法岗面试中，手撕代码是几乎绕不过去的一环。面试官会盯着你从零实现 Attention、MHA、GQA、LayerNorm、RMSNorm、SafeSoftmax、Cross-Entropy 等模块，既考察你对原理的理解，也考察你是否能在紧张的环境下把数值稳定性、维度对齐、broadcasting 这些细节处理干净。
>
> 这篇文章把这些高频手撕题系统梳理一遍：每一节都给出**核心原理 → 数学公式 → 从零手写的 PyTorch 实现 → 面试容易追问的点**，读完之后这一类题你应该都能在白板上 10 分钟内写出来。

---

## 1. Self-Attention：所有 Transformer 的起点

### 1.1 核心思想

Self-Attention 要回答的问题非常简单：

> 给定一个序列里的每个 token，它应该从其它 token 里"抄"多少信息过来？

它的三件套是 `Query`、`Key`、`Value`：

- `Query`：当前 token 想"找什么"
- `Key`：其它 token"能提供什么"（用来被检索）
- `Value`：其它 token"真正要传递的内容"

计算流程就一句话：**Query 和 Key 做点积得到相似度，softmax 归一化后再加权求和 Value**。

### 1.2 数学公式

标准的 Scaled Dot-Product Attention：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中：

- $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$
- $n$ 是 query 序列长度，$m$ 是 key/value 序列长度
- $d_k$ 是每个头的维度

**为什么要除以 $\sqrt{d_k}$？**

当 $d_k$ 很大时，$QK^\top$ 的方差会随着 $d_k$ 线性增长。点积数值过大会让 softmax 落到极端区域——梯度趋近 0，模型训不动。除以 $\sqrt{d_k}$ 可以把方差拉回 $O(1)$ 的量级。

简单推导：假设 $q$ 和 $k$ 每个分量均值 0、方差 1 且独立，那么

$$
\text{Var}(q \cdot k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k
$$

所以除以 $\sqrt{d_k}$ 后方差变回 1。

### 1.3 从零手撕实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [B, n, d_k]
    K: [B, m, d_k]
    V: [B, m, d_v]
    mask: [B, n, m]  True 的位置会被屏蔽
    """
    d_k = Q.size(-1)
    # [B, n, m]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)       # [B, n, m]
    out = torch.matmul(attn, V)            # [B, n, d_v]
    return out, attn
```

### 1.4 面试常见追问

- **Q：为什么用点积而不是加性注意力（Bahdanau）？**
  点积可以用矩阵乘法高效实现，GPU 友好；加性注意力多一层线性变换，不利于大规模并行。

- **Q：mask 怎么做？**
  Decoder 的 causal mask 是一个上三角为 True 的矩阵；Padding mask 是把 padding 位置置为 True。二者常合并使用。

- **Q：为什么要用 softmax 而不是别的归一化？**
  softmax 保证权重非负且和为 1，符合"加权平均"的语义；同时它是可导的。

---

## 2. Multi-Head Attention (MHA)

### 2.1 为什么要多头？

单头注意力只能学到一种"相似度"模式。多头允许模型在**不同的子空间里关注不同类型的关系**——比如一个头学句法，一个头学语义，一个头学远距离依赖。

### 2.2 数学公式

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

其中每个头独立计算：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

维度上：

- 输入维度 $d_{\text{model}}$，头数 $h$，每头维度 $d_k = d_{\text{model}} / h$
- $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$

注意：**参数总量不变**——切成 $h$ 个头之后每个头更"瘦"，但拼起来总维度还是 $d_{\text{model}}$。

### 2.3 从零手撕实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 一次性投影，效率更高
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q: [B, n, d_model]
        k, v: [B, m, d_model]
        mask: [B, 1, n, m] 或 [B, n, m]
        """
        B, n, _ = q.shape
        m = k.size(1)

        # 1. 线性投影
        Q = self.W_q(q)  # [B, n, d_model]
        K = self.W_k(k)
        V = self.W_v(v)

        # 2. 切分成多头: [B, n, d_model] -> [B, h, n, d_k]
        Q = Q.view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, m, self.num_heads, self.d_k).transpose(1, 2) # [B, h, m, d_k]
        V = V.view(B, m, self.num_heads, self.d_k).transpose(1, 2) # [B, h, m, d_k]

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [B, h, n, m]
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)   # 广播到 head 维
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1) # [B,h,n,m]
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)        # attn [B, h, n, m] V: [B, h, m, d_k] -> [B, h, n, d_k]

        # 4. 拼回来: [B, h, n, d_k] -> [B, n, d_model]
        out = out.transpose(1, 2).contiguous().view(B, n, self.d_model)

        # 5. 输出投影
        return self.W_o(out)
```

### 2.4 面试常见追问

- **Q：头数 h 越多越好吗？**
  不一定。头数过多会让每个头的 $d_k$ 过小，表达能力反而下降；同时 KV Cache 也会膨胀。实际工程通常 32~64 个头。

- **Q：MHA 的时间复杂度？**
  $O(n^2 \cdot d_{\text{model}})$，序列长度是瓶颈。这也是 Flash Attention、线性 Attention 等工作的优化对象。

---

## 3. Grouped-Query Attention (GQA)

### 3.1 为什么要 GQA？

推理阶段的显存瓶颈主要是 **KV Cache**——每一步解码都要存下所有历史 token 的 K 和 V。

- MHA：每个 Q 头都有独立的 K、V 头，KV Cache = $2 \cdot n \cdot h \cdot d_k$
- MQA（Multi-Query Attention）：所有 Q 头共享一组 K、V，KV Cache 缩小 $h$ 倍，但质量下降
- **GQA**：折中方案——Q 头分成 $g$ 组，每组共享一组 K、V

当 $g = h$ 就是 MHA，当 $g = 1$ 就是 MQA。LLaMA-2/3、Mixtral 等主流开源模型都用的是 GQA。

<br />
<img align="center" width="800" src="{{ site.url }}/images/posts/gqa-comparison.svg" alt="MHA vs GQA vs MQA comparison">
<br />

### 3.2 数学形式

设 Q 头数为 $h$，KV 头组数为 $g$，每组包含 $h / g$ 个 Q 头共享同一对 K、V：

$$
\text{head}_i = \text{Attention}(Q_i, K_{\lfloor i / (h/g) \rfloor}, V_{\lfloor i / (h/g) \rfloor})
$$

### 3.3 从零手撕实现

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_groups == 0, "头数必须能被组数整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.d_k = d_model // num_heads
        self.group_size = num_heads // num_kv_groups

        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        # K, V 只用组数个头
        self.W_k = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, n, _ = q.shape
        m = k.size(1)

        # Q: [B, h,  n, d_k]
        # K,V: [B, g, m, d_k]
        Q = self.W_q(q).view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, m, self.num_kv_groups, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, m, self.num_kv_groups, self.d_k).transpose(1, 2)

        # 把 K, V 在头维度上"重复 group_size 次"对齐到 Q
        # [B, g, m, d_k] -> [B, h, m, d_k]
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, n, self.d_model)
        return self.W_o(out)
```

**实现细节**：`repeat_interleave` 在工程上其实可以避免——直接用 einsum 或在 attention 计算里广播更省显存。但面试场景下写 `repeat_interleave` 更直观易读。

### 3.4 面试常见追问

- **Q：GQA 为什么比 MQA 效果好？**
  MQA 所有 Q 头共享一套 K、V，表达能力瓶颈明显；GQA 保留了多组 K、V，能在"显存"和"表达"之间做权衡。

- **Q：KV Cache 具体省多少？**
  对 LLaMA-2-70B，MHA KV Cache 每层是 $2 \cdot 64 \cdot d_k$，GQA 是 $2 \cdot 8 \cdot d_k$，直接省 8 倍。

---

## 4. Transformer Encoder 模块

### 4.1 结构图

一个完整的 Transformer Encoder Block 由以下部分组成：

```
x ──▶ LayerNorm ──▶ MHA ──▶ + ──▶ LayerNorm ──▶ FFN ──▶ + ──▶ out
  │                          ▲                           ▲
  └──────────────────────────┘                           │
                │                                         │
                └─────────────────────────────────────────┘
```

这是 Pre-Norm 结构（GPT、LLaMA 都用这种）。原版 Transformer 用 Post-Norm，训练稳定性差一些，现在主流都切到了 Pre-Norm。

### 4.2 公式

$$
\begin{aligned}
z &= x + \text{MHA}(\text{LN}(x)) \\
y &= z + \text{FFN}(\text{LN}(z))
\end{aligned}
$$

其中 FFN 通常是：

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

中间维度一般取 $4 \times d_{\text{model}}$。

### 4.3 从零手撕实现

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-Norm + Residual
        h = self.ln1(x)
        x = x + self.dropout(self.attn(h, h, h, mask=mask))

        h = self.ln2(x)
        x = x + self.dropout(self.ffn(h))
        return x
```

### 4.4 面试常见追问

- **Q：Pre-Norm 和 Post-Norm 的区别？**
  Post-Norm（原版）：$y = \text{LN}(x + \text{Sublayer}(x))$，残差链路被 LN 截断，深层训练不稳定。
  Pre-Norm：$y = x + \text{Sublayer}(\text{LN}(x))$，残差是"干净"的恒等映射，梯度更稳，但可能轻微损失表达能力。

- **Q：FFN 中间维度为什么是 4 倍？**
  经验值。它提供了模型的主要参数量（约占 2/3），也是存储世界知识的主要场所。

- **Q：GELU 和 ReLU 的区别？**
  GELU 是 $x \cdot \Phi(x)$，平滑版 ReLU，负半轴有小幅激活，在语言模型上效果更好。LLaMA 进一步用 SwiGLU。

---

## 5. LayerNorm vs BatchNorm

### 5.1 区别一句话

- **BatchNorm**：对每个**特征维度**，在 **batch 维度**上算均值和方差
- **LayerNorm**：对每个**样本**，在**特征维度**上算均值和方差

### 5.2 公式

设输入 $x \in \mathbb{R}^{B \times L \times D}$（Batch × SeqLen × Dim）。

**BatchNorm**（对 NLP 几乎不用）：

$$
\mu_d = \frac{1}{B \cdot L} \sum_{b, l} x_{b, l, d}, \quad
\sigma_d^2 = \frac{1}{B \cdot L} \sum_{b, l} (x_{b, l, d} - \mu_d)^2
$$

$$
\hat{x}_{b, l, d} = \frac{x_{b, l, d} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}}, \quad
y = \gamma \hat{x} + \beta
$$

**LayerNorm**：

$$
\mu_{b, l} = \frac{1}{D} \sum_d x_{b, l, d}, \quad
\sigma_{b, l}^2 = \frac{1}{D} \sum_d (x_{b, l, d} - \mu_{b, l})^2
$$

$$
\hat{x}_{b, l, d} = \frac{x_{b, l, d} - \mu_{b, l}}{\sqrt{\sigma_{b, l}^2 + \epsilon}}, \quad
y = \gamma \hat{x} + \beta
$$

注意 LayerNorm 的 $\gamma, \beta$ 是 $D$ 维向量，不依赖 batch 和 seq。

### 5.3 为什么 NLP 用 LayerNorm 而不是 BatchNorm？

1. **变长序列**：NLP 输入有大量 padding，padding 位置参与 BN 统计会污染结果。
2. **小 batch**：语言模型 batch 常常不大（长序列更吃显存），BN 在小 batch 上统计量不稳。
3. **训练/推理一致**：BN 推理时用 running mean/var，语言模型分布漂移敏感；LN 训推完全一致。
4. **每个 token 独立归一化**更贴合语言模型"逐 token 建模"的直觉。

### 5.4 从零手撕实现

```python
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: [..., dim]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        # x: [B, D]
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            # 更新 running 统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
```

**注意**：`unbiased=False` 表示用 $1/N$ 而不是 $1/(N-1)$，这是神经网络里常用的有偏估计，和 PyTorch 默认实现一致。

---

## 6. RMSNorm：LLaMA 时代的归一化标配

### 6.1 动机

LayerNorm 做了两件事：**减均值**（中心化） + **除标准差**（缩放）。
但有研究发现，**减均值这一步对性能的贡献非常小**——真正起作用的是"缩放"。

于是 RMSNorm 提出：直接去掉减均值，只做缩放，用 RMS（Root Mean Square）替代标准差：

- 省掉一次均值计算和减法
- 实测 7%~64% 的速度提升
- 效果和 LayerNorm 持平甚至更好

LLaMA、LLaMA-2/3、Mistral、Qwen 等主流模型全都用 RMSNorm。

### 6.2 公式

$$
\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{d=1}^{D} x_d^2}
$$

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x) + \epsilon} \odot \gamma
$$

注意：

- 没有 $\beta$（平移项）
- 分母里是 RMS 而不是 std
- $\gamma$ 是可学习的缩放向量

### 6.3 和 LayerNorm 的关系

如果 $x$ 的均值恰好为 0，那么 $\text{RMS}(x) = \text{std}(x)$，RMSNorm 就退化为没有 bias 的 LayerNorm。

换句话说，**RMSNorm = LayerNorm 扔掉均值平移**。

### 6.4 从零手撕实现

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [..., dim]
        # 注意用 float32 算 rms 避免 fp16 下溢
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.gamma
```

**工程细节**：
- `rsqrt()` 比 `1 / sqrt()` 更快
- 混合精度下先 cast 到 fp32 计算，结果再 cast 回原 dtype，可以避免数值不稳定
- eps 通常取 $10^{-6}$ 或 $10^{-5}$

### 6.5 面试常见追问

- **Q：RMSNorm 为什么比 LayerNorm 快？**
  少了一次"求均值 + 做减法"的操作，访存和计算都减半。

- **Q：去掉 $\beta$ 不会损失表达能力吗？**
  理论上会，但实验发现对语言模型几乎无影响。可能原因是残差连接本身已经提供了足够的 bias 能力。

---

## 7. Safe Softmax：数值稳定性的必考点

### 7.1 朴素 Softmax 的问题

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

当 $x_i$ 很大时（比如 1000），$e^{x_i}$ 直接 **上溢为 inf**；当 $x_i$ 很小时，$e^{x_i}$ 下溢为 0。fp16/bf16 下这个问题尤其严重（fp16 的最大值只有 65504）。

### 7.2 Safe Softmax 技巧

利用 softmax 的**平移不变性**：

$$
\text{softmax}(x_i) = \text{softmax}(x_i - c)
$$

因为分子分母同时乘 $e^{-c}$ 会约掉。所以我们可以取 $c = \max_j x_j$：

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max_j x_j}}{\sum_j e^{x_j - \max_j x_j}}
$$

这样：

- 指数的最大值变成 $e^0 = 1$，永远不会上溢
- 分母至少有一项是 1，不会下溢为 0

### 7.3 从零手撕实现

```python
def safe_softmax(x, dim=-1):
    """
    数值稳定的 softmax
    x: 任意 shape 的 tensor
    dim: 在哪个维度做 softmax
    """
    # 减去最大值防溢出（注意 detach/不 detach 都不影响梯度，因为是平移不变的）
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max

    exp_x = torch.exp(x_shifted)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp


def safe_log_softmax(x, dim=-1):
    """
    数值稳定的 log_softmax，后面算交叉熵要用到
    log_softmax(x) = x - max - log(sum(exp(x - max)))
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    # log-sum-exp
    log_sum_exp = torch.log(torch.exp(x_shifted).sum(dim=dim, keepdim=True))
    return x_shifted - log_sum_exp
```

### 7.4 面试常见追问

- **Q：减最大值这个操作会影响梯度吗？**
  不会。softmax 对平移不变，数学上完全等价，梯度也完全等价。

- **Q：如果所有输入都是 -inf（比如全被 mask）怎么办？**
  会出现 NaN（0/0）。实际实现里对 attention 会保证至少有一个非 mask 的位置，或者在 softmax 之后把整行置零。

- **Q：Flash Attention 里的 online softmax 是什么？**
  Flash Attention 在 tiling 过程中不能一次看到所有的 logits，它用 **递推公式** 维护当前已见的最大值和分母，逐块更新。这是 safe softmax 的分块在线版本。

---

## 8. Cross-Entropy Loss：分类任务的灵魂

### 8.1 公式

对于 $C$ 分类问题，模型输出 logits $z \in \mathbb{R}^C$，真实标签 $y \in \{0, 1, \dots, C-1\}$。

先 softmax 得到概率：

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

交叉熵损失：

$$
\mathcal{L} = -\log p_y = -\log \frac{e^{z_y}}{\sum_j e^{z_j}} = -z_y + \log \sum_j e^{z_j}
$$

最右边的形式就是我们常说的 **LogSumExp** 形式，非常适合数值稳定实现。

**批量版**：

$$
\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \log p_{n, y_n}
$$

### 8.2 为什么要用交叉熵而不是 MSE？

1. **梯度性质好**：对 logits 求导，$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \mathbb{1}[i = y]$，形式简洁且不会出现"梯度消失"。
2. **概率语义契合**：交叉熵衡量两个分布的距离，和 softmax 输出的"概率"天然配对。
3. **MSE 配 softmax 会梯度饱和**：预测很离谱时梯度反而很小，训练慢。

### 8.3 从零手撕实现

```python
def cross_entropy_loss(logits, targets, reduction='mean'):
    """
    logits: [N, C]  未经 softmax 的原始输出
    targets: [N]    每个样本的真实标签 index
    """
    # 用 log_softmax 的形式，避免 log(0)
    log_probs = safe_log_softmax(logits, dim=-1)  # [N, C]

    # gather 出真实标签对应的 log_prob
    # log_probs.gather(1, targets.unsqueeze(1)).squeeze(1): [N]
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll
```

### 8.4 带 Label Smoothing 的版本

大模型训练里经常用到 Label Smoothing：不把真实标签当成 one-hot，而是留一点点给别的类别，防止模型"过分自信"。

$$
\tilde{y}_i = \begin{cases}
1 - \epsilon + \epsilon / C & i = y \\
\epsilon / C & i \neq y
\end{cases}
$$

$$
\mathcal{L} = -\sum_{i} \tilde{y}_i \log p_i
$$

```python
def cross_entropy_with_label_smoothing(logits, targets, smoothing=0.1):
    N, C = logits.shape
    log_probs = safe_log_softmax(logits, dim=-1)

    # 构造平滑后的标签分布
    with torch.no_grad():
        true_dist = torch.full_like(log_probs, smoothing / C)
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - smoothing + smoothing / C)

    # -(true_dist * log_probs).sum(dim=-1).mean()
    return -(true_dist * log_probs).sum(dim=-1).mean()
```

### 8.5 面试常见追问

- **Q：PyTorch 的 `nn.CrossEntropyLoss` 输入是 logits 还是概率？**
  logits！它内部融合了 log_softmax + nll_loss，比手写分开两步更稳定、更快。

- **Q：对 logits 求导的结果推一下？**
  $\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \mathbb{1}[i = y]$，这就是为什么反向传播时只需要"预测概率减去 one-hot"。

- **Q：语言模型训练时怎么处理 padding 的 loss？**
  用 `ignore_index`（PyTorch 原生支持），或者构造 mask 在 reduction 之前把 padding 位置的 loss 置零。

---

## 9. 一个完整的"白板样板"

最后给一个浓缩版的"应急套路"——如果面试官让你 5 分钟手撕一个精简 Transformer Block，就按下面这个最小实现来：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        q = self.wq(x).view(B, N, self.h, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, N, self.h, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, N, self.h, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.wo(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.n2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.ffn(self.n2(x))
        return x
```

把这套板子背熟，再根据面试官追问填 GQA、KV Cache、RoPE 等变种即可。

---

## 10. 总结

本文把大模型算法岗最常考的一组手撕题串起来：

| 模块 | 一句话总结 |
|------|----------|
| Self-Attention | Q·Kᵀ / √d_k 再 softmax 加权 V |
| MHA | 切成 h 个头并行算 Attention，拼回再投影 |
| GQA | Q 独立 h 个头，KV 只有 g 组头，省 KV Cache |
| Transformer Encoder | Pre-Norm + MHA + Pre-Norm + FFN，两段残差 |
| LayerNorm | 每个样本在特征维上做归一化 |
| BatchNorm | 每个特征在 batch 维上做归一化（NLP 不用） |
| RMSNorm | LayerNorm 去掉均值平移，只保留 RMS 缩放 |
| Safe Softmax | 减去最大值再做 exp，避免上溢 |
| Cross-Entropy | $-\log p_y$，实战用 log_softmax + nll_loss |

建议的学习路径：先把每一节的公式自己手推一遍，再白板默写实现，然后用 `torch.allclose` 和 `nn` 自带模块对拍一下数值，最后在纸上做时空复杂度分析。真正把这一整套走完之后，这类题目你都能在面试里淡定应付了。

---

## 参考资料

1. Vaswani et al., *Attention is All You Need*, 2017
2. Ba et al., *Layer Normalization*, 2016
3. Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019
4. Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023
5. Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023
