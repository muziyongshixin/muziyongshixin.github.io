---
title: '图解 Wan2.1 I2V：从一张图到一段视频，模型到底发生了什么'
date: 2026-04-24
permalink: /posts/2026/04/wan21-i2v-explained/
categories:
  - blog
tags:
  - diffusion
  - video generation
  - image-to-video
  - DiT
  - multimodal
toc: true
---

最近视频生成模型卷得很快，`Wan2.1` 是阿里 Wan 团队开源的那一套。它最常用的场景之一就是 **I2V（Image-to-Video）**：给一张参考图加一句文字 prompt，模型给你生成一段几秒的视频，首帧基本还是那张图，后续的镜头就按你写的文字去演。

这篇文章想做的事情是：

> 把 Wan2.1 I2V 里**每一步数据发生了什么**讲清楚，让从没接触过视频生成的人也能看懂。

我们会从最外层的"图像 + 文字 → 视频"讲起，一路剥开壳子：  
VAE 到底在压缩什么、CLIP 和 T5 各自管什么、DiT 内部是怎么把图像信息和文字信息混进去的、采样循环为什么要跑那么多步、以及为什么首帧会这么"像"你给的那张图。

<br />
<img align="center" width="1000" src="{{ site.url }}/images/posts/wan21-i2v-overview.svg" alt="Wan2.1 I2V overall architecture">
<br />

这张图是全文的总地图。下面的每一节都是在放大它的某一块。

---

## 1. 先做一次"外行翻译"：I2V 到底在做什么

如果用一句日常语言来描述 I2V，其实是：

> 我们有一张图（`3 × H × W`，RGB 像素），想把它"续写"成一段视频（`3 × F × H × W`，F 帧），而且这段视频的内容要符合文字 prompt。

朴素想法是直接训练一个"图 + 文字 → 视频"的网络。问题有二：

1. 视频的体积太大。即便是 480p × 24fps × 4 秒，也已经是 1.1 亿像素级别，直接建模太贵。
2. 我们希望生成过程是**可控的**——能调 guidance，能控制风格，能多步修正——而不是一次性跑完一个巨大网络就结束。

Diffusion 模型的套路恰好能解决这两件事：

- **压缩**：用 VAE 把视频压到一个小很多的 latent 空间，之后所有运算都在 latent 上做。
- **迭代**：扩散模型天然是多步的，每一步都在"把更接近噪声的视频"往"更清晰的视频"方向推一点。

所以 Wan2.1 I2V 的骨架分成两大块：

| 模块 | 角色 |
|---|---|
| **Wan-VAE** | 像素 ⇄ latent 的翻译员 |
| **DiT** | 在 latent 空间里"去噪"的大脑 |

外加两个**条件编码器**：

| 模块 | 角色 |
|---|---|
| **CLIP ViT-H/14** | 把参考图变成"这张图看起来讲了什么"的高层语义向量 |
| **umT5** | 把文字 prompt 编码成一串 token embedding |

接下来我们分别看每一块。

---

## 2. Wan-VAE：把视频压缩 256 倍后再还原

`Wan-VAE` 是一个 **3D Causal VAE**。它做的事很朴素：

- 输入：`[3, F, H, W]` 的视频（或单张图当作 `F=1`）
- 输出：`[16, F/4, H/8, W/8]` 的 latent

换句话说：

- 空间下采样 `8×8` 倍
- 时间下采样 `4` 倍
- 通道数从 `3` 变成 `16`（表达能力变强）

总体积约压缩 **256 倍**（`8·8·4 / (16/3) ≈ 24×24 / ...`，算下来大约 48× 的"信息体积"，但浮点数要少 200+ 倍）。

> **为什么叫 Causal？** 指的是它的时间卷积只看"过去"不看"未来"，这样可以支持变长视频、流式推理，和后续滚动生成新帧。

一个关键点是 **I2V 里 VAE 会被用两次**：

1. **编码参考图**：把那张图当成一个 `F=1` 的视频编码，得到它的 latent。
2. **解码最终视频**：DiT 输出 latent，扔给 VAE 解码回像素视频。

其中第一次编码的结果被塞进 DiT 作为"低层像素/结构"条件——这是后面讲 I2V 双路条件时的关键一环。

---

## 3. 两路文字/图像条件编码：CLIP 和 T5 各自做什么

这两个模型很多人容易搞混，但它们在 Wan2.1 里分工很清晰。

### 3.1 umT5：把文字变成 512 × 4096 的 token 序列

`umT5` 是 T5 的多语言版。输入是你的 prompt，输出是：

```
[seq_len, 4096]   # 每个 token 一个 4096 维向量
```

Wan2.1 统一把这个序列 padding / truncate 到 **512 个 token**，所以文本总是 `[512, 4096]`。

> T5 是一个纯文本的大模型，它的向量很"语言化"，擅长表达语义、句法关系。

### 3.2 CLIP ViT-H/14：把图像变成 257 × 1280 的 token 序列

`CLIP` 是一个**跨模态**模型（图像 + 文本对齐训练的），这里我们只用它的**图像编码器**（ViT-H/14）。

它吃一张 `224 × 224` 的图，输出：

```
[257, 1280]
```

**257 从哪来？** 这是一个很常见的数字：

- `ViT-H/14` 把 224×224 切成 `14×14` 的 patch
- `224 / 14 = 16`，所以一张图变成 `16 × 16 = 256` 个 patch token
- 再加一个 `CLS` token，总共 **257** 个

每个 token 的通道数是 `1280`（ViT-H 的隐藏维度）。

> CLIP 给出的是**图像的高层语义**：它知道这张图里是"一只猫"、"傍晚的海边"、"油画风格"之类的语义抽象，但几乎不保留像素级的精细结构。

### 3.3 CLIP vs T5：为什么两个都要？

这是 I2V 非常关键的一点。两者的"关注点"不一样：

| | 擅长 | 不擅长 |
|---|---|---|
| **T5** | 文字描述的动作、意图、场景 | 图像具体长什么样 |
| **CLIP** | 参考图的整体风格、主体 | 精确的像素/空间结构 |

所以两者是**互补**的——都给 DiT 看一遍，DiT 再自己挑。这也是为什么后面会看到 cross-attention 是"双流"的。

---

## 4. 把"参考图的像素"也塞进模型：条件 latent `y`

到这里我们已经有了两条图像通路：CLIP（语义）和 T5（文字）。但对 I2V 来说，仅靠 CLIP 的语义是不够的——生成的第一帧如果不能"长得非常像"输入图，用户立刻会觉得不对。

于是 Wan2.1 加了第三条通路：**把参考图用 VAE 编码后，直接在通道维度拼到噪声 latent 上**。

### 4.1 构造 `y`

假设目标视频是 `F` 帧，latent 形状 `[16, F/4, H/8, W/8]`。我们把 `T_latent = F/4`。

**第 1 步：把参考图放到第 0 帧，其余帧置零。**

```python
video_clip = concat([
    img_resized,                # [3, 1, H, W]  ← 第 0 帧 = 参考图
    zeros(3, F-1, H, W)         # 其余帧为 0
], dim=1)                       # → [3, F, H, W]
```

**第 2 步：VAE 编码。**

```python
y_latent = VAE.encode(video_clip)   # → [16, T_latent, H/8, W/8]
```

**第 3 步：构造时间 mask，标记"哪些帧是已知的"。**

```python
msk = ones(1, F, H_lat, W_lat)
msk[:, 1:] = 0                         # 只有第 0 帧 = 1
# 把 msk[:, 0:1] 沿时间 repeat 4 次，和 msk[:, 1:] 拼接
# 再 reshape 成 [4, T_latent, H_lat, W_lat]
```

这里的 `4` 是 VAE 的时间 stride——我们需要让 mask 通道数足够"表达"被 VAE 压缩掉的时间细节。

**第 4 步：mask 和 VAE latent 通道拼接，得到 `y`。**

```python
y = concat([msk, y_latent], dim=0)    # [4 + 16 = 20, T_latent, H_lat, W_lat]
```

> 把 `y` 想成一块"透明纸"：第 0 帧那一层写满了"你要照着这张图画"，其它帧那一层是空白，同时还有一层专门标注"哪里非空白"。

### 4.2 `y` 怎么进 DiT

DiT 的输入是噪声 latent `x_t: [16, T_latent, H_lat, W_lat]`。进网络前做一次通道拼接：

```
x = concat(x_t, y) = [16 + 20, T, H, W] = [36, T, H, W]
```

所以 I2V 的 DiT 输入通道是 **36**（T2V 是 16）。也正因为这个差别，I2V checkpoint 的 `patch_embedding` 卷积权重和 T2V 不是一回事。

---

## 5. DiT 内部：一个时间步里到底跑了什么

接下来进入最核心的部分。我们放一下 DiT 单层的结构图：

<br />
<img align="center" width="1100" src="{{ site.url }}/images/posts/wan21-i2v-block.svg" alt="Wan2.1 DiT block internal">
<br />

整体看，DiT 是一个典型的 Transformer 栈，但有三个重要定制：

1. **时空 3D RoPE**（self-attention 里的位置编码）
2. **双流 cross-attention**（image KV + text KV）
3. **AdaLN-Zero 风格的 timestep 调制**

下面一条一条讲。

### 5.1 Patchify：把视频 latent 变成 Transformer 的 token 序列

```python
self.patch_embedding = nn.Conv3d(36, dim, kernel_size=(1,2,2), stride=(1,2,2))
```

这是一个"**3D patchify**"：用 `Conv3d` 把每个 `1×2×2` 的时空小块打成一个 token。

- 时间方向 kernel=1，意味着**时间维度不被合并**（每一个 latent 帧仍然是独立的一层 token）。
- 空间方向 kernel=2，把 `H_lat × W_lat` 的网格再进一步压 `2×2`，得到 `H_lat/2 × W_lat/2` 个 token。

最终序列长度是：

```
S = T_latent × (H_lat/2) × (W_lat/2)
```

每个 token 是 `dim` 维向量（1.3B 版本里 `dim=2048`）。

### 5.2 Timestep embedding：让每层都知道"现在在第几步"

扩散模型的一个关键差别是每一步的处理方式不一样。T=T_max 时几乎全是噪声，T=0 时已经是完整视频，所以模型在不同 step 应该"轻重不一"。

Wan2.1 的做法是 **AdaLN-Zero**（DiT 论文里的那一套）：

```python
e = sinusoidal_embedding_1d(256, t)          # 标量 t → 256 维向量
e = time_embedding(e)                        # MLP 投到 dim
e0 = time_projection(e).unflatten(-1, (6,dim))  # 再投成 [B, 6, dim]
```

然后把这 6 份向量分发给每个 block，块内再加上自己可学习的 `modulation` 参数，切成 6 组：

```
(shift1, scale1, gate1,  shift2, scale2, gate2)
```

- `shift, scale` 用在 LayerNorm 之后：`x' = norm(x) · (1 + scale) + shift`
- `gate` 用在残差分支：`x = x + gate · f(x')`

> **"Zero" 的含义**：`gate` 初始化为 0，使得模型训练开始时每个 block 都是恒等映射——DiT 从一个干净的起点开始学。

注意：cross-attention **不被 AdaLN 调制**，只有 self-attention 和 FFN 被调制。

### 5.3 Self-Attention：3D 全局注意力 + 分解式 RoPE

这一步做的事很简单：**视频 token 之间互相看**。

代码上是标准的 QKV flash attention，但有两处定制：

**① QK 做 RMSNorm**。这是稳定训练用的技巧：

```python
q = RMSNorm(Linear_q(x))
k = RMSNorm(Linear_k(x))
v = Linear_v(x)
```

**② 3D 分解式 RoPE 作用在 Q/K 上**（不作用于 V）。

视频 token 有三个坐标：`(frame, height, width)`。Wan2.1 把每个 head 的维度 `d` 切成三段：

| 段 | 通道数 | 编码的是 |
|---|---|---|
| 时间 | `d − 4·(d/6)` | 帧索引 `f` |
| 高 | `2·(d/6)` | 行索引 `h` |
| 宽 | `2·(d/6)` | 列索引 `w` |

三段分别应用一维 RoPE（复数旋转），然后沿通道拼回一起：

```
q_rot = RoPE_T(q[:, :, :dT], f) ⊕ RoPE_H(q[:, :, dT:dT+dH], h) ⊕ RoPE_W(q[:, :, dT+dH:], w)
```

**为什么这样设计？**

- 可以支持**任意分辨率和帧数**，因为 RoPE 是外推良好的位置编码。
- 时间/空间的频率独立，模型可以各自学合适的"时间尺度"和"空间尺度"。
- 相比绝对位置嵌入，训练时可以在一个尺度下训，推理时换尺度不会崩。

**注意力的范围是"全 3D 全局"**——所有视频 token 互相能看。这就是为什么视频生成模型这么贵：序列长度是 `T × H × W`，attention 是 O(S²)。

### 5.4 Cross-Attention：双流融合图像 + 文字

到了 I2V 最有意思的设计。先回忆一下 context 长什么样：

```
context = [ CLIP_257 ∥ T5_512 ]         # shape [769, dim]
         ─── 前 257 ───   ── 后 512 ──
             (图像)           (文本)
```

如果用朴素 cross-attention，你会一次算 `attn(q, K=k_all, V=v_all)`，让视频 token 对这 769 个 token 做 softmax。问题是图像和文本的分布差距很大，softmax 会把注意力偏到一侧。

Wan2.1 的做法是**双流独立**：

```python
# 共享 Query
q = Linear_q(x)

# 图像分支（独立的 k_img, v_img）
k_img = Linear_k_img(context[:, :257])
v_img = Linear_v_img(context[:, :257])

# 文本分支（共享 T2V 的 k, v）
k_txt = Linear_k(context[:, 257:])
v_txt = Linear_v(context[:, 257:])

out_img = flash_attn(q, k_img, v_img)
out_txt = flash_attn(q, k_txt, v_txt)

out = Linear_o(out_img + out_txt)   # 逐元素相加再过输出投影
```

几个关键设计点：

1. **独立的 K/V 投影**：图像用 `k_img, v_img`，文本用 `k, v`。每一模态在自己的几何空间里算 attention，不会互相挤压 softmax。
2. **两次独立 attention 再相加**：相当于两种信号**分别**给每个视频 token 打了一次分，再叠加作为新的残差。
3. **Q 共享**：视频 token 只有一份"问题"，问图和文字同一个问题："你们谁和我相关？"
4. **无 RoPE**：cross-attn 中的 K/V 是外部序列，不需要视频的时空位置编码。

> 直观理解：**image 分支管"我希望长什么样"，text 分支管"我希望怎么演"，两个加在一起就是视频 token 的条件梯度**。

### 5.5 FFN：标准 MLP，再来一次 AdaLN 门控

```python
y = ffn(norm2(x) · (1+scale2) + shift2)
x = x + gate2 · y
```

`ffn` 就是常规的 `Linear → GELU → Linear`，中间维度是 `4 × dim`（比如 dim=2048 时 ffn_dim=8192）。

到这里一个 block 就结束了。把这个 block 叠 32 层（1.3B 版）或 40 层（14B 版），最后过一个 `Head`（也带 AdaLN 和 unpatchify），就能把 `[B, S, dim]` 变回 `[16, T, H, W]`——也就是模型对**当前时间步的"速度场" `v`** 的预测。

---

## 6. 训练目标：为什么叫 Flow Matching，不再叫"预测噪声"

DDPM 早年是让模型预测"这张图里的噪声 `ε`"。Wan2.1 用的是 **Flow Matching / Rectified Flow** 的范式——本质上是把扩散过程理解成**一条从噪声到数据的直线路径**，模型学的是这条路径上每一点的"速度"。

具体来说，定义一条插值：

$$
x_t = (1 - t) \cdot x_0 + t \cdot \epsilon, \quad t \in [0, 1], \quad \epsilon \sim \mathcal{N}(0, I)
$$

那么真值速度就是：

$$
v^* = \frac{d x_t}{d t} = \epsilon - x_0
$$

训练目标：

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_0, \epsilon, t} \left\| v_\theta(x_t, t, c) - (\epsilon - x_0) \right\|^2
$$

其中 `c = {y, CLIP_fea, T5_text}` 是所有条件的合集。

**Flow Matching 相比预测 ε 有什么好处？**

- 训练 loss 更稳定，对 `t` 的依赖更平滑。
- 采样时可以用更少的步数。典型配置 **25–50 步**即可出不错结果（早期 DDPM 需要 1000 步）。
- 路径"直"这件事意味着模型不容易陷入局部的噪声拟合。

---

## 7. 一次完整的推理：25 步里到底发生了什么

现在把所有东西串起来。假设你给了一张 `H × W` 的图、一句 prompt，让模型生成 F 帧的视频：

```
─── 推理前准备（只做 1 次）────────────────────────────────
1. t5_ctx  = umT5(prompt)                      # [512, 4096] → MLP → [512, dim]
2. clip_fea = CLIP.visual(image)               # [257, 1280] → MLPProj → [257, dim]
3. img_lat = VAE.encode([image, zeros, ...])   # [16, T, H_lat, W_lat]
4. msk     = build_mask(first_frame=1)         # [4, T, H_lat, W_lat]
5. y       = concat(msk, img_lat)              # [20, T, H_lat, W_lat]
6. context = concat(clip_fea, t5_ctx)          # [769, dim]
7. x_T     ~ N(0, I)                           # [16, T, H_lat, W_lat]

─── 采样循环（跑 25~50 次）─────────────────────────────
for t in schedule:  # e.g. [1.0, 0.96, ..., 0.0]
    # (可选) CFG: 各跑一次有/无条件
    x_in = concat(x_t, y)          # [36, T, H_lat, W_lat]
    v_cond = DiT(x_in, t, context, clip_fea, y)
    # v_uncond = DiT(x_in, t, empty_context, ...)
    # v = v_uncond + s · (v_cond - v_uncond)
    v = v_cond
    
    x_{t-Δt} = x_t - v · Δt        # flow matching 欧拉步

─── 解码 ──────────────────────────────────────────
video_latent = x_0                  # [16, T, H_lat, W_lat]
video        = VAE.decode(video_latent)  # [3, F, H, W]
```

几个细节：

- **`y` 只构造一次**，在整个 25 步里都用同一份。因为参考图是不变的。
- **CFG（Classifier-Free Guidance）**：Wan2.1 训练时会随机丢弃条件，所以推理时可以通过 `v = v_u + s·(v_c - v_u)` 放大条件信号（典型 `s=5~7.5`）。每步需要跑两遍 DiT。
- **首帧为什么保真？**：因为第 0 帧的 `mask=1` 和 `VAE(img)` 一直被塞进输入，DiT 每步都在"被提醒"首帧应该长什么样。随着 t 变小，模型越来越相信这个约束。

---

## 8. 几个关键数字一张表带走

| 参数 | 值 | 解释 |
|---|---|---|
| VAE 空间 stride | 8 | H/W 方向下采样倍率 |
| VAE 时间 stride | 4 | F 方向下采样倍率 |
| VAE latent 通道 | 16 | 压缩后的通道数 |
| I2V `y` 通道 | **20** | 4 (mask) + 16 (VAE latent) |
| DiT 输入通道 | **36** | 16 (noise) + 20 (y) |
| Patch size | (1, 2, 2) | 时间不并、空间 2×2 |
| 文本 token 数 | 512 | umT5 输出 padded |
| CLIP token 数 | **257** | 1 CLS + 16×16 patches |
| CLIP 维度 | 1280 | ViT-H 的 hidden |
| DiT hidden | 2048 (1.3B) / 更大 (14B) | |
| DiT 层数 | 32 / 40 | |
| 注意力头 | 16 | head_dim=128 |
| Sampling 步数 | 25–50 | Flow Matching 下 |

---

## 9. T2V vs I2V：到底改了哪里

最后来一张对比表，帮你一眼看清两种模型的差别：

| 方面 | T2V | I2V |
|---|---|---|
| 输入条件 | 只有文本 | 文本 + 参考图 |
| `patch_embedding` in_channels | 16 | **36** |
| Cross-Attention 类型 | 单流（只有文本 K/V） | **双流**（image K_img/V_img + text K/V） |
| `img_emb` (CLIP → dim MLP) | ❌ 无 | ✅ 有 |
| `y`（mask + image latent） | ❌ 无 | ✅ 有，通道拼接到 `x` |
| `clip_fea` | ❌ 无 | ✅ 前置到 context |
| 采样过程 | 一样（flow matching） | 一样 |

所以 T2V → I2V 的改造量其实并不大：**多了两条图像通路（CLIP 语义 + VAE 像素），外加一组额外的 cross-attention K/V 权重**，其它骨架完全一致。这也是为什么很多团队能从 T2V checkpoint 微调出 I2V 版本。

---

## 10. 常见疑问答疑

**Q1：为什么不只用 CLIP、不要 VAE latent？**  
只用 CLIP 的话，模型知道"这是一只猫"，但不知道"这只猫在图里具体长什么样、坐在什么位置、毛色分布怎样"。CLIP 太高层。VAE latent 保留了像素级结构，所以首帧能做到"几乎像素级一致"。

**Q2：为什么不只用 VAE latent、不要 CLIP？**  
VAE latent 是"为了重建像素而设计的压缩特征"，它缺乏跨模态语义。CLIP 的语义向量能让模型在后续帧里理解"这张图在讲什么"，从而和 prompt 对齐得更好。两者是语义和像素的两极，缺一不可。

**Q3：mask 通道为什么要 4 维，不能是 1 维？**  
因为 VAE 的时间 stride = 4，一个 latent 帧对应 4 个像素帧。4 通道的 mask 让每个 latent 帧能独立标记"这 4 帧里各自是不是已知"。这样在滚动生成或多帧条件 I2V 里能无缝扩展。

**Q4：为什么 cross-attn 不做 RoPE？**  
RoPE 是为 query/key 在同一个坐标系下的相对距离准备的。cross-attn 的 key 来自外部序列（文本/图像 token），没有和视频 token 共享的"时空坐标"，用 RoPE 反而有害。

**Q5：CFG 在 I2V 里到底丢的是什么？**  
Wan2.1 做 CFG 时通常**只丢文本**（把 `t5_ctx` 置空），保留 CLIP 和 VAE latent。因为 I2V 的核心约束是参考图，不能丢；被用来"放大信号"的是文本 prompt。有些实现也会同时丢 CLIP，做"image guidance"。

**Q6：能不能做多图 / 多首尾帧条件？**  
可以。`y` 的结构天然支持——只需要把对应帧位置的 mask 设为 1、在 VAE 输入里把那些帧填真实图像即可。这就是社区里各种"首尾帧控制"、"关键帧插值"玩法的实现基础。

---

## 11. 总结

回到开头那张大图，现在你应该能一眼看懂每一块发生了什么：

- **VAE** 负责压缩像素和还原像素；
- **T5** 负责理解文字；
- **CLIP** 负责理解图像的"长相和风格"；
- **DiT** 在一个压缩的 latent 空间里，一步一步把噪声拉回视频，拉的方向由前三个模块的条件决定；
- **I2V** 的所有"魔法"就是把参考图的信息**同时**从两条通路（像素 / 语义）塞给 DiT，再用 cross-attention 双流、AdaLN 门控把它们融合进每个视频 token。

一旦把这张图想清楚，你去读 Wan2.1 源码、甚至去扩展它（做首尾帧、多图参考、风格迁移），都会容易很多。

---

## Sources

- [Wan2.1 官方仓库](https://github.com/Wan-Video/Wan2.1)
- [Wan 技术报告 arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- [Wan2.1 model.py 源码（HF 镜像）](https://huggingface.co/spaces/2chch/Wan2.1/blob/2a07a689c8837aac720a73915b783ea23b371927/wan/modules/model.py)
- [Wan2.1 image2video.py](https://github.com/Wan-Video/Wan2.1/blob/main/wan/image2video.py)
- [DiT: Scalable Diffusion Models with Transformers (Peebles & Xie, 2022)](https://arxiv.org/abs/2212.09748)
- [Flow Matching for Generative Modeling (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747)
- [Rectified Flow (Liu et al., 2022)](https://arxiv.org/abs/2209.03003)
- [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [RoFormer: RoPE (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
