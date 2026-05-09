---
title: 'Diffusion 模型的条件注入演进史：从通道拼接到单流 DiT'
date: 2026-05-09
permalink: /posts/2026/05/diffusion-condition-injection/
categories:
  - blog
tags:
  - diffusion
  - generative model
  - DiT
  - ControlNet
  - IP-Adapter
  - multimodal
toc: true
---

如果你看过 Stable Diffusion、ControlNet、IP-Adapter，又听说过最近的 Qwen-Image 和 Z-Image，可能会有一个共同的疑问：

> 这些模型架构看上去差别很大，但它们要解决的问题其实都是同一个：**怎么把"用户想要什么"这件事告诉模型？**

文本 prompt、参考图、姿态骨架、深度图、音频、mask……每一种"控制信号"进入网络的方式都不一样。这篇文章想做的事情是：

> 把过去几年 Diffusion 里**主流的条件注入方式**串成一条线，讲清楚每一步的动机、做法、优劣，以及它们最后是怎么汇聚到今天的 DiT 体系里的。

读完之后，你应该能：

- 理解为什么 inpainting 用 channel concat，而 text-to-image 用 cross-attention；
- 知道 ControlNet 和 IP-Adapter 各自解决的是什么 prompt 解决不了的问题；
- 看懂 adaLN-Zero 为什么是 DiT 的"默认调制器"；
- 理解 Qwen-Image 的多流 MMDiT 和 Z-Image 的单流 S3-DiT 在条件控制上到底差在哪里；
- 对未来 Diffusion 架构的演进方向有一个直觉性的判断。

---

## 1. 起点：Diffusion 模型为什么需要"条件"

在最朴素的 DDPM 里，模型学的是数据分布 `p(x)`：

> 什么样的样本看起来像"真实世界中的自然图像"。

它知道怎么把高斯噪声慢慢拉回自然图像流形，但它**不知道你想要什么**。给它噪声，它能生成一张图，但你没法控制是猫是狗、是写实还是油画、是白天还是夜晚。

所以条件 Diffusion 真正学的是：

$$
p(x \mid c)
$$

也就是"给定条件 `c` 的情况下，样本 `x` 长什么样"。这里的 `c` 可以是文本、参考图、姿态骨架、音频、深度图……几乎任何能数字化表示的"用户意图"。

问题来了：**`c` 怎么进入模型？**

这看似是个工程细节，但它直接决定了模型能不能用这个条件、能用得多准、计算成本多大。过去几年，Diffusion 社区在这个问题上其实经历了好几代演化。我们一个一个看。

---

## 2. 第一代：通道拼接 (Channel Concat)

### 2.1 动机：inpainting 是最早的"条件生成"

最早的"有条件"扩散模型其实就是 **inpainting** —— 给一张图、一个 mask，让模型把 mask 区域补全。这种任务的特点是：

- 条件本身是**和图像同一空间结构**的（mask 是 H×W 的图，masked image 也是 H×W 的图）；
- 用户的意图就是"按照原图的结构和上下文，把这块补好"。

那最简单的做法是什么？

**直接把 mask 和 masked image 当成额外通道，拼到 noisy latent 上一起送进 U-Net。**

Stable Diffusion Inpainting 就是这么干的。原本的 U-Net 输入是 `[B, 4, H, W]`（4 个 latent channel），inpainting 版本变成 `[B, 9, H, W]` —— 多出来的 5 个通道分别是 1 个 mask 和 4 个 masked latent。

```
unet_input = concat[
    noisy_latent,        # [B, 4, H, W]   被去噪的对象
    mask,                # [B, 1, H, W]   告诉模型哪里要改
    masked_latent,       # [B, 4, H, W]   告诉模型其它地方长什么样
]
```

### 2.2 LatentSync：把这个思路用到 lip-sync

最近字节开源的 [LatentSync](https://github.com/bytedance/LatentSync) 把这个思路扩展到了视频口型同步。它的 U-Net 输入是 13 通道：

```
unet_input = concat[
    noisy_gt_latents,    # 4 channels  当前 diffusion step 下的 noisy target
    masks,               # 1 channel   嘴部 mask
    masked_latents,      # 4 channels  嘴部被遮住的当前帧 latent
    ref_latents,         # 4 channels  同一视频里另一段参考帧 latent
]                        # total: 13
```

可以看出来，channel concat 适合的是**和图像在空间上对齐的低层视觉条件**：mask、被遮挡的图像、参考帧。它的优点是简单直接，VAE encoder 的输出可以直接拼上去；缺点是它没法处理**变长**或**异构模态**的条件，比如一段文字、一段音频。

### 2.3 局限

如果用户的 prompt 是 "a red sports car running on the highway"，你怎么把它"拼"到一张图上？文字根本不是 H×W 的张量。

这就引出了下一代的方案。

---

## 3. 第二代：交叉注意力 (Cross-Attention)

### 3.1 动机：Stable Diffusion 让文本成为 prompt

2022 年 Stable Diffusion 一炮走红的关键，不是它的 VAE，也不是它的 U-Net 结构，而是它把**文本→图像**这件事做成了 cross-attention：

- 文本经过 CLIP text encoder 变成 `[N_text, D]` 的 token 序列；
- U-Net 中的图像 feature 作为 query，文本 token 作为 key/value；
- 在每一层 attention 里，图像 feature 会去"问"文本 token：你想让我画什么？

数学上：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

- $Q = W_q \cdot \text{image\_feature}$
- $K = W_k \cdot \text{text\_token}$
- $V = W_v \cdot \text{text\_token}$

这种设计天然适合**变长、异构**的条件：文本可以是 5 个词也可以是 50 个词，attention 自适应处理。

### 3.2 不只是文本：音频、video embedding 都能这么干

LatentSync 用 Whisper 提取音频 embedding，然后通过 cross-attention 注入 U-Net；Wan2.1 用 T5 编码文本、CLIP 编码图像，两路条件都通过 cross-attention 进入 DiT。这套机制已经成了"高层语义条件"的事实标准。

### 3.3 局限

cross-attention 不便宜。原始 DiT 论文实验过几种条件注入方案，发现 cross-attention 比 adaLN 多大约 **15% 的 FLOPs**[^1]。当模型规模上到几十亿参数时，这个开销不容忽视。

更深层的问题是：cross-attention 的条件信息**只在 attention 层生效**，对 LayerNorm、MLP 这些非 attention 层是"透明"的。如果条件本身是一个全局信号（比如"现在是 timestep=500"、"这是猫这一类"），用 cross-attention 杀鸡用牛刀。

---

## 4. 第三代：自适应归一化 (FiLM / adaLN)

### 4.1 动机：有些条件其实是"全局调制信号"

看一下 cross-attention 在做什么：它让每个图像 token 去关注一些条件 token。但如果条件本身就是一个**全局向量**（比如时间步 `t`、类别 `c`、说话人 id），让每个图像 token 都去 attend 它，纯粹是浪费算力。

更高效的做法是 **FiLM (Feature-wise Linear Modulation)**：

$$
\text{FiLM}(x, c) = \gamma(c) \odot x + \beta(c)
$$

也就是用条件 `c` 算出一组缩放和偏移参数，直接对 feature 做线性调制。

### 4.2 adaLN：把 FiLM 用到 LayerNorm 上

普通的 LayerNorm 是这样：

$$
\text{LN}(x) = \gamma \cdot \text{normalize}(x) + \beta
$$

这里的 `γ, β` 是学习出来的固定参数。**adaLN (Adaptive LayerNorm)** 把它换成条件的函数：

$$
\text{adaLN}(x, c) = \gamma(c) \cdot \text{normalize}(x) + \beta(c)
$$

其中 `γ(c), β(c) = MLP(c)`。

DiT 的论文系统比较了几种条件注入方案——in-context conditioning、cross-attention、adaLN——最终发现 **adaLN 是 FLOPs 最低、效果最好的方案**[^2]。

### 4.3 adaLN-Zero：让深层 Transformer 训练得更稳

DiT 在 adaLN 上又加了一个改动，叫 **adaLN-Zero**：除了 scale 和 shift，还预测一个 **gate**，并把它的初始值设为 0：

```python
shift, scale, gate = MLP(c).chunk(3, dim=-1)

y = LN(x)
y = y * (1 + scale) + shift
x = x + gate * Attention(y)   # gate 初始为 0，所以这一项一开始是 0
```

这意味着：**训练初始时，每个 block 都近似 identity function**，新加进来的网络层不会立即破坏预训练的表征。这个技巧让深层 DiT 训练得非常稳，几乎成了今天所有 Transformer-based diffusion 的标配。

### 4.4 局限

adaLN 的本质是**全局调制**：它对所有 token 施加同一组 `(γ, β, gate)`。这意味着它擅长做的事情是：

- 时间步注入（`t` 是一个标量）
- 类别注入（class id 是一个 one-hot）
- 全局风格控制（style embedding）

它不擅长的事情是：

- 告诉模型"左手在这里"
- 告诉模型"这条边缘要保留"

要做空间结构控制，还需要更专门的机制。

---

## 5. 第四代：ControlNet —— 给 U-Net 外挂一支控制分支

### 5.1 动机：文本说不清楚"姿态"

文字 prompt 有一个根本局限：**它没法精确描述空间结构**。你说 "a person standing with left hand raised"，模型可能给你一个右手抬起来的人，或者两只手都抬着的人。

但如果你能给模型一张姿态骨架图、一张边缘图、一张深度图，事情就完全不一样了 —— 这些条件本身就是**空间对齐**的，每个像素都精确地告诉模型"这个位置应该是什么"。

### 5.2 做法：复制一份 encoder + 零卷积

ControlNet 的设计很巧妙[^3]：

1. **冻结**原始 Stable Diffusion U-Net，保留它所有的预训练能力；
2. **复制**一份 U-Net 的 encoder（包括 down blocks 和 mid block）作为 trainable branch；
3. condition 图（pose / edge / depth）作为这条分支的输入；
4. 这条分支在每个尺度产生 residual feature，**通过零卷积加到原 U-Net 的对应层**。

```
condition_map (pose/edge/depth)
    ↓
[ControlNet branch (trainable copy of encoder)]
    ↓
multi-scale residuals
    ↓
加到 frozen U-Net 的 down/mid blocks
```

### 5.3 零卷积：让训练从"零干扰"开始

ControlNet 最关键的技巧是 **zero convolution**：连接 ControlNet 分支和原 U-Net 的卷积层，**初始权重和 bias 都是 0**。

这意味着：

- 训练第 0 步：ControlNet 输出全是 0，原 U-Net 行为完全不变；
- 随着训练进行，零卷积逐渐学到非零参数，ControlNet 的影响逐渐增强；
- **不会有训练初期"噪声破坏预训练模型"的问题**[^3]。

这个设计让 ControlNet 在很小的数据量上就能训练成功 —— 不像 fine-tuning 那样需要担心 catastrophic forgetting。

### 5.4 局限

- **每种 condition 要训一个 ControlNet**：pose、edge、depth、normal、segmentation 各自一个分支；
- **额外参数量不小**：复制了一半 U-Net，参数量大约是原模型的 50%；
- **只控制结构，不控制语义/身份**：ControlNet 告诉模型"这个位置有边缘"，但没法告诉它"这个人长得像谁"。

最后这一点引出了下一个机制。

---

## 6. 第五代：IP-Adapter —— 让图片成为 prompt

### 6.1 动机：有些东西文字真的描述不了

试着用文字描述一个具体的人长什么样：

> "a woman with long brown hair, brown eyes, oval face, slightly upturned nose..."

写得再细，模型也只能给你一个**满足这些属性的随机面孔**，不是你脑子里那个**具体的人**。

类似地：

- 一个 logo 的精确视觉风格；
- 一件衣服的具体花纹；
- 一种独特的画风。

这些"视觉概念"用文字 prompt 几乎不可能精确传达。但如果你能直接给模型一张**参考图**，事情就简单多了。

### 6.2 朴素做法：把图像和文字 token 拼起来

最直观的想法是：用 CLIP image encoder 编码参考图，然后把图像 token 和文本 token 拼到一起，喂给同一个 cross-attention。

但 IP-Adapter 论文指出，**这种朴素拼接会导致图像信息被文本特征"覆盖"**[^4]。原因是：原模型的 cross-attention `K, V` projection 是针对文本特征训练的，强行让图像特征通过同一组 projection 进入 attention，会损失大量图像信息。

### 6.3 解耦交叉注意力 (Decoupled Cross-Attention)

IP-Adapter 的核心创新是 **decoupled cross-attention**：**给图像单独开一套 cross-attention**，而不是和文本共享。

```python
def ip_adapter_attention(x, text_tokens, image_tokens):
    q = Wq(x)

    # 原本的 text cross-attention（保持不变）
    k_text = Wk_text(text_tokens)
    v_text = Wv_text(text_tokens)
    out_text = attention(q, k_text, v_text)

    # 新增的 image cross-attention（新训练的）
    k_img = Wk_img(image_tokens)
    v_img = Wv_img(image_tokens)
    out_img = attention(q, k_img, v_img)

    return out_text + scale * out_img
```

注意几个关键设计：

1. **复用 query**：图像 attention 和文本 attention 共享同一个 `Q`，因为 query 来自图像 latent；
2. **独立的 K/V projection**：`Wk_img, Wv_img` 是新训练的，专门为图像特征设计；
3. **加和融合**：两路 attention 的输出直接相加，可以用 `scale` 控制图像 prompt 的强度。

### 6.4 优势：极小的可训练参数

IP-Adapter 论文报告，**只需要约 22M 可训练参数**，就能在冻结的 Stable Diffusion 上实现强大的图像 prompt 能力，并且和文本 prompt、ControlNet 等工具完全兼容[^4]。

### 6.5 IP-Adapter vs ref_latent (LatentSync)

值得对比一下，因为它们看起来都是"用参考图做条件"：

| 方式 | 信息类型 | 注入方式 | 优点 |
|---|---|---|---|
| **LatentSync 的 ref_latents** | 低层像素/纹理/位置 | VAE latent → channel concat | 空间对齐强，重建保真度高 |
| **IP-Adapter 的 image tokens** | 高层语义/身份/风格 | CLIP image encoder → cross-attention | 泛化强，可以"风格迁移" |

简单来说：
- 想保留**精确的像素结构**（同一个人、同一个场景的不同角度）→ ref latent concat；
- 想保留**风格和身份语义**（这个人的"长相"，但姿势可以不一样）→ IP-Adapter。

---

## 7. 一张表把前五代串起来

到这里我们已经看了五种主流的条件注入方式。它们其实是**互补**而不是替代关系。一个现代的 Diffusion 系统通常会同时用到好几种：

| 机制 | 数学形式 | 条件类型 | 空间对齐 | 参数量 | 典型用途 |
|---|---|---|---|---|---|
| **Channel Concat** | `concat([z_t, c], dim=1)` | 空间对齐的低层视觉信息 | 强 | 几乎为 0 | inpainting, ref frame, mask |
| **Cross-Attention** | `Attn(Q=z_t, K=c, V=c)` | 变长高层语义 | 弱 | 中 | text, audio, video tokens |
| **adaLN-Zero** | `x = x + gate(c) · f(scale(c)·LN(x)+shift(c))` | 全局信号 | 无 | 极小 | timestep, class, style |
| **ControlNet** | `h_i ← h_i + ControlNet_i(cond)` | 空间结构图 | 强 | 大（~50% U-Net） | pose, edge, depth, seg |
| **IP-Adapter** | `Attn_text + λ · Attn_image` | 参考图语义/身份 | 中 | 极小（~22M） | reference image, style |

注意一个有意思的事实：**这五种机制里，真正被 diffusion 去噪的只有 `z_t` 一个**。其他所有的"条件" —— 不管是 mask、文本 token、ControlNet residual、image token —— 都只是**改变 U-Net 对 `z_t` 噪声预测方向的指引**，它们自己不会被更新。

理解这一点，对接下来看 DiT 的演进非常重要。

---

## 8. 第六代：从 U-Net 到 DiT，条件注入也跟着进化

### 8.1 为什么大家开始用 Transformer 替代 U-Net

U-Net 的核心是卷积 + skip connection，它的归纳偏置很适合处理图像，但它有几个问题：

1. **难以 scale**：当模型规模上到 10B 以上，U-Net 的训练不如 Transformer 稳定；
2. **跨模态融合受限**：U-Net 主要靠 cross-attention 注入文本/图像条件，深度有限；
3. **不适合统一多模态**：当条件不只是文本、还包括图像、音频、视频时，U-Net 的结构不够灵活。

DiT (Diffusion Transformer) 解决了前两个问题：把 U-Net 换成 ViT 风格的 Transformer，把图像 patchify 成 token 序列，然后用 Transformer block 去噪[^2]。

但**条件怎么注入到 DiT 里**，又出现了不同的设计哲学。这就引出了今天最热的两条路线：**多流 MMDiT** vs **单流 S3-DiT**。

### 8.2 多流 MMDiT：文本和图像各走一条流

代表作是阿里的 **Qwen-Image**[^5]。它是一个 20B 参数的多模态 Diffusion Transformer，核心架构包括三个组件：

1. **冻结的 Qwen2.5-VL**（视觉语言模型）：负责文本和图像的语义对齐；
2. **VAE encoder/decoder**：负责图像潜变量的压缩和重建；
3. **MMDiT diffusion backbone**：负责在 latent 空间去噪。

它的"多流"体现在：

```
text prompt ─→ Qwen2.5-VL ─→ semantic tokens ─┐
                                              ├─→ MMDiT (cross-modal fusion) ─→ noise
input image ─→ VAE ───────→ reconstruction tokens ┐
              + Qwen2.5-VL → semantic tokens   ──┘
              ↑
              这就是 "dual encoding" 机制
```

**Dual encoding 是 Qwen-Image 的关键创新**：同一张输入图同时被 Qwen2.5-VL 和 VAE 编码，前者提供语义信息，后者提供重建信息，两者在 MMDiT 里融合。这种设计在图像编辑任务上特别有用 —— 编辑既要"理解你想改什么"（语义），也要"保持原图其他地方不变"（重建保真度）[^5]。

### 8.3 单流 S3-DiT：所有 token 拼成一条序列

代表作是 **Z-Image**，由 Tongyi Lab 提出的 6B 参数 Scalable Single-Stream Diffusion Transformer (**S3-DiT**)[^6]。

它的设计哲学和 MMDiT 完全相反：**所有模态的 token 都拼到一条序列里，共享同一个 Transformer 处理**：

```
[text tokens | visual semantic tokens | noisy image VAE tokens]
                         ↓
                Single Transformer (S3-DiT)
                         ↓
             只取 image token 部分预测 noise
```

S3-DiT 的几个关键技术点：

1. **3D RoPE**：把文本、空间、通道位置都编码到同一空间；
2. **轻量的模态 stem**：每种模态有自己的小 MLP，把它映射到共享的 hidden space；
3. **FiLM 式条件适配器**：timestep 和 global condition 通过 FiLM-like scale/shift 注入；
4. **流匹配 + 蒸馏**：用 flow matching loss 训练，并通过蒸馏得到 Z-Image-Turbo，可以在消费级 GPU 上做亚秒级推理[^6]。

**为什么 6B 单流能挑战 20B 多流？** 因为单流架构的参数效率更高 —— 文本和图像共享同一套 Transformer 权重，避免了双流之间的冗余表征。

### 8.4 两种路线的对比

| 维度 | 多流 MMDiT (Qwen-Image) | 单流 S3-DiT (Z-Image) |
|---|---|---|
| **token 组织** | 文本流 + 图像流，分开处理后融合 | 所有 token 拼成统一序列 |
| **条件注入** | cross-modal fusion + dual encoding | unified self-attention + FiLM |
| **控制风格** | 显式、模块化、可分解 | 隐式、统一、端到端 |
| **优势** | 强语义、强编辑、复杂条件稳定 | 参数效率高、结构简洁、推理友好 |
| **劣势** | 参数巨大、计算重、系统复杂 | 长序列 attention 成本高，强空间控制需额外设计 |
| **典型场景** | 专业图像编辑、文字渲染、多条件控制 | 高效 T2I、统一多模态生成、轻量部署 |

### 8.5 直观比喻

如果把 Diffusion 模型比作一个画师：

- **多流 MMDiT** 像是一个团队：一个文本理解专家、一个图像理解专家、一个绘画师傅，三人开会沟通后画师傅落笔。每个人都是该领域的专家，分工明确，但维护成本高。
- **单流 S3-DiT** 像是一个全能选手：他自己读 prompt、看参考图、想空间结构、动笔作画，全在一个脑子里完成。沟通成本低、效率高，但需要训练数据足够多样才能学会所有这些技能。

---

## 9. 一个例子：图像编辑任务下两种路线怎么处理

假设任务是：

> "输入一张人像，把衣服改成红色西装，脸和背景保持不变。"

**多流 MMDiT 的做法**：

```
原图          ─→ Qwen2.5-VL ─→ "这是一个穿白衬衫的男人" (语义)
              ─→ VAE        ─→ pixel-level 重建特征
文本指令       ─→ Qwen2.5-VL ─→ "把衣服改成红色西装" (编辑指令)
noisy target  ─→ MMDiT denoising stream

MMDiT 通过显式的 cross-modal fusion：
- 语义层面理解"衣服→红色西装"；
- 重建层面知道"脸和背景保持原样"；
- denoising 层面生成最终结果。
```

**单流 S3-DiT 的做法**：

```
[ text instruction tokens
| input image semantic tokens
| input image VAE tokens
| noisy target tokens ]
                ↓
        Single Transformer
                ↓
        从 target tokens 输出 noise
```

模型自己通过 attention 学习"哪些 token 对哪些区域重要"。这种端到端的学习理论上更灵活，但需要大量精心标注的编辑数据才能训稳。

---

## 10. 未来趋势：混合路线 + 多模态统一

看完前面这一切，我对未来 Diffusion 架构的演进有几个判断。

### 10.1 单流会成为基础架构，但不会"纯单流"

单流的优势是简洁和效率，但它在**强空间控制**（pose、depth、edge）和**复杂编辑**上还有差距。最实用的系统大概率是 hybrid：

```
Single-stream DiT backbone (主干)
    + Control branch (空间结构控制，类似 ControlNet)
    + Reference adapter (参考图，类似 IP-Adapter)
    + Mask/edit branch (区域级编辑)
    + Layout/typography expert (排版、文字渲染)
```

也就是说，**单流解决"统一和效率"，专门分支解决"精确和可控"**。

### 10.2 控制粒度从 prompt-level 走向 region-level

现在的主流是"一句 prompt 控制整张图"。但实际场景中，用户更需要：

- 这个区域保持不变；
- 这个物体换材质；
- 这个人保持身份；
- 这几个字必须准确显示；
- 这个 logo 不能变形。

这要求模型支持**区域绑定、对象绑定、文字位置绑定、多参考图绑定**。Qwen-Image 强调的"复杂文字渲染"和"精确图像编辑"已经在往这个方向走。

### 10.3 生成与编辑会统一为一个模型

以前模型经常分得很细：T2I model、inpainting model、editing model、ControlNet model……。未来会逐渐合并：

> 一个模型同时支持 T2I、image editing、multi-image composition、style transfer、layout generation、text rendering、object replacement。

Qwen-Image 的 multi-task training 已经包括 T2I、TI2I (text+image→image)、I2I reconstruction[^5]。这说明主流方向已经是**统一训练**，而不是每个任务单独一个模型。

### 10.4 高效化会变成核心竞争力

Z-Image 的 6B 单流路线说明：**不是只能靠堆参数**。数据质量、架构设计、蒸馏和推理优化同样重要。未来会越来越重视：

- few-step generation（4 步、2 步、甚至 1 步采样）；
- distillation（把大模型的能力蒸馏到小模型）；
- FP8 / INT8 / NF4 quantization；
- KV cache / feature cache；
- MoE DiT（稀疏激活的 Diffusion Transformer）；
- consumer GPU fine-tuning。

### 10.5 跨模态联合学习将成为标配

未来的"条件"不会只有文本和图像。视频、音频、3D、甚至 IMU/sensor 数据都会一起训练。模型需要更复杂的条件注入策略来处理多模态信息 —— adaLN、cross-attention、ControlNet-like branch、IP-Adapter-like adapter 都会在不同位置发挥作用。

---

## 11. 总结

回到一开始那个问题：**怎么把"用户想要什么"告诉模型？**

过去几年，Diffusion 社区给出的答案大致是这样一条主线：

1. **Channel Concat**：条件和图像在同一空间结构 → 直接拼通道。简单粗暴，适合 inpainting。
2. **Cross-Attention**：条件是变长语义 → 让图像 token 去 attend 它。文本/音频条件的标配。
3. **adaLN-Zero**：条件是全局信号 → 用它生成 LayerNorm 的 scale/shift/gate。极低开销，DiT 标配。
4. **ControlNet**：条件是空间结构图 → 复制一份 encoder + 零卷积。强空间控制，但每种条件要训一个分支。
5. **IP-Adapter**：条件是参考图 → 解耦 cross-attention，给图像单开一套。极小参数实现图像 prompt。
6. **多流 MMDiT (Qwen-Image)**：文本和图像各走一条流，通过 cross-modal fusion 融合。强编辑、强语义。
7. **单流 S3-DiT (Z-Image)**：所有 token 拼成一条序列共享 Transformer。参数高效、推理友好。

**核心洞察**：这些机制不是替代关系，而是互补的。一个现代 Diffusion 系统往往同时用到多种 —— DiT 主干用 adaLN-Zero 做时间步调制，cross-attention 接文本，ControlNet 做空间控制，IP-Adapter 做参考图。理解每种机制的"擅长什么、不擅长什么"，比死记某一个架构更有用。

未来的方向是**单流主干 + 多种专门分支的 hybrid 架构**，再叠加蒸馏、量化和高效推理。从条件注入的角度看，这场演化远没有结束。

---

## Sources

- [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [LatentSync (字节, 2024)](https://github.com/bytedance/LatentSync)
- [Stable Diffusion / Latent Diffusion (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752)
- [DiT: Scalable Diffusion Models with Transformers (Peebles & Xie, 2022)](https://arxiv.org/abs/2212.09748)
- [ControlNet (Zhang et al., 2023)](https://arxiv.org/abs/2302.05543)
- [IP-Adapter (Ye et al., 2023)](https://arxiv.org/abs/2308.06721)
- [Qwen-Image 官方仓库](https://github.com/QwenLM/Qwen-Image)
- [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)
- [Z-Image / S3-DiT](https://github.com/Tongyi-MAI/Z-Image)
- [FiLM (Perez et al., 2017)](https://arxiv.org/abs/1709.07871)
- [Flow Matching for Generative Modeling (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747)

[^1]: DiT 论文中比较了 in-context conditioning、cross-attention、adaLN 三种条件注入方式，cross-attention 比 adaLN 多约 15% Gflops。
[^2]: Peebles & Xie. *Scalable Diffusion Models with Transformers*. ICCV 2023. adaLN-Zero 把 residual block 中调制参数初始化为零，使 block 初始接近 identity function。
[^3]: Zhang et al. *Adding Conditional Control to Text-to-Image Diffusion Models*. ICCV 2023. ControlNet 通过 zero-initialized convolution 让参数从零逐渐增长，避免训练初期破坏预训练模型。
[^4]: Ye et al. *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models*. 2023. 关键设计是 decoupled cross-attention，把 text 和 image 的 cross-attention 分开。
[^5]: Qwen-Image 是 20B MMDiT 模型，使用 Qwen2.5-VL + VAE 双编码机制，支持 T2I、TI2I、I2I 多任务训练，强调复杂文字渲染和图像编辑能力。
[^6]: Z-Image 是 6B 参数的 Scalable Single-Stream Diffusion Transformer (S3-DiT)，把 text、visual semantic tokens、image VAE tokens 拼成统一输入流，通过 3D RoPE 和 FiLM 适配器注入条件，支持亚秒级推理。
