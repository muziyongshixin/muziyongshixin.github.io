---
title: '从 Classifier Guidance 到 Classifier-Free Guidance：一文讲清 Diffusion 里的 CFG'
date: 2026-04-20
permalink: /posts/2026/04/diffusion-cfg/
categories:
  - blog
tags:
  - diffusion
  - generative model
  - machine learning
  - computer vision
toc: true
---

Diffusion 模型发展到今天，`CFG` 几乎已经成了文本生成图像系统里的“默认组件”。  
但很多人第一次看到它时都会困惑：

- 为什么一个模型要做“有条件前向”一次、“无条件前向”一次？
- 为什么不能直接训练一个很强的条件模型？
- `Classifier Guidance` 和 `Classifier-Free Guidance` 到底是什么关系？
- 为什么后来的蒸馏模型又常常说“已经不需要 CFG 了”？

这篇文章想做的事情，就是把这条知识脉络从头理顺：  
从无条件 Diffusion 的起点出发，讲到 `Classifier Guidance`，再讲到今天真正主流的 `Classifier-Free Guidance (CFG)`，最后补上 few-step / distillation 路线和近两年的一些延伸工作。

<br />
<img align="center" width="1000" src="{{ site.url }}/images/posts/diffusion-cfg-evolution.svg" alt="Diffusion guidance evolution timeline">
<br />

上图可以先当作全文导航来读：  
最左边是无条件 diffusion 的起点，中间是两代 guidance，最右边是把 guidance 效果蒸进 few-step 模型的后续路线。

---

## 1. 起点：无条件 Diffusion 学的只是 `p(x)`

先从最原始的 DDPM 视角看问题。  
一个**无条件** diffusion 模型学的是数据分布 `p(x)`，也就是：

> 什么样的样本看起来像“真实世界中的自然图像”。

它并不知道你想生成什么。  
所以如果你只给它高斯噪声，它学会的是“把噪声慢慢拉回自然图像流形上”，而不是“把噪声拉成一只猫”或者“拉成一辆红色跑车”。

在 DDPM 的常见参数化里，前向加噪可以写成：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

训练时模型通常预测噪声：

$$
\epsilon_\theta(x_t, t)
$$

Ho 等人在 DDPM 里说明了这种噪声预测与 denoising score matching 存在紧密联系，因此我们常把 diffusion 模型理解为在不同噪声水平下学习一个 score function 的近似器。[1]

更直观一点说：

- `x_t` 是某个时刻的噪声图
- 模型要回答的问题是：“这张图里哪一部分是噪声？应该往哪个方向去噪？”
- 如果模型只学了 `p(x)`，它最多只能说“往更像自然图像的方向去”

但条件生成想做的是：

$$
p(x|c)
$$

也就是：

> 在满足条件 `c` 的前提下，什么样的图像是合理的？

这里的 `c` 可以是类别标签、文本、另一张图像、语音 embedding，甚至更复杂的多模态条件。

问题于是变成了：

> 怎么把条件信号注入去噪过程？

这条路上，Diffusion 社区先后给出了两代代表性答案。

---

## 2. 第一代方案：Classifier Guidance

`Classifier Guidance` 的代表工作是 Dhariwal 和 Nichol 在 2021 年的 *Diffusion Models Beat GANs on Image Synthesis*。[2]

它的核心想法非常优雅：  
先保留一个无条件 diffusion 模型来提供“自然图像先验”，然后再额外训练一个分类器，告诉模型“怎样更像某个类别”。

### 2.1 贝叶斯公式是整个故事的起点

对于类别条件 `c`，有：

$$
p(x_t|c) \propto p(c|x_t)\,p(x_t)
$$

两边取对数并对 `x_t` 求梯度，可得：

$$
\nabla_{x_t}\log p(x_t|c)=\nabla_{x_t}\log p(x_t)+\nabla_{x_t}\log p(c|x_t)
$$

这行式子的含义特别重要：

- `\nabla_{x_t}\log p(x_t)`：让样本更像“自然图像”
- `\nabla_{x_t}\log p(c|x_t)`：让样本更像“类别 c”

于是条件生成的 score 可以拆成：

> 无条件生成能力 + 分类器给的条件方向

### 2.2 它在采样时是怎么工作的

直觉化地写，Classifier Guidance 的采样过程可以理解为：

```text
Step 1: diffusion 模型在当前 x_t 上做一次前向
        得到无条件噪声预测或 score

Step 2: 分类器在同一个 x_t 上判断“这有多像类别 c”
        然后对 x_t 求梯度

Step 3: 用这个梯度去修正 diffusion 模型的去噪方向

Step 4: 按修正后的方向走一步，得到 x_{t-1}
```

在 Dhariwal & Nichol 的论文里，这个修正以“对采样均值加上分类器梯度项”的形式写进算法；如果改写到更常见的 `\epsilon`-prediction 记号中，会看到大家熟悉的“沿着分类器梯度方向做引导”。  
需要注意的是：**不同采样器、不同参数化下常数项会略有差异**，所以教程里出现的公式看起来可能不完全一样，但核心思想是一致的。[2]

### 2.3 这个方法为什么当时很重要

因为它第一次清楚地证明了：

> diffusion 模型也可以像 GAN 一样，通过引导在“样本保真度”和“分布覆盖度”之间做可控权衡。

Dhariwal & Nichol 在 ImageNet 上展示了很强的结果：通过 classifier guidance，他们把 conditional diffusion 的质量显著拉高，并把 diffusion 真正推到了“能和当时顶级 GAN 正面竞争”的阶段。[2]

### 2.4 但它有一个很重的代价：分类器必须在噪声图上工作

这是 Classifier Guidance 最大的工程痛点。

普通分类器只见过干净图像 `x_0`，但 diffusion 采样时给它的是各种噪声水平下的 `x_t`。  
因此你不能直接拿一个普通 ImageNet 分类器来引导 diffusion，而必须训练一个**噪声感知分类器**：

```python
def train_noisy_classifier(x0, y):
    t = sample_timestep()
    noise = torch.randn_like(x0)
    x_t = add_noise(x0, noise, t)
    logits = classifier(x_t, t)
    loss = cross_entropy(logits, y)
    return loss
```

也就是说，整个系统变成了：

1. 一个 diffusion 模型
2. 一个额外分类器
3. 分类器还要在所有噪声水平上都能稳定工作

而且推理时为了拿到 `\nabla_{x_t}\log p(c|x_t)`，还需要对分类器做反向传播，这会带来额外显存和速度开销。

---

## 3. 第二代方案：Classifier-Free Guidance

真正把 diffusion 条件生成推成主流工业范式的，是 Ho 和 Salimans 在 2021 年提出的 *Classifier-Free Diffusion Guidance*。[3]

这篇工作的标题其实就已经说明了本质：

> classifier guidance without a classifier

也就是：

> 我还想要 guidance 的效果，但我不想再训练一个分类器。

### 3.1 关键替换：把“分类器梯度”改写成两个 score 的差

从贝叶斯公式出发：

$$
p(c|x_t)=\frac{p(x_t|c)p(c)}{p(x_t)}
$$

对数化后对 `x_t` 求梯度：

$$
\nabla_{x_t}\log p(c|x_t)
=
\nabla_{x_t}\log p(x_t|c)
-\nabla_{x_t}\log p(x_t)
$$

这里 `p(c)` 对 `x_t` 来说是常数，所以梯度为 0。

这一步非常关键，因为它说明：

> 分类器梯度，本质上可以看成“条件 score”和“无条件 score”的差值。

而 diffusion 模型本来就在学 score 的近似。  
所以如果同一个模型既能输出条件版本，又能输出无条件版本，那么分类器的作用就可以被“模型自己的两次前向”替代。

### 3.2 从 score 形式到大家熟悉的 CFG 公式

在常见 VP diffusion / `\epsilon`-prediction 的记号下，可以把这件事写成：

$$
\hat{\epsilon}
=
\epsilon_\theta(x_t,t,\varnothing)
+ w\cdot\left[\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)\right]
$$

其中：

- `\epsilon_\theta(x_t,t,c)`：有条件噪声预测
- `\epsilon_\theta(x_t,t,\varnothing)`：无条件噪声预测
- `w`：guidance scale

有时你也会看到等价写法：

$$
\hat{\epsilon}
=
(1+w)\epsilon_\theta(x_t,t,c)-w\epsilon_\theta(x_t,t,\varnothing)
$$

两者完全等价，只是展开方式不同。

### 3.3 为什么 score 能改写成噪声预测

上面这一步里，很多人最容易卡住的地方是：

> 前面推的是 score，为什么后面突然变成了噪声预测 `\epsilon_\theta`？

关键在于 VP diffusion 的前向分布：

$$
q(x_t|x_0)=\mathcal{N}\left(\sqrt{\bar{\alpha}_t}x_0,\,(1-\bar{\alpha}_t)I\right)
$$

对 `x_t` 求对数梯度：

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}
=
-\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

因为：

$$
x_t-\sqrt{\bar{\alpha}_t}x_0=\sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

所以在这个参数化下，score 和噪声只差一个与时间步有关的缩放因子。  
这也是为什么 diffusion 文献里常把：

$$
\nabla_{x_t}\log p_t(x_t|c)
$$

近似写成：

$$
-\frac{1}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t,c)
$$

严格地说，这里对应的是扰动后边缘分布的 score 近似；不同参数化和不同 sampler 下写法会有差别，但对理解 CFG 来说，这个关系已经足够用了。

### 3.4 直觉上它到底在干什么

把上面的公式拆开看：

$$
\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)
$$

这一项可以理解为：

> 条件 `c` 相对于“什么都不说”所带来的额外方向

所以 CFG 其实是在做一件非常朴素的事：

1. 先得到“自然去噪方向”
2. 再提取出“条件带来的额外偏移”
3. 把这部分偏移乘上一个更大的系数

这就是为什么很多人把 CFG 理解成“方向放大器”。  
它不是凭空发明一个新方向，而是在**有条件**和**无条件**之间做对比，把“条件真正贡献的那一小段方向”放大。

---

## 4. 训练阶段：为什么一个模型能同时学会有条件和无条件

CFG 成立的前提是：

> 同一个模型既会做 `\epsilon(x_t,t,c)`，也会做 `\epsilon(x_t,t,\varnothing)`。

Ho 和 Salimans 的做法很简单：**训练时随机把条件丢掉**。[3]

伪代码如下：

```python
def training_step(x0, c):
    t = sample_timestep()
    noise = torch.randn_like(x0)
    x_t = add_noise(x0, noise, t)

    if random() < p_drop:
        c = null_condition

    eps_pred = model(x_t, t, c)
    loss = mse(eps_pred, noise)
    return loss
```

其中 `p_drop` 通常设成 10% 到 20% 左右。

这意味着训练过程中模型会见到两类样本：

- 大多数时候：正常条件训练
- 少数时候：条件被替换为空条件

于是模型自然学会了两种行为模式：

```text
输入真实条件 c   -> 输出条件去噪结果
输入空条件 ∅     -> 输出无条件去噪结果
```

所以推理时只要把同一个 `x_t` 喂给模型两次：

- 一次给真实条件
- 一次给空条件

就能得到 CFG 所需的两个分支。

---

## 5. Classifier Guidance 和 CFG 到底差在哪里

把两代方法摆在一起，对比会非常清楚。

| 维度 | Classifier Guidance | Classifier-Free Guidance |
|---|---|---|
| 需要几个模型 | 2 个 | 1 个 |
| 是否需要额外分类器 | 需要 | 不需要 |
| 分类器是否要适配噪声图 | 需要 | 不需要 |
| 训练方式 | diffusion 无条件训练 + 分类器单独训练 | 条件训练 + condition dropout |
| 推理开销 | diffusion 前向 + 分类器反向 | 两次 diffusion 前向 |
| 条件类型 | 更适合离散类别 | 任意 embedding 条件 |
| 工程复杂度 | 高 | 低 |
| 当前使用情况 | 历史重要，但已非主流 | 现代主流 |

如果只看数学推导，这两者像是“同一家族的两个版本”。  
但如果从工程实现看，差距非常大。

<br />
<img align="center" width="1000" src="{{ site.url }}/images/posts/diffusion-cfg-compare.svg" alt="Classifier Guidance versus Classifier-Free Guidance">
<br />

如果你只想先抓住最核心的工程差异，那么看这张图就够了：  
`Classifier Guidance` 是“diffusion 模型 + 外部分类器”，`CFG` 则是“同一个 diffusion 模型跑两次”。

### 5.1 为什么 Classifier Guidance 会被边缘化

主要有四个原因。

#### 原因 1：分类器负担太重

你不只是多训练了一个网络，而是多训练了一个**噪声鲁棒**分类器。  
这件事本身就不轻，而且每换一种条件形式都要重来。

#### 原因 2：推理要反向传播

Classifier Guidance 采样时需要拿分类器对 `x_t` 的梯度。  
这意味着：

- 需要保留更多中间激活
- 显存压力更大
- 速度通常不如“两次前向”的 CFG

#### 原因 3：条件类型太受限

分类器天然适合“猫 / 狗 / 车”这种离散标签。  
但现代生成任务的条件往往是：

- 长文本 prompt
- 图像条件
- layout / depth / segmentation
- 多模态混合 embedding

这时候“训练一个噪声图上的条件分类器”就变得很不自然。

#### 原因 4：高噪声阶段梯度不稳定

当 `x_t` 很接近纯噪声时，分类器给出的判断很容易不可靠。  
这会导致引导方向 noisy、过激，甚至带来类似对抗样本的伪影。

---

## 6. 为什么“直接做条件生成”还不够，偏偏要有无条件分支

这往往是大家第一次学 CFG 时最困惑的点。

一个很自然的问题是：

> 如果模型本来就是条件模型，为什么不直接用 `\epsilon_\theta(x_t,t,c)` 去采样？  
> 为什么还要再算一次无条件分支？

答案是：**因为 CFG 不只是“让模型有条件”，而是在推理阶段对条件方向进行再加权。**
这并不是说纯条件模型不能工作，而是说它少了一个可以在推理时显式放大条件信号的控制手柄。

### 6.1 纯条件模型只是在做“条件均值意义下的去噪”

如果 prompt 很宽泛，比如“a cat”，那么满足这个条件的图像分布其实很宽：

- 橘猫
- 黑猫
- 正脸
- 侧脸
- 写实
- 插画

单纯条件模型学到的是这个条件分布下的平均去噪规律。  
而 MSE 型训练目标天然倾向于学习“平均意义上最稳妥的预测”。

结果往往是：

- 条件满足了
- 但语义不够“尖锐”
- 细节和风格不够坚定

CFG 做的事情则更像是在说：

> 我不仅要满足条件，我还要更坚定地朝条件特征前进。

### 6.2 无条件分支提供了一个“基线”

这一点非常重要。

有了无条件预测，你才能问出下面这个问题：

> 相比于“什么都不说”时的去噪方向，这个条件到底额外改变了什么？

也就是这项：

$$
\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)
$$

这就是条件的“纯增量”。

没有无条件分支，你只有 `\epsilon_\theta(x_t,t,c)`，却不知道里面有多少是：

- 来自数据分布本身的通用图像先验
- 来自条件 `c` 的额外要求

CFG 恰恰把这两部分显式分离开了。

### 6.3 从分布角度看，CFG 可以理解为“后验锐化”

很多教程会把 CFG 理解成对条件分布的 sharpen。  
直觉上它对应于：

$$
\tilde{p}(x|c)\propto \frac{p(x|c)^w}{p(x)^{w-1}}
$$

也就是在对数空间里放大条件分布相对于无条件分布的优势方向。

这个视角对于理解“为什么更贴 prompt，但多样性下降”非常有帮助：  
`w` 越大，分布越尖，语义通常越强，但 mode coverage 往往会下降。

这里要加一个重要注记：

> 上面这个“锐化分布”视角是一个非常有用的直觉，但并不是对所有离散采样器、所有有限步采样过程都严格成立的完整结论。

近年的理论工作开始更仔细地讨论 CFG 的有限步行为，以及它为什么会出现“过饱和、模式坍缩、编辑不可逆”等副作用；例如 CFG++ 就把部分问题解释为 off-manifold 现象。[7]

---

## 7. CFG 的实际效果，为什么它能长期统治文本生成图像

CFG 能成为主流，不只是因为“省了一个分类器”，更因为它刚好踩中了现代大模型系统的需求。

### 7.1 它天然支持任意条件 embedding

只要条件能被编码成向量，CFG 就能工作：

- class embedding
- text encoder 输出
- image encoder 输出
- audio embedding
- layout / depth / pose control signal

这跟文本生成图像时代的需求几乎完美匹配。

### 7.2 它给了推理阶段一个可调节旋钮

`guidance scale` 是一个极其实用的控制参数。

- 小一些：更多样，但可能没那么贴 prompt
- 大一些：更贴 prompt，但更容易失真、过饱和、重复

这让同一个基础模型能覆盖很多场景，而不必为每种“对齐强度”重新训练一份。

### 7.3 它非常契合 latent diffusion / Stable Diffusion 这类架构

现代文本生成图像系统通常采用：

1. 文本编码器把 prompt 编成 embedding
2. latent diffusion / UNet 在潜空间做去噪
3. 采样时同时跑 conditional 和 unconditional 分支
4. 用 CFG 线性组合

这套接口很简单，也很模块化。  
所以从 Stable Diffusion 到许多后来的文本生成图像系统，CFG 都成为了默认推理机制。

---

## 8. CFG 的副作用：为什么 guidance scale 不能无限加大

CFG 不是越大越好。

如果把 `w` 开得很高，常见问题包括：

- 图像过饱和
- 纹理僵硬
- 构图重复
- 多样性下降
- 细节出现“被强行往 prompt 上扯”的伪影

这背后的直觉并不难理解：

> 你在不断放大“条件增量方向”，但这个方向本来只是一个局部修正。  
> 放大过头，就会从“更对齐”变成“过度纠偏”。

很多用户在 Stable Diffusion 里都有很直观的经验：

- `scale` 太小，图像“没听话”
- `scale` 太大，图像“过于用力”

从理论和经验上看，CFG 一直在做一件 trade-off：

> 对齐更强，通常意味着多样性更弱。

这也是后来很多改进工作的出发点。

---

## 9. 后续发展：为什么蒸馏模型常常“不再需要 CFG”

理解了 CFG 之后，再看 few-step 模型就顺了。

很多人第一次看到 LCM、SDXL Turbo 这类模型时会觉得奇怪：

> 为什么原始模型要几十步、还要 CFG；  
> 蒸馏后的模型却几步就行，甚至不再依赖传统 CFG？

答案是：

> 因为 CFG 的效果可以在训练时被“蒸”进学生模型里。

### 9.1 Progressive Distillation：先解决“步数太多”

Salimans 和 Ho 在 2022 年提出 *Progressive Distillation for Fast Sampling of Diffusion Models*，核心思想是把一个多步 deterministic sampler 逐轮蒸成更少步数的模型，每一轮把步数减半。[4]

它解决的是：

> diffusion 太慢，如何把 8192 步、1024 步慢慢蒸成 4 步？

这一步不一定直接针对 CFG，但为后面 few-step 生成打了基础。

### 9.2 Consistency Models：直接学习从噪声到数据的快速映射

Song 等人在 2023 年提出 *Consistency Models*，把目标推进到“一步或少步生成”。[5]

它的关键思想是让模型学会不同时间点之间的一致映射，从而绕开传统 diffusion 逐步积分的高成本。

### 9.3 Latent Consistency Models：把 CFG 蒸进 latent diffusion 体系

真正和现代文本生成图像工作流贴得很近的是 2023 年的 *Latent Consistency Models (LCM)*。[6]

LCM 很关键的一句话是：

> 它是从**预训练的 classifier-free guided diffusion models** 高效蒸馏出来的。

换句话说，teacher 本身就带着 CFG 的行为。  
学生模型学习的是：

> teacher 做完 guidance 之后的结果

于是推理阶段就不必再显式执行：

1. 一次 conditional 前向
2. 一次 unconditional 前向
3. 线性组合

学生模型已经把“有 guidance 的好处”折进自己参数里了。

### 9.4 Adversarial Diffusion Distillation：把 few-step 做到更激进

2023 年的 *Adversarial Diffusion Distillation (ADD)* 更进一步，把 few-step / one-step 的质量继续往上推。[8]

它利用预训练 diffusion 模型作为 teacher signal，再加上 adversarial loss，目标是在极少步数下依然维持高质量图像。

所以如果把这一整条线串起来，你会得到一个很清晰的演化逻辑：

```text
先有多步 diffusion
-> 再有 CFG，让条件生成更强
-> 再有蒸馏，把“多步 + CFG 的能力”压缩进少步学生模型
-> 最后出现 few-step / one-step 的实用系统
```

这也是为什么今天很多快速模型虽然表面上“不再跑传统 CFG”，但它们背后的 teacher 往往仍然深受 CFG 体系影响。

---

## 10. 最近两年的一些延伸：大家在改进 CFG 的什么

如果说 2021 到 2023 年的主线是“把 CFG 变成标准件”，那么 2024 到 2025 年的很多工作则是在回答：

> CFG 很有用，但它到底哪里还不够好？

### 10.1 Self-Attention Guidance：不用额外条件，也能做训练自由的引导

Hong 等人在 *Self-Attention Guidance* 中提出，除了 classifier guidance 和 CFG 之外，还可以利用模型内部 self-attention 信息来做 guidance。[9]

这个方向的重要意义在于：

- guidance 不一定非得来自外部分类器
- guidance 也不一定非得来自条件 dropout 训练
- 可以从模型内部结构本身提取“纠偏信号”

### 10.2 PAG：把 guidance 扩展到无条件与下游任务场景

2024 年的 *Perturbed-Attention Guidance (PAG)* 则进一步展示：  
即使在 unconditional generation 或某些 CFG 不方便使用的任务里，也可以通过扰动 attention 构造 guidance 信号。[10]

这说明一个更大的趋势：

> “guidance” 已经不再只是一条固定公式，而是在演化成一个更宽的推理控制框架。

### 10.3 CFG++：讨论 vanilla CFG 的 off-manifold 问题

2025 年的 *CFG++* 指出，传统 CFG 的一些副作用并不一定是 diffusion 本身的问题，而可能和 CFG 把采样轨迹推离数据流形有关。[7]

这类工作之所以值得关注，是因为它们开始从“经验调 scale”走向：

- 更系统地理解 CFG 为什么有效
- 更具体地解释 CFG 为什么会失真
- 更有针对性地修复它的缺点

---

## 11. 一张总图，把整条知识脉络串起来

```text
DDPM / score-based diffusion
  先学会从噪声中恢复“自然样本”
  核心对象是 p(x) 或其 score

        |
        v

Classifier Guidance (2021)
  用无条件 diffusion 提供 p(x)
  再训练噪声分类器提供 ∇ log p(c|x_t)
  条件 score = 无条件 score + 分类器梯度

        |
        v

Classifier-Free Guidance (2021/2022)
  不再训练分类器
  通过条件 dropout 让同一模型同时学会：
    ε(x_t, t, c)
    ε(x_t, t, ∅)
  采样时做：
    ε_cfg = ε_∅ + w (ε_c - ε_∅)

        |
        v

现代文本生成图像系统
  CFG 成为默认推理机制
  通过 guidance scale 控制 prompt 对齐和多样性

        |
        v

Few-step / Distillation 路线
  Progressive Distillation
  Consistency Models
  LCM
  ADD

  把“多步 + CFG”的效果蒸进学生模型

        |
        v

近期改进
  SAG / PAG / CFG++
  从训练自由 guidance、attention guidance、
  以及理论修正等角度继续优化 CFG
```

---

## 12. 一个最常见的误区

很多教程会把 CFG 说成：

> “就是条件减去无条件，再乘一个 scale。”

这当然没错，但如果只停在这一步，会漏掉最关键的理解：

> `ε_cond - ε_uncond` 不是一个拍脑袋的 engineering trick，  
> 它来自贝叶斯公式下“分类器梯度 = 条件 score - 无条件 score”的推导。

也就是说，CFG 不是“经验上好用的 hack”，而是一个有明确 probabilistic 来历、又极度工程友好的近似方案。

它真正厉害的地方在于：

- 数学上和 classifier guidance 是一脉相承的
- 工程上却省掉了最麻烦的那个分类器
- 还顺手把条件接口扩展成了任意 embedding

这就是为什么它几乎成为现代 diffusion 条件生成的默认答案。

---

## 13. 总结

如果只用一句话总结整条脉络，我会写成：

> `Classifier Guidance` 证明了 diffusion 可以被“引导”；  
> `Classifier-Free Guidance` 则把这种引导从一个昂贵、受限的两模型系统，变成了一个几乎所有条件 diffusion 都能直接使用的标准模块。

更具体地说：

1. 无条件 diffusion 只学 `p(x)`，不知道用户要什么。
2. Classifier Guidance 用额外分类器给出 `\nabla \log p(c|x_t)`，第一次把条件引导明确做进 diffusion 采样。
3. CFG 发现这个分类器梯度可以由“条件 score - 无条件 score”替代，于是只靠一个模型、两次前向就能完成引导。
4. CFG 因为简单、通用、兼容文本条件，最终成为文本生成图像时代的主流。
5. 后续的蒸馏与 consistency 路线，又把“多步 + CFG”的能力进一步压缩进 few-step 模型。

所以从历史上看，CFG 不是 diffusion 里的一个小技巧，它几乎就是现代条件 diffusion 能真正大规模落地的关键转折点之一。

---

## 参考资料

1. Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.  
   [Paper](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)

2. Dhariwal, Nichol. *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS 2021.  
   [Paper](https://proceedings.neurips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)

3. Ho, Salimans. *Classifier-Free Diffusion Guidance*. NeurIPS 2021 Workshop / OpenReview.  
   [OpenReview](https://openreview.net/forum?id=qw8AKxfYbI)

4. Salimans, Ho. *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR 2022.  
   [arXiv](https://arxiv.org/abs/2202.00512)

5. Song, Dhariwal, Chen, Sutskever. *Consistency Models*. ICML 2023.  
   [PMLR](https://proceedings.mlr.press/v202/song23a.html)

6. Luo, Tan, Huang, Li, Zhao. *Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference*. 2023.  
   [arXiv](https://arxiv.org/abs/2310.04378)

7. Chung, Kim, Park, Nam, Ye. *CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models*. ICLR 2025.  
   [OpenReview](https://openreview.net/forum?id=E77uvbOTtp)

8. Sauer, Lorenz, Blattmann, Rombach. *Adversarial Diffusion Distillation*. 2023.  
   [arXiv](https://arxiv.org/abs/2311.17042)

9. Hong, Lee, Jang, Kim. *Improving Sample Quality of Diffusion Models Using Self-Attention Guidance*. ICCV 2023.  
   [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Hong_Improving_Sample_Quality_of_Diffusion_Models_Using_Self-Attention_Guidance_ICCV_2023_paper.html)

10. Ahn et al. *Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance*. ECCV 2024 / arXiv.  
    [arXiv](https://arxiv.org/abs/2403.17377)
