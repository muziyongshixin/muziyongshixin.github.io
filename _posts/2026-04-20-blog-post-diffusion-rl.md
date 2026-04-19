---
title: '从 DDPO 到 Flow-GRPO：一文看懂 Diffusion 模型的强化学习过程与发展脉络'
date: 2026-04-20
permalink: /posts/2026/04/diffusion-rl/
categories:
  - blog
tags:
  - diffusion
  - reinforcement learning
  - generative model
  - machine learning
  - computer vision
toc: true
---

Diffusion 模型最初是按“去噪 MSE / 似然近似”来训练的，但真正上线时，我们更关心的往往不是似然，而是：

- 人类是否更喜欢这张图
- 图像和 prompt 是否更对齐
- 视频动作是否更连贯
- 输出是否更安全
- 结果是否更符合物理或任务约束

这类目标通常没有一个漂亮、统一、可微、稳定的监督损失。  
于是 2023 年开始，一批工作把 Diffusion 的采样过程重新解释成**多步决策**，再用 RL、偏好优化、reward backprop 等方法对它做后训练。

这篇文章想讲清两件事：

1. Diffusion 模型为什么能被看成一个 RL 问题，以及一次 RL fine-tuning 到底在做什么。
2. 这条线是如何从 `DDPO / DPOK`，发展到 `Diffusion-DPO / D3PO`、`DRaFT / AlignProp`、视频对齐，再走到 `Flow-GRPO` 和 2026 年更高效的方法。

<br />
<img align="center" width="1000" src="{{ site.url }}/images/posts/diffusion-rl-evolution.svg" alt="Diffusion RL evolution timeline">
<br />

上图可以先当全文导航：  
2023 年是“把去噪过程变成决策过程”的起点；2023 年下半年到 2024 年，方法开始沿着不同反馈类型分叉；2025 年进入 Flow Matching 时代；到 2026 年，研究重点明显转向了**效率、credit assignment 和对 reverse process likelihood 的替代**。

---

## 1. 为什么要用 RL 优化 Diffusion

标准 Diffusion 训练，学到的是一个“如何把噪声慢慢拉回数据分布”的模型。  
在 DDPM 记号下，它通常优化的是噪声预测误差：

$$
\mathcal{L}_{\text{diffusion}} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\right]
$$

这个目标和“生成结果更符合人类偏好”之间，并不是一回事。

更准确地说，预训练阶段优化的是：

> 生成样本要像训练分布中的样本。

而后训练阶段往往优化的是：

> 在不要严重偏离预训练分布的前提下，让样本在某个外部指标上拿更高分。

外部指标可以是：

- `ImageReward`、`HPSv2` 这类偏好或美学 reward model
- `CLIP`、VLM 或 OCR-based 的对齐指标
- 视频里的时序一致性、运动平滑性
- 分子、蛋白、材料里的任务分数
- 人类偏好对本身

RL 的价值就在这里：  
你不再要求目标必须长得像一个监督学习 loss，而是只要求它能为最终样本给出一个分数，或者至少给出偏好关系。

---

## 2. 关键重写：去噪过程其实是一个 MDP

Diffusion RL 的真正起点，不是某个具体算法，而是这个重写：

```text
state      s_t = (x_t, t, c)
action     a_t = sample or predict the next denoising move
transition x_t -> x_{t-1}
reward     r(x_0, c) or preference over final samples
policy     the diffusion / flow model itself
```

如果用 DDPM 风格的随机反向过程来看，模型每一步都定义了一个条件高斯分布：

$$
p_\theta(x_{t-1}\mid x_t, c)=\mathcal{N}(\mu_\theta(x_t,t,c), \sigma_t^2 I)
$$

这时“动作”可以理解为：  
**在状态 `x_t` 下，策略选择了一个从高斯反向转移里采样出来的 `x_{t-1}`。**

于是整条采样轨迹就像：

```text
x_T -> x_{T-1} -> ... -> x_1 -> x_0
```

最后只在 `x_0` 上拿到一个终止 reward。  
这正是 RL 里最经典、也最麻烦的一类问题：

- 奖励是稀疏的
- credit assignment 跨越很多步
- 你还不希望模型把预训练学到的图像先验彻底破坏掉

### 2.1 `compute_log_prob` 到底在算什么

很多人第一次看 DDPO 代码时，最困惑的是 `compute_log_prob`。  
它算的其实非常朴素：

> 在当前策略下，模型把 `x_t` 变成这一步实际采样到的 `x_{t-1}`，这件事的对数概率有多大。

因为反向过程是高斯分布，所以：

$$
\log p_\theta(x_{t-1}\mid x_t,c)
\propto
-\frac{1}{2\sigma_t^2}\|x_{t-1}-\mu_\theta(x_t,t,c)\|^2
$$

这件事之所以重要，是因为 policy gradient 需要的正是：

$$
\nabla_\theta \log \pi_\theta(a_t\mid s_t)
$$

在 Diffusion 里，它就变成了每一步反向转移的 log-prob gradient。

### 2.2 为什么 DDPM 比 DDIM 更容易做 policy gradient

如果你用的是 DDPM 风格采样，每一步天然有随机性，高斯转移和 log-prob 都是良定义的。  
但 DDIM 和很多 Flow Matching 采样器本质上更接近确定性 ODE，这会带来两个麻烦：

1. exploration 不够自然
2. likelihood / log-prob 不容易直接写出来

这也是为什么后面 `Flow-GRPO` 需要专门做 **ODE-to-SDE conversion**，而 2026 年一些方法开始尝试**不再直接依赖 reverse-process likelihood**。[11][13]

---

## 3. 一次 Diffusion RL 训练迭代到底发生了什么

先不急着分论文，先看一次最典型的 `DDPO / PPO` 风格训练循环。  
把细节都压缩掉，它本质上只有五步：

### 3.1 第一步：用旧策略生成完整轨迹

给一批 prompt，用当前模型 `\theta_{\text{old}}` 从噪声开始生成图像，并记录整条去噪轨迹：

- 每个时间步的 `x_t`
- 实际采样得到的 `x_{t-1}`
- 旧策略下对应的 `old_log_prob`

### 3.2 第二步：对最终样本打分

只在最后生成出的 `x_0` 上调用 reward：

$$
r = r(x_0, c)
$$

这个 reward 可以是：

- 美学分数
- prompt 对齐分数
- 安全分数
- 视频时序 reward
- 人类偏好数据训练出来的 reward model

### 3.3 第三步：把 reward 变成 advantage

最简单时，整条轨迹共享同一个终止 reward；更稳一些时会做 batch normalization、baseline 或组内归一化：

$$
A = \frac{r - \mathrm{mean}(r)}{\mathrm{std}(r) + \epsilon}
$$

### 3.4 第四步：用新参数重算每一步 log prob

对同一条轨迹，用当前正在更新的参数重新计算：

$$
\rho_t = \exp\left(\log p_\theta(x_{t-1}\mid x_t,c) - \log p_{\theta_{\text{old}}}(x_{t-1}\mid x_t,c)\right)
$$

然后套进 PPO clip 或带 KL 的目标里。

### 3.5 第五步：让高 reward 轨迹更可能、低 reward 轨迹更不可能

如果一张图最终分数高，就提升这条去噪轨迹上各步动作的概率；  
如果一张图分数低，就降低这些动作的概率。

直觉上，Diffusion RL 学到的不是某个“神秘奖励魔法”，而是：

> 哪些去噪路径更容易通向人类真正想要的结果。

---

## 4. 发展脉络：这条线是怎么长出来的

到这里已经能理解“过程”了，接下来再看历史就会顺很多。  
我更推荐按**反馈类型**和**建模约束**来看，而不是只按年份背论文名。

### 4.1 2023：从 reward model 到 online RL

2023 年最重要的转折，是社区开始承认：

> Diffusion 的预训练目标和下游目标不一致，所以需要单独的 post-training。

这一年最早的标志性工作之一是 `ImageReward`。它不仅提出了一个通用文本生成图像 reward model，还给出了 `ReFL`，把 reward feedback 直接用于模型调优。[1]

紧接着，`DDPO` 在 2023 年 5 月把 denoising 明确建模成多步决策过程，并系统引入 policy gradient / PPO 风格更新。[2]

几天后，`DPOK` 进一步强调了 **KL regularization** 的重要性：  
你不只是要优化 reward，还要约束模型不要偏离预训练分布太远，否则很快就会 reward hacking、图像质量塌陷。[3]

这一阶段的核心思想可以压缩成一句话：

> 先承认 reward 存在，再把 Diffusion 当策略来优化。

### 4.2 2023 下半年到 2024：按反馈类型开始分叉

当“Diffusion 可以做 RL 后训练”这个大门打开后，下一步的问题自然变成：

> 你手里到底有什么反馈信号？

如果答案不同，方法也会不同。

#### 路线 A：你有黑盒 scalar reward

那就最适合 `DDPO / DPOK` 这种 policy gradient 路线：

- reward 可黑盒
- 不要求可微
- 但方差高，采样贵

#### 路线 B：你有偏好对，没有 reward model

那就更接近 `DPO` 家族。

`Diffusion-DPO` 在 2023 年 11 月把 LLM 里的 DPO 思路迁到 text-to-image diffusion：不先训 reward model，而是直接用人类偏好对优化模型相对偏好概率。[6]

同月提交、后被 CVPR 2024 接收的 `D3PO`（*Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model*）则进一步把“无 reward model 的直接偏好优化”扩展到多步 denoising MDP 视角。[7]

这一路的核心不是“直接做 RL”，而是：

> 如果你已经拿到了 winner / loser 对，就没必要再绕一圈学一个 reward model。

#### 路线 C：你的 reward 本身可微

那就完全没必要忍受 REINFORCE / PPO 的高方差。

`DRaFT` 在 2023 年 9 月提出，直接把 reward 梯度穿过采样过程反传回来，并进一步给出：

- `DRaFT-K`：只反传最后 `K` 步
- `DRaFT-LV`：在 `K=1` 时进一步降方差

它的关键 trade-off 很直白：

- 梯度更准
- 样本效率更高
- 但显存和反传成本更高

`AlignProp` 也属于这条 reward-backprop 路线，并通过 `LoRA + gradient checkpointing` 让直接反传更实用。[5]

这里有一个需要澄清的点：  
`AlignProp` 的 arXiv 条目后来被作者撤回，并在页面上注明内容被后续工作吸收；但“通过 reward gradient 直接调 diffusion”的路线本身没有消失，反而继续扩展到了视频。[5][9]

---

## 5. 三条主技术路线，到底该怎么区分

如果把 2023 到 2026 的方法压缩成一个表，最有用的不是“谁早谁晚”，而是下面这张图。

<br />
<img align="center" width="1000" src="{{ site.url }}/images/posts/diffusion-rl-selector.svg" alt="How to choose among diffusion RL methods">
<br />

再配合这张表会更直观：

| 路线 | 代表方法 | 你需要什么反馈 | 优点 | 代价 |
|---|---|---|---|---|
| Policy Gradient | DDPO, DPOK, Flow-GRPO | 黑盒 scalar reward | 通用，reward 不必可微 | 方差高，采样贵 |
| Preference Optimization | Diffusion-DPO, D3PO | winner / loser 偏好对 | 不必先训 reward model | 需要成对偏好数据，likelihood 近似更复杂 |
| Direct Backprop | DRaFT, AlignProp, Video Reward Gradients | 可微 reward | 梯度低方差，样本效率高 | 显存大，reward 必须可微 |

这里最值得记住的一点是：

> 这三条线不是互相否定，而是在回答不同的问题。

不是所有场景都该上 PPO，也不是所有场景都该上 DPO。  
真正的分界线其实是：**你的反馈是什么形式、能不能反传、采样预算有多贵。**

---

## 6. 视频生成把问题又抬高了一个量级

图像里已经够难了，视频里问题会立刻放大。

原因很简单。  
视频 reward 往往不是一个“单帧好不好看”的问题，而是至少同时包含：

- 单帧质量
- 文本对齐
- 时序一致性
- 运动合理性
- 镜头与物理连续性

而视频采样又比图像慢得多，显存开销也高得多。

### 6.1 InstructVideo：把视频 reward fine-tuning 重新表述成 editing

`InstructVideo` 在 2023 年 12 月提交、后被 CVPR 2024 接收。  
它的做法很典型：不是傻乎乎每次都把完整 DDIM 采样链跑到底，而是把 fine-tuning 重写成 editing，从而减少 full-chain sampling 成本。[8]

更重要的是，它面对了视频路线最现实的问题之一：

> 当时并没有一个像 ImageReward 那样成熟的视频偏好 reward model。

所以它把图像 reward model 重用于视频，并提出：

- `Segmental Video Reward`
- `Temporally Attenuated Reward`

本质上是在说：  
视频 reward 不能只看“最后整段视频的一个总分”，而要想办法把 reward 更稳定地分配到片段和时序结构上。

### 6.2 Video Diffusion Alignment via Reward Gradients：把 direct backprop 扩展到视频

2024 年 7 月的 `Video Diffusion Alignment via Reward Gradients` 则把 reward-backprop 路线明确推进到视频 Diffusion：  
既然 reward model 对 RGB 像素有稠密梯度，那就把这个梯度直接反传回视频生成过程。[9]

这条线的意义在于：

- 它说明 direct backprop 不只是图像 trick
- 在视频这种搜索空间更大、采样更贵的场景里，低方差梯度反而更有价值

所以如果你关心的是“视频 Diffusion 的 RL 怎么做”，真正要抓住的不是某个单独 paper 名，而是这两个事实：

1. 视频 reward 一定要显式考虑时序结构。
2. 由于采样成本太高，视频场景通常更偏爱 editing、局部反传、LoRA 和稀疏 reward 设计。

---

## 7. Flow Matching 时代：为什么 `Flow-GRPO` 是新的转折点

到了 2025 年，主流大模型里已经不全是传统 DDPM/DDIM 了。  
像 SD3、FLUX 这类系统更接近 **Flow Matching / Rectified Flow** 范式。

这时旧问题又回来了：

> RL 需要随机性和可处理的 log-prob；  
> 但 Flow Matching 的采样通常是确定性 ODE。

`Flow-GRPO` 在 2025 年 NeurIPS 上给出的回答是两个关键设计：[11]

### 7.1 ODE-to-SDE conversion

它把原本的确定性 ODE 改写成与原边际分布一致的 SDE，于是：

- 采样过程重新拥有随机性
- exploration 成立
- 每一步转移又能写成统计上可处理的形式

这一步非常重要，因为它不是在“给 Flow 模型硬套 DDPO”，而是在修补：

> Flow 模型天然不适合直接做 reverse-process policy gradient

### 7.2 Denoising Reduction

`Flow-GRPO` 的第二个关键点是：  
训练时减少 denoising steps，推理时保留原本的高质量步数。

这说明一个很现实的经验：

> RL 后训练并不一定需要最完美的生成样本，只需要足够稳定、足够可区分的 reward 信号。

从工程角度看，这让 Flow 模型的 RL 后训练第一次真正变得可用。

---

## 8. 截至 2026 年 4 月，这条线又在往哪里走

如果只看 2023 到 2025，你会觉得主线大概是：

`DDPO -> DPO / direct backprop -> Flow-GRPO`

但到 2026 年，研究重点已经明显转向了两个更细的问题。

### 8.1 更好的 credit assignment 和 rollout 复用

`TreeGRPO`（ICLR 2026）把 denoising 过程重写成一棵搜索树，通过共享轨迹前缀来提升样本效率，并尝试解决“终止 reward 被粗暴平均分给所有时间步”的问题。[12]

这件事说明社区已经意识到：  
**uniform terminal reward assignment** 其实是老一代 DDPO / GRPO 风格方法的一个核心瓶颈。

### 8.2 不再执着于 reverse-process likelihood

`DiffusionNFT`（ICLR 2026 Oral）更进一步，直接质疑了“必须在 reverse sampling 里估 log-prob 才能做 online RL”这个前提。[13]

它的出发点很清楚：

- reverse likelihood 受 solver 选择限制
- 和 CFG 的兼容性复杂
- trajectory 级策略优化的成本太高

所以它改走了 forward-process / flow-matching 风格目标，并声称在效率上显著超过 `Flow-GRPO`。[13]

这说明截至 **2026 年 4 月 20 日**，这个领域已经不再只是“把 PPO 套到去噪轨迹上”，而是在认真重构：

- policy objective 到底该写成什么
- log-prob 是不是必须的
- terminal reward 该如何分配到各步
- 采样器、CFG、solver 和 RL 目标能不能更自然地统一

---

## 9. 实践里最难的其实不是公式，而是 reward design

从工程上看，Diffusion RL 最大的敌人一直都不是“推不出梯度”，而是 **reward hacking**。

典型例子包括：

- 只优化美学分，模型会学会“糖水色”和过饱和
- 只优化 CLIP 对齐，模型可能学会骗 VLM
- 只优化时序一致性，视频可能变成几乎静止

所以大多数可用系统都会同时做四件事：

### 9.1 用 KL 约束守住预训练分布

`DPOK` 之后，这几乎已经是标配。  
你可以把它理解为一个护栏：

> 允许模型变好，但不允许它为了 reward 快速偏航。

### 9.2 用多维 reward 而不是单分数独裁

实际系统里更常见的是加权组合：

- 质量
- 对齐
- 安全
- 时序
- OCR / counting / composition

单一 reward 往往最容易被钻空子。

### 9.3 用 LoRA、低步数反传和局部更新控制成本

这是 `DRaFT`、`AlignProp`、视频 alignment 工作都反复验证过的经验：

- 不是所有参数都要动
- 不是所有时间步都要反传
- 不是每次都要完整链路采样

### 9.4 先问清楚“你手里的反馈是什么”

这是最重要的实操建议：

- 如果 reward 可微，优先考虑 direct backprop
- 如果只有偏好对，优先考虑 DPO 类方法
- 如果 reward 是黑盒且可调用，再考虑 policy gradient
- 如果模型已经是 Flow Matching，要特别注意随机性和 likelihood 的定义问题

---

## 10. 应该怎么选方法

如果只给一个实用版判断树，我会这样用：

1. reward 可微吗？
   可微：优先 `DRaFT / AlignProp / Video Reward Gradients` 这类直接反传路线。

2. reward 不可微，但有偏好对吗？
   有：优先 `Diffusion-DPO / D3PO`。

3. 既没有可微 reward，也没有偏好对，只有黑盒打分器？
   图像 DDPM 类：`DDPO / DPOK`。  
   Flow Matching 类：先看 `Flow-GRPO`，再关注 `DiffusionNFT` 这类新范式。

4. 是视频吗？
   默认把“reward 设计”和“采样成本”放在首位，再决定是 editing 式 fine-tuning、direct backprop，还是更传统的 RL 更新。

---

## 11. 总结

如果只用一句话总结这条脉络，我会写成：

> Diffusion RL 的本质，不是把 PPO 生搬硬套到生成模型上；  
> 而是把“逐步去噪”重新看成一个可优化的决策过程，再根据你手里的反馈形式，选择 policy gradient、偏好优化或 reward backprop 这三类不同工具。

更具体地说：

1. `DDPO / DPOK` 解决的是“黑盒 reward 怎么直接优化”。
2. `Diffusion-DPO / D3PO` 解决的是“只有偏好对时，能不能跳过 reward model”。
3. `DRaFT / AlignProp` 解决的是“如果 reward 可微，为什么还要忍受高方差 RL”。
4. `InstructVideo` 和后续视频工作说明，视频不是图像方法的简单复制，而是一个 reward design 和效率问题都更尖锐的场景。
5. `Flow-GRPO` 则标志着这条线正式进入 Flow Matching 时代。
6. 到 2026 年，研究已经开始进一步追问：`log-prob` 是否必须、credit assignment 能否更细、采样器与 RL 目标能否更统一。

所以从历史上看，Diffusion 模型的“强化学习过程”并不是一套固定算法，而是一条不断重新定义**策略、反馈、轨迹和约束**的演化路线。

---

## 参考资料

1. Xu, Liu, Wu, Tong, Li, Ding, Tang, Dong. *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*. arXiv 2023.  
   [arXiv](https://arxiv.org/abs/2304.05977)

2. Black, Janner, Du, Kostrikov, Levine. *Training Diffusion Models with Reinforcement Learning*. arXiv 2023.  
   [arXiv](https://arxiv.org/abs/2305.13301) | [Project](https://rl-diffusion.github.io/)

3. Fan, Watkins, Du, Liu, Ryu, Boutilier, Abbeel, Ghavamzadeh, K. Lee, K. Lee. *DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models*. arXiv 2023.  
   [arXiv](https://arxiv.org/abs/2305.16381)

4. Clark, Vicol, Swersky, Fleet. *Directly Fine-Tuning Diffusion Models on Differentiable Rewards*. arXiv 2023 / ICLR 2024.  
   [arXiv](https://arxiv.org/abs/2309.17400)

5. Prabhudesai, Goyal, Pathak, Fragkiadaki. *Aligning Text-to-Image Diffusion Models with Reward Backpropagation*. arXiv 2023.  
   [arXiv](https://arxiv.org/abs/2310.03739) | [Project](https://align-prop.github.io/)

6. Wallace, Dang, Rafailov, Zhou, Lou, Purushwalkam, Ermon, Xiong, Joty, Naik. *Diffusion Model Alignment Using Direct Preference Optimization*. arXiv 2023.  
   [arXiv](https://arxiv.org/abs/2311.12908)

7. Yang, Tao, Lyu, Ge, Chen, Li, Shen, Zhu, Li. *Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model*. arXiv 2023 / CVPR 2024.  
   [arXiv](https://arxiv.org/abs/2311.13231)

8. Yuan, Zhang, Wang, Wei, Feng, Pan, Zhang, Liu, Albanie, Ni. *InstructVideo: Instructing Video Diffusion Models with Human Feedback*. arXiv 2023 / CVPR 2024.  
   [arXiv](https://arxiv.org/abs/2312.12490)

9. Prabhudesai, Mendonca, Qin, Fragkiadaki, Pathak. *Video Diffusion Alignment via Reward Gradients*. arXiv 2024.  
   [arXiv](https://arxiv.org/abs/2407.08737)

10. Uehara, Zhao, Biancalani, Levine. *Understanding Reinforcement Learning-Based Fine-Tuning of Diffusion Models: A Tutorial and Review*. arXiv 2024.  
    [arXiv](https://arxiv.org/abs/2407.13734)

11. Liu, Liu, Liang, Li, Liu, Wang, Wan, Zhang, Ouyang. *Flow-GRPO: Training Flow Matching Models via Online RL*. NeurIPS 2025.  
    [OpenReview](https://openreview.net/forum?id=oCBKGw5HNf)

12. Ding, Ye. *TreeGRPO: Tree-Advantage GRPO for Online RL Post-Training of Diffusion Models*. ICLR 2026.  
    [OpenReview](https://openreview.net/forum?id=3rZdp4TmUb)

13. Zheng, Chen, Ye, Wang, Zhang, Jiang, Su, Ermon, Zhu, Liu. *DiffusionNFT: Online Diffusion Reinforcement with Forward Process*. ICLR 2026 Oral.  
    [OpenReview](https://openreview.net/forum?id=VJZ477R89F)
