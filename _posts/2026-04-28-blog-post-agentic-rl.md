---
title: 'Agentic RL 训练全景：环境、信号、分布与系统的协同闭环'
date: 2026-04-28
permalink: /posts/2026/04/agentic-rl/
categories:
  - blog
tags:
  - reinforcement learning
  - agentic
  - llm
  - post-training
  - infrastructure
toc: true
---

> 过去一年，各家大模型公司公开的技术报告透出的最重要信号，不是又出现了一个更好的 PPO/GRPO 变体，而是**真正有效的 Agentic RL 已经从"单轮文本优化"转向了"在长上下文、工具调用、部分可观测、异步执行环境中的系统性策略学习"**。
>
> Kimi K1.5[1] 把长上下文 RL、partial rollout 重用和 mirror-descent 风格的 policy optimization 拉到了台前；Kimi K2[2]/K2.5[3] 又把 agentic 数据合成、多模态 RL、token-level clipping、GRM rubric、Toggle、PARL / Agent Swarm 这些关键部件公开；MiniMax 把另一个事实讲得更彻底：**当 rollout 时长从秒级扩到小时级，训练瓶颈就不再是 loss design，而是吞吐、稳定性与 agent 灵活性之间的三难权衡**；GLM 则强调分阶段 RL：Reasoning RL、Agentic RL、General RL 不是混在一起一次训完，而是通过顺序化 pipeline 逐步推进，并借助异步 RL 基础设施与跨阶段蒸馏来兼顾长时程 agent 学习与能力保持。
>
> Agentic RL 的核心问题，已经从"怎么更新参数"扩展为"**怎么在真实 Agent 环境里持续制造可用的学习信号，并用在线交互的轨迹数据驱动优化**"。

---

## 一、为什么 Agentic RL 与传统 RLHF / RLVR 不同

Agentic RL 的训练对象不再是"给定一个 prompt，输出一个答案"的单轮文本映射，而是**一个在环境中交互的策略**。这个策略要处理：状态更新、工具调用、外部观察、上下文整理、子任务委派、终止条件判断，以及成本 / 时延 / 安全约束。换句话说，agentic RL 更像是在做一类带有长时间尺度、部分可观测性和结构化动作空间的策略学习，而不是简单地对文本续写概率做后验重排。

这直接带来四个训练上的变化：

1. **状态不再只由用户输入决定**：它由历史轨迹、工具返回、环境回馈、记忆摘要和当前上下文共同构成。
2. **动作也不再只是下一个 token**：它可能是"选哪个工具、填什么参数、要不要压缩上下文、是否并行分派子任务"。
3. **奖励更延迟、更稀疏、更复合**：既要看结果对不对，也要看过程是否准确、是否高效、是否节省 token 和单位时间有效训练效率。
4. **Rollout 时间高度不均匀**：同步训练代价高，异步训练又引入分布偏移。

因此，agentic RL 的本质**不是把 GRPO / PPO 套到更长的输出上**，而是把环境、奖励、采样、调度、缓存、优化器和评测接到同一个闭环里。

---

## 二、理解 Agentic RL 的三个不变量

如果把 Agentic RL 理解成一个"在真实环境里持续交互、持续采样、持续更新"的策略学习系统，那么真正重要的就不再是"这一步用哪种 RL 算法"，而是**训练闭环能否长期守住三个更底层的条件**。

这里的"不变量"不是指某个量在数学上严格恒定，而是指它们虽然会天然漂移，却必须在整个训练过程中被不断拉回到一个仍然可学习、可优化的区间里。前两个是**不应跌破的下限**，第三个是**不应越过的上限**。

### 1）第一不变量：策略的可探索空间不能过早塌缩

第一不变量**不是**要求输出更随机、token 熵更高，而是：**模型在给定状态下，仍然保有一组彼此可区分、语义上不同、并且真实可行的行为路径**。

对 Agentic RL 来说，这个探索空间不只是"不同措辞"，而是：

- 不同的**任务分解**方式
- 不同的**工具调用顺序**
- 不同的**记忆读写**策略
- 不同的**上下文整理**方式
- 不同的**停止条件**与**自我修正**路径

它之所以会塌缩，是因为训练天然会把概率质量压向"少数当前最占优的模式"。只要训练目标主要奖励"更短、更像标准流程、更容易被 verifier 识别"的行为，模型就会把其他原本也可能成功的路径边缘化。在 agent 场景下，这种压缩比单轮问答更严重——工具接口、scaffold、上下文模板和终止逻辑本身就会**暗中偏好**某类固定 workflow。

**保持这一不变量的意义**：它决定了后续 RL 是否还有真正的搜索空间。RL 的价值不是把已知最好答案重复推高概率，而是让模型在交互中持续发现"此前还没被放大的高回报行为"。如果可探索空间已经提前塌缩，后面的采样大多只是对同一种套路做表面扰动，reward spread 越来越小，训练看似还在继续，实际上只是在一个已经缩水的空间里做局部扰动。

### 2）第二不变量：学习信号必须持续非退化

即使模型仍然保有多种可行路径，这些路径也不一定会被**学到**。参数更新依赖的不是"存在别的可能性"，而是**不同轨迹之间的差异能否稳定地转成非零、方向明确、尺度合理的梯度**。

Agentic RL 的奖励结构天然容易让信号塌缩：真实任务奖励延迟、结果稀疏、过程很长，最终常常只有成败标签、粗粒度 rubric，或少数高层质量分。于是同一组采样很容易出现两种退化情形——

- **简单任务几乎全对**（模型已在该局部饱和）
- **困难任务几乎全错**（模型尚未进入可学习区域）

但对梯度而言，这两类样本都会导向同一个结果：**组内没有足够差异，优势接近消失，更新方向随之退化**。再叠加长轨迹的信用分配、部分可观测性带来的归因模糊、工具噪声和 verifier 噪声对比较关系的污染——系统表面上在大量收集交互数据，实际上却在不断生产"不可学样本"。

**这里有一个关键观察**：学习信号的质量，不取决于奖励项有多少，而取决于**比较是否可学**。奖励可以很复杂，但如果它无法在模型当前边界附近稳定区分"略好"与"略差"的轨迹，它仍然会产生退化梯度。反过来，一个看上去更简单的反馈，只要能持续打开轨迹间的有效差异，也能成为高质量学习信号。

**第二不变量真正要求保持不变的，不是奖励总量，而是可比较性与可更新性。**

### 3）第三不变量：训练 / 更新 / 部署三者的分布偏移必须可控

前两个不变量解决"还有没有别的路径"和"这些路径能不能变成梯度"，第三个不变量解决**这些梯度是不是作用在了正确的分布上**。

在 Agentic RL 中，有三个天然不一致的分布：

- 策略模型**采样**出的 rollout 分布
- learner 真正**拿来更新**的样本分布
- 最终**部署执行**的策略分布

Agent 训练会持续制造分布漂移：

- 轨迹长短差异极大，严格同步的 on-policy 不现实，异步采样、缓存、续跑、复用、过滤都会让"生成样本时的策略"和"更新参数时的策略"发生时间错位；
- Agent 状态由工具返回、环境反馈、上下文裁剪、记忆摘要、调度决策共同构成，只要其中任何一层在 rollout / training / serving 三阶段的表示不完全一致，模型学到的可能就不是同一个动作语义；
- 训练和部署脚手架常常并不完全相同：解码设置、context packing、tool schema、tokenizer/engine、middleware、日志序列化方式都会改变模型真正面对的决策问题。

结果是：被优化的不再是一个干净统一的策略分布，而是**多个相似但不相同的分布拼接而成的近似对象**。

对长轨迹 Agent，这一点尤其致命——轨迹越长，前面每一点小的偏移都会沿着后续状态转移不断累积，最终把策略推向"在训练里看起来合理、在真实环境里却不可执行"的方向。

**Agentic RL 里的分布偏移，不只是外部环境变化带来的，它在很大程度上是系统自己制造出来的**。这也是为什么第三不变量不是单纯的算法修正问题，而是一个系统级的一致性问题。

### 4）为什么这三个不变量要放在一起理解

它们不是彼此独立的三条要素，而是**同一个训练系统的三个耦合边界**：

| 不变量 | 本质问题 | 失守后果 |
|---|---|---|
| 第一 | 策略空间是否还够宽 | 没有可探索的新路径 |
| 第二 | 空间里的差异能否转成有效梯度 | 有路径但学不到 |
| 第三 | 梯度是否作用在正确分布上 | 学到的行为在部署时失真 |

- 只有**探索**没有**信号**：训练变成高噪声试错；
- 只有**信号**没有**探索**：训练迅速收缩到狭窄局部最优；
- 探索和信号都有，但**分布偏移失控**：学到的也不是部署时真正需要的行为。

它们彼此之间还天然存在张力：探索更强，会让比较更稀、分布偏移更难控；过度追求稳定更新，又容易压平探索空间；为了制造更锋利的信号把 verifier 设计得过于严格，又会让模型朝少数投机模式收缩。

**Agentic RL 真正要解决的，不是把某个 loss 降得更低，而是在一个持续变化、持续异步、持续与外部环境交互的系统里，始终把探索、信号和分布维持在同一个可学习区间内。**

---

## 三、Agentic RL 的九个关键维度

三个不变量是"要守住什么"；下面九个维度是"在哪些具体位置守"。前八个维度对应训练系统的核心环节，第九个维度（评测与可观测性）回答的是一个更基础的问题——**如果你连三个不变量是否正在被守住都测不出来，就根本谈不上管理它们**。

### 1. 环境与接口建模：先搞清楚"环境允许 Agent 做什么"，再谈"正确答案是什么"

Agentic RL 和普通"对一道题生成一个答案"的最大差别在于：模型不再只是从 prompt 里猜一个 completion，而是在一个**可交互、可执行、带状态转移**的世界里学 policy。

决定 Agentic RL 训练效果的**第一个变量**不是 reward model，而是环境和接口本身是否设计清楚：

- 每一步模型能看到哪些信息？
- 能采取哪些动作、哪些工具调用是允许且有效的？
- 任务在什么条件下结束、成功如何判断？
- 训练时使用的工具接口和交互流程，是否和真实部署一致？

当前几家的共识非常一致：

- Kimi K2 把大规模 agentic 数据合成 + 真实/合成环境 RL 放进后训练主线；
- K2.5 把 Agentic RL 统一到 **Gym-like 接口**，并支持大规模异步任务管理；
- GLM-5[8] 把 agentic RL 扩展到**超过 10K 个可验证的软件工程环境、terminal 环境和多跳搜索任务**；
- Forge[9] 强调系统跨越了十万级 real-world scaffolds 与数千种工具调用格式。

真正的 agentic capability 不是从静态数据里背下来的，而是从**结构化、可验证、可迁移的环境**里训练出来的。

环境建模的核心，不是把现实世界完整模拟出来，而是把真实工作转写成一个**结构上不失真的可训练决策过程**——重要的不是表面真实，而是 **structural fidelity**：动作空间、关键信息流、失败模式和成功判据，是否与真实部署保持一致。举一个典型例子：一个客服 agent 不必复现公司所有噪声，但必须保留库存状态、退款规则、权限边界、上下文记忆、工具接口、升级流程和最终评分 rubric；否则学到的只是"像在做客服"，而不是"真的能做客服"。

**环境覆盖度，是 Agentic RL 的第一条 scaling axis**。但真实任务的难点往往不在 data scaling，而在 **specification scaling**：很多高价值任务之所以难进训练闭环，不是因为模型不够聪明，而是因为任务没有被写成机器可执行、机器可验证的规范。下一代 env scaling 更像三个"编译器"问题：

- **task compiler**：把模糊请求编译成初始状态、工具、约束和终止条件；
- **verifier compiler**：把"做得好不好"编译成可执行检查、rubric 和必要时的人类审阅；
- **scaffold compiler**：把同一能力放进不同 agent scaffold、tool schema 和 orchestration loop，避免模型只记住单一 workflow。

Forge 强调跨大规模 scaffold 训练，本质上就在处理第三个问题。真实人类任务里最大的问题不是"任务太少"，而是 **evaluator 太弱**——一旦 verifier 失真，模型就会学会 hacking，而不是学会工作。SWE-Universe[10] 把环境构建、self-verification 和 hacking detection 自动化，说明大家已经开始把"防投机评测"当成环境的一部分。

### 2. 探索能力与多样性保持：不是把 temperature 调高，而是维护可探索行为的空间

很多人一谈"探索"就想到：调高 temperature、多采几个 rollout、加 entropy regularization。但对 agentic RL，这些都只是表层现象。核心问题是：**模型在训练的不同阶段，是否仍然保有一组彼此可区分、都可能成功、且在参数空间里真实可达的行为路径**。

对 reasoning 模型，这个问题已经被直接观察到：随着 SFT 推进，Pass@1 可以继续上升，但 **Pass@k 会快速恶化**，而且后续 RL 往往也恢复不了；仅靠 token-level 的多样化解码，距离理论上的 oracle 上界仍有明显差距。**真正塌缩的不是采样温度，而是模型权重层面的行为可探索空间。**

所以这一节最本质的思想是：**探索本质上是一个 support management 问题**。你要管理的不是 token 级噪声，而是模型是否还保有：

- 多种合法任务分解
- 多种工具调用顺序
- 多种上下文组织方式
- 多种长度的 reasoning path
- 在 agent 场景下的多种 memory / planning / action 组合

只要这些分支在参数里还活着，后续 RL 才有可能通过 verifier 和 rollout 把它们放大；一旦在进入 RL 前就被压没，训练再稳定也只是在缩水的空间里做局部优化。

**预训练 / 基座阶段**决定的是 **reachable support**——模型是否已经具备足够多的技能碎片、长上下文耐受性、工具使用先验和任务分解能力：

- MiniMax-M1[11] 把额外 7.5T continual pretraining 直接称为 "*Foundation for RL Scaling*"；
- Kimi K2 用 diverse agents、tool combinations 和 rubric-guided tasks，把未来 agent 可能探索的 action space 和 task space 提前做宽；
- DeepSeek-R1-Zero[12] 提供了另一个很有代表性的例子：它在没有 SFT 冷启动的前提下直接 RL，模型会自然增加思考时长，并逐步长出更长推理和自我修正的行为——这说明对能力足够强的基础模型，**RL 过程本身就可能激发并放大更长程的推理与自我修正行为**。

**冷启动 / SFT 阶段**真正要解决的，不仅是"把模型教得更会答题"，而是**不要在进入 RL 之前就把分布压塌**：

- GEM[4] 的重要性不在于又提出一个新的 SFT loss，而在于它把问题说透了：标准交叉熵 SFT 会压缩输出分布，抹掉很多 alternative plausible outputs，而在线 RL 恰恰需要这些行为分歧来形成探索空间；
- Getting Your LLMs Ready for RL[13] 进一步指出：最适合接 RL 的 checkpoint，**往往不是 validation 上表现最好的那个**——在传统过拟合发生之前，模型就可能已经出现 distributional forgetting，过度偏离 base distribution，从而损害后续 RL 的潜力。

**到了在线 RL 阶段**，探索问题又会表现成另一种形态：即便模型内部还保留着多种路径，如果 RL 目标只盯 correctness，训练仍然会把概率质量持续推向少数高回报模式：

- DAPO[14] 把 Clip-Higher 明确写成 "*promotes diversity and avoids entropy collapse*"；
- Diversity-Aware Policy Optimization[15] 在 12 个 LLM 上给出更强的经验结论：solution diversity 与 Potential@k 存在强正相关，因此**在 RL 目标中显式促进 token-level diversity**，平均带来 3.5% 的数学推理提升。

这里真正重要的不是某一个技巧，而是一个更深的转向：**探索，第一次从"训练自然会保住的东西"，变成了需要被显式优化的对象**。

这一维度今天仍有几个未解决的问题：

1. 当前很多方法管理的仍然是 token entropy 或字符串级 diversity，但 agentic RL 真正需要保住的是**语义层和策略层的多样性**——不同工具顺序、不同 memory 操作、不同任务分解不一定表现为更高的 token entropy；
2. 很多系统的 verifier 偏 outcome-only，天然低估那些"短期看更绕、长期却更有价值"的探索路径；
3. 社区仍过度依赖 Pass@1，而对 Pass@k、Potential@k、解法簇数量、跨 scaffold 迁移这些更接近探索前沿的指标重视不够。

### 3. 算力分配与学习信号整理：谁拿到 rollout，谁才真正有机会被学到

上一节讨论"多样化的采样路径是否存在"，这一节讨论**在固定 rollout 预算下，这些路径里哪些会真正进入梯度**。探索解决**可达性**，算力分配解决**可学习性**。

对 reasoning / agentic RL 来说，模型内部也许还保留着多种策略，但如果 rollout 总是平均分给"已经学会的简单题"和"暂时完全学不会的极难题"，训练既看不到组内差异，也形成不了有效梯度——在稀疏奖励和 group baseline 设置下，很多 prompt-group 会退化成全 0 或全 1，**advantage energy 为 0，gate 关闭**，这些组消耗了算力却没有产生 usable learning signal。

因此，真正该优化的目标不只是平均 reward，而是更接近训练动力学本身的量：

- non-zero gradient ratio
- gate-open probability
- 组内 reward spread
- 单位训练时间内的有效样本率

**算力分配是 credit assignment 的上游机制**：谁拿到更多 rollout，谁就更有机会被比较、被区分、被学到。

主流做法可以分成三类——

**① 方差控制视角**。既然不同 prompt 对梯度方差的贡献不同，那么 rollout 预算就应该优先投给那些最能减少估计方差、最可能恢复学习信号的 prompt：

- GVM-RAFT[17] 从 acceptance rate 和 gradient noise 的角度做动态分配；
- VIP[18] 更系统，用轻量高斯过程预测 prompt 成功概率，再转成 gradient variance 估计，并在固定预算约束下解一个 rollout allocation 优化问题。VIP 明确把目标写成 *minimize the expected gradient variance of the policy update*，而不是机械拉高 pass rate。

这标志着 rollout allocation 开始从经验 heuristics 变成 **policy optimization 的一部分**。

**② 学习价值—成本权衡视角**。Knapsack RL[6] 把每个任务的探索看成"具有不同 value 和 cost 的 item"，由此推出自适应资源分配规则——把预算从已经学饱和的题转移到更可能产出信号的题。预算分配不是为了省钱，而是**避免把大量算力烧在注定不会更新参数的地方**。

**③ 主动恢复信号视角**。Reinforce-Ada[19] 认为很多"所谓难 prompt"没法学，其实是 undersampling 造成的统计假象，而不是模型真没潜力。于是它不再用固定小组、统一采样被动等待 mixed outcomes，而是**根据 prompt 难度动态增加推理预算**，主动去找出那些本来会被 uniform GRPO 漏掉的信号。

这个话题还有不少未解问题：

- 现有 allocator 主要依赖 pass rate、variance proxy 或近期 rollout 统计，但这些量不等于**长期训练价值**——一个 prompt 今天方差大，不代表明天最值得更多预算；
- 现有方法仍把单条 prompt 作为分配单位，但 agentic RL 的训练难度更多取决于**交互结构和执行状态**（scaffold、工具链、历史记忆、任务阶段），而不只是 prompt 文本；
- 大多数分配器优化的是局部训练效率，还没有把预算分配、reward 结构、hinting、off-policy freshness、长时程 credit assignment **联合起来**。

下一步真正值得做的是：把 semantic difficulty、uncertainty、verifier sharpness、历史 learning gain、scaffold transfer 价值、甚至 hinting 后的 gate-open probability，**一起纳入 allocation policy**；把 prompt-level allocation 推广到 trajectory segment、tool-call branch、memory operation 这类更细的 agent 单位。到那时，算力分配才会真正从"更高效的训练技巧"变成 **agentic RL 的核心算法层**。

### 4. 目标函数与策略优化：不先问"用哪种 RL"，先问"现在到底坏在哪"

这一部分重点不是 PPO、GRPO、REINFORCE 的技术细节，而是**Agentic RL 的优化器究竟在控制什么**。更本质地说，它在回答两个问题：

1. 高回报轨迹要以多大力度被推回当前策略？
2. rollout 分布、learner 更新分布、deployment 执行动作之间允许多大偏移？

这里有一条**常被忽略的基本事实**：PPO 那一整套 value network machinery 未必是必要的。ReMax[5] 提醒我们，在文本生成这种快仿真、近似确定性转移、轨迹级奖励的设定下，REINFORCE 路线也可以既简单又稳定。Kimi K1.5 则把长 CoT RL 明确写成 **relative-entropy regularized 的 online mirror descent** 问题。

到了 K2.5、MiniMax-M1 和 GLM-5，问题进一步从"如何估 advantage"转成"如何控制长轨迹、异步 rollout、训练 / 推理 mismatch 下的 off-policy drift"，于是出现了这些看起来很细但实际上很关键的设计：

- **K2.5 的 token-level clipping**：处理 train-inference framework 差异放大的 off-policy divergence；
- **M1 的 CISPO**：裁 importance weights 而不是裁 token updates，在保留更多 token 级梯度的同时控制比值爆炸；
- **GLM-5 的 TITO + 双边重要性采样**：确保被优化的动作尽可能还是当时真正被采样的动作。

**未来真正有价值的优化研究，不是继续修改 PPO 或 GRPO 的公式**，而是先诊断：训练当前究竟受限于哪一类瓶颈——

- 梯度噪声过大？
- 策略漂移过快？
- 训练目标与真实任务不匹配？

只有先定位清楚，才能决定是改进优势估计、采样方式、更新约束，还是训练调度策略。

### 5. Rollout 采样、异步并行与调度：调度策略本身就是算法的一部分

在真实 agent 场景，理想化的同步 on-policy RL 很难被满足：不同 rollout 完成时间差异极大，短的几秒，长的可能几十分钟甚至更久。**坚持严格同步会被 straggler 拖死；完全贪心异步又把训练拖入过重的 off-policy 偏移**。

各家给出的折中方案非常有代表性：

- **Kimi K1.5 的 partial rollout**：长轨迹切段，未完成部分进 replay buffer，下一轮继续，只有当前段要求 on-policy；
- **K2.5**：每个 agent task 都当作独立异步 coroutine，通过专门的 Rollout Manager 支持高并发；
- **MiniMax 的 Windowed FIFO**：在"严格 FIFO（稳但慢）"和"完全异步（快但漂移大）"之间做折中——不要求全局严格排队，只在有限窗口内保持大致顺序，让窗口里的已完成任务可以灵活先训练；
- **GLM-5**：直接把采样和训练分开，一边持续并行生成轨迹，另一边独立消费数据，再用 TITO + 双边重要性采样 + 陈旧样本过滤来控制异步训练中不可避免的 off-policy 偏移。

**很多人把 queueing、resume、tail-latency、staleness 当成工程问题，但在 agentic RL 里，调度实际上会改写训练分布**。K1.5 的 partial rollout 意味着一条长轨迹由新旧段拼接而成；MiniMax 的 Windowed FIFO 直接控制了"允许新鲜样本先于更早提交的样本进入训练"的程度；GLM-5 的异步 Agent RL 更是明确承认"不现实去追踪所有历史行为策略，必须在可接受的偏差内做近似校正"。

Agentic RL 的核心不是"如何保持纯 on-policy"，而是**如何在不可避免的异步与陈旧性下，让偏移保持在仍然有学习价值的范围内**。这就是为什么 rollout system 不是承载算法的底座——**它本身就是算法的一部分**。

### 6. 奖励、验证器与效率约束：Reward 定义的不只是"答对"，而是"怎样工作才算好"

很多关于 agentic RL 的讨论会说"verifier 就够了"。这对真实 Agent 任务其实不成立：agent 的成功不只体现在 final correctness 上，还体现在动作是否合理、工具调用是否合适、是否浪费上下文、是否无意义过度思考、是否拖慢总完成时间、以及输出是否符合更高层的质量和交互要求。

几家的具体做法非常有参考价值：

- **K2.5**：可验证任务用 rule-based outcome reward，token 成本用 budget-control reward，开放任务用多 rubric GRM，并通过 Toggle 在"尽量做对"和"尽量省 token"之间交替优化；
- **MiniMax-M1**：verifiable 与 unverifiable 任务分开处理，用 GenRM 处理不能靠规则验证的任务，并**特别讨论了长 CoT 下 GenRM 的 length bias**——奖励模型偏好更长但未必更好的回答，会直接诱发 reward hacking；
- **GLM-5**：把 rule-based reward、ORM、GRM 组合成 hybrid reward system，并明确写出三者权衡——规则奖励精确但窄，ORM 低方差但容易被 exploit，GRM 更灵活但方差更高；
- **Forge**：进一步把中间过程质量和**任务完成时间**都纳入 agent RL——真实用户需要的不是"最终做对但过程低效、等待很久"的系统，而是"既能做对、又能较快完成"的 agent。

**对 reward 正确的理解是"工作方式的规范化"，而不只是"答案质量的评分器"**：

- K2.5 用多个 GRM rubric，是因为单一偏好信号太容易被过拟合；
- M1 专门处理 GenRM 长度偏置，是因为 reward model 一旦系统性偏向 verbose response，整个 RL 就会被带偏；
- Forge 引入完成时间相关奖励，是因为真实部署中 agent 的效用不只由正确率决定，还取决于实际耗时。

Reward design 的关键**不是给模型更多分数**，而是把 correctness、quality、efficiency、robustness 拆开，再决定哪些可以硬验证、哪些要用模型判断、哪些必须通过对抗测试和 OOD transfer 来防止被投机。

### 7. 记忆、层级与并行 Agent：被训练的对象已经不只是 Token Policy，而是 Operating Policy

很多人一谈 long-context agent 就想"把 context window 做大一点"。但**长上下文不等于记忆，更不等于好的 agent**。核心问题是：当交互历史越来越长、工具观察越来越多时，模型如何决定什么该保留、什么该丢弃、什么该压缩、什么时候拆任务、什么时候并行多个子 agent？

- **MiniMax Forge 的 Context Rot**：即使没有触到绝对 context window 上限，长轮次交互中累积的中间推理和冗余 observation 也会造成 attention dilution，让模型失焦。于是 Forge 直接把 **Context Management 纳入 RL 交互回路**，把它当作一种显式 action，让 context transition 本身成为环境状态转移的一部分；
- **GLM-5** 在搜索 agent 上也观察到极长上下文会明显伤害性能，因此使用 **keep-recent-k 与 discard-all 的层级式 context management**；
- **K2.5 的 Agent Swarm 与 PARL**：当单 agent 顺序执行的延迟变得不可接受时，让 orchestrator 学会**动态任务分解、子 agent 创建和并行调度**。训练时只更新 orchestrator、冻结 sub-agent，以规避最难的 credit ambiguity 与训练不稳定。

**被优化的对象已经从"token 级生成策略"扩展成"操作系统级策略"**——模型不再只决定下一个 token，而是在决定：

- 算力怎么花
- 上下文怎么管
- 任务怎么拆
- 子 agent 怎么协作

K2.5 的一个关键 insight：**真正的并行 agent 不是把同一个模型复制几份并发运行**，而是让 orchestrator 学会"什么时候值得并行、如何分配子任务、如何在最终汇总时保持全局一致性"。Forge 则强调：记忆管理如果只在 inference 端手工加规则、训练时没见过这种状态转移，最终会形成严重的 inference-training mismatch。

未来 agentic RL 的 frontier，未必是让模型"再想更久"，而是**把 memory editing、hierarchical decomposition 和 agent orchestration 一起纳入训练目标**。

### 8. Infra 基础设施：它不是承载算法的底座，而是在塑造训练分布

如果说 RLHF 是在一个相对规整的 prompt → completion → reward → update 闭环上做优化，那么 Agentic RL 面对的是**长短极不均匀、工具调用密集、环境反馈异步、动作语义复杂**的真实交互轨迹。

在这种设定下，基础设施直接决定：

- rollout 以什么顺序完成
- 哪些样本因过时被丢弃
- 哪些前缀能够复用
- 训练端和推理端看到的是否还是同一个动作空间

这里有三层 infra：

**① 塑造训练分布的 rollout / learner 基础设施**。由于任务完成时间可能从秒级跨到小时级，同步 on-policy 几乎不可能，系统必须处理 actor–learner 解耦、队列调度、buffer freshness、checkpoint staleness、partial rollout reuse、stale sample filtering。MiniMax Forge 把 strict FIFO / greedy async / Windowed FIFO 的权衡直接写成"吞吐与分布稳定之间的核心矛盾"；GLM-5 通过异步 generation-training 解耦 + TITO + double-sided importance sampling 控制偏移；K1.5 的 partial rollout reuse 说明**长轨迹能否被复用，本身就是训练 recipe 的一部分**。这一层 infra 直接塑造了"模型真正看到的训练分布"。

**② 提升吞吐与成本效率的规模化训练 / 推理 infra**。包括训练 / 推理解耦、数据池缓存、KV / prefix 复用、动态 batching、各种并行化和异构资源调度策略。它们解决的核心问题不是"单点算法是否成立"，而是"这些方法能否在现实成本下真正跑到足够规模"。对 agent workload 来说，模型生成、环境执行、工具调用、verifier 计算、日志存储的资源瓶颈完全不同，基础设施**必须是解耦和分层的**，不能继续沿用单一、同步、同构的训练范式。

**③ 保证数值一致性和训练—推理一致性的 serving infra**。最容易被低估但其实最关键：Agentic RL 优化的不是抽象文本，而是**具有明确执行语义的动作序列**——训练时、采样时、部署时对动作的表示或接口稍有错位，模型学到的策略就可能在上线时部分失效。GLM-5 的 TITO 之所以重要，不只是为了省一次 re-tokenization，而是为了精确保持 sampled action 与 optimized action 的对应；MiniMax Forge 的 gateway 与 middleware 设计本质上也在做 action interface standardization。因此，tokenizer / engine 对齐、tool schema 标准化、trajectory serialization、metadata logging、train-serving alignment——都不再只是工程细节，而是在决定**训练时被优化的动作，是否真的是部署时会执行的那个动作**。

### 9. 评测与可观测性：测不出来的不变量，就守不住

前面八节讲了"在哪里守不变量"，但有一个被大多数文章忽视的基础问题：**如果你连三个不变量是否正在被守住都测不出来，就根本无从管理它们**。

Agentic RL 的 evaluation 不能只看 Pass@1 或 final reward，至少需要三类互补的观测维度：

**① 探索健康度（对应第一不变量）**：

- Pass@k、Potential@k、解法簇数量（semantic cluster count）
- 行为路径的 scaffold 迁移率（同一能力在不同 scaffold 下的成功率）
- 长期 entropy trajectory 与 action-level 多样性（而不仅是 token-level）

**② 学习信号健康度（对应第二不变量）**：

- **non-zero advantage ratio**：一个 batch 内多少 group 产生了非零梯度
- **gate-open probability**：group-based 方法中 advantage 有效的样本比例
- **组内 reward spread** 和 **gradient SNR**
- **单位训练时间的有效样本率**（effective tokens per GPU-hour）

这些量往往比 loss 曲线更能解释"为什么训练看起来还在跑，但能力没长"。

**③ 分布一致性（对应第三不变量）**：

- **training–serving KL**：相同 prompt 下训练 checkpoint 与部署 checkpoint 的输出分布差异
- **rollout staleness 分布**：样本被生成时的策略与被学习时的策略相隔多少步
- **tokenizer / tool schema mismatch 率**：训练端与部署端接口一致性的硬指标
- **长轨迹误差累积曲线**：模型表现随交互步数退化的速度

在更高层次上，还需要一套**对抗性评测**：verifier-hacking 检测、reward-model OOD 探针、scaffold 替换测试、工具噪声注入测试——这些不是"锦上添花的 benchmark"，而是**第一不变量和第二不变量是否被守住的直接证据**。

SWE-Universe 把 hacking detection 自动化进环境，本质上就是在承认：**评测已经不是 pipeline 的末端，而是训练系统的一部分**。没有这层观测，所谓"调参"就只是在黑箱里做随机扰动。

---

## 四、结语：Agentic RL 的真正竞争，不在单点算法

回到开头那句话——**Agentic RL 的核心问题，已经从"怎么更新参数"扩展为"怎么在真实 Agent 环境里持续制造可用的学习信号"**。

把三条技术路线放在一起看，信号非常清楚：

- **Kimi 路线**告诉我们：(1) 长上下文本身是一条 RL scaling axis；(2) 复杂的 value function / MCTS / process RM 不是唯一道路，简洁但分布一致的 policy optimization 也能跑出很强的长链能力；(3) 当 agent 工作流变复杂后，奖励模型、token-level clipping、token efficiency 控制和 learned parallel orchestration 会越来越重要。K1.5 → K2.5 的演进，本质上是从"把长 reasoning RL 跑通"走向"把多步 agentic / multimodal RL 规模化"；
- **MiniMax 路线**说明：长时程 agent RL 一进到真实环境，首要问题很快就从"模型能不能推理"转向"**系统能不能稳定地持续学习**"。M1 的 CISPO 的价值在于修复长轨迹 RL 的 off-policy 和梯度裁剪副作用；Forge 进一步证明，异步调度、上下文管理、完成时间奖励、跨任务联合训练、前缀树合并这类"看起来很工程"的东西，**实际上决定了你最终能否在大规模真实环境里把 RL 跑起来**；
- **GLM 路线**强调：后训练不应该一股脑混在一起，而要按能力类型分阶段组织，并借助蒸馏机制保护已有能力。Reasoning RL → Agentic RL → General RL 的顺序**不只是训练日程安排，而是一种能力编排方式**。GLM-5 对异步 RL 基础设施、TITO、double-sided importance sampling 的强调，也再次说明：**训练系统与策略优化之间已经没有清晰边界**。

综合这些路线，一个清晰的结论是：

> Agentic RL 不只是"更大模型 × 更多数据 × 更多 token"，而是：
>
> - 更丰富的**环境覆盖**
> - 更高密度的**有效学习信号**
> - 更一致的 **rollout / update / serving 分布**
> - 更高的**单位时间有效训练效率**
> - 以及能让你**确认前四者正在发生**的评测与可观测性

在完善高效的 infra 支持下，谁在这五个维度上同时做得更好，谁就更可能真正把 agent 训出来。

---

## 参考文献

[1] Kimi Team. *Kimi k1.5: Scaling Reinforcement Learning with LLMs*. arXiv:2501.12599, 2025. [https://arxiv.org/abs/2501.12599](https://arxiv.org/abs/2501.12599)

[2] Kimi Team. *Kimi K2: Open Agentic Intelligence*. arXiv:2507.20534, 2025. [https://arxiv.org/abs/2507.20534](https://arxiv.org/abs/2507.20534)

[3] Kimi Team. *Kimi K2.5: Visual Agentic Intelligence*. arXiv:2602.02276, 2026. [https://arxiv.org/abs/2602.02276](https://arxiv.org/abs/2602.02276)

[4] Ziniu Li et al. *Preserving Diversity in Supervised Fine-Tuning of Large Language Models*. arXiv:2408.16673, 2024. [https://arxiv.org/abs/2408.16673](https://arxiv.org/abs/2408.16673)

[5] Ziniu Li et al. *ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models*. arXiv:2310.10505, 2023. [https://arxiv.org/abs/2310.10505](https://arxiv.org/abs/2310.10505)

[6] Ziniu Li et al. *Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation*. arXiv:2509.25849, 2025. [https://arxiv.org/abs/2509.25849](https://arxiv.org/abs/2509.25849)

[7] Hanze Dong. *Curate the Learning Signal for Reinforcement Learning: Variance Minimization, Adaptive Sampling, and Self-Hinting*. Blog post, 2026. [https://hendrydong.github.io/blogs/pages/rl-ada.html](https://hendrydong.github.io/blogs/pages/rl-ada.html)

[8] GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering*. arXiv:2602.15763, 2026. [https://arxiv.org/abs/2602.15763](https://arxiv.org/abs/2602.15763)

[9] MiniMax. *Forge: Scalable Agent RL Framework and Algorithm*. MiniMax News, 2026. [https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)

[10] Mouxiang Chen et al. *SWE-Universe: Scale Real-World Verifiable Environments to Millions*. arXiv:2602.02361, 2026. [https://arxiv.org/abs/2602.02361](https://arxiv.org/abs/2602.02361)

[11] MiniMax. *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*. arXiv:2506.13585, 2025. [https://arxiv.org/abs/2506.13585](https://arxiv.org/abs/2506.13585)

[12] DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948, 2025. [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)

[13] Xinran Li et al. *Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT*. OpenReview / ICLR 2026. [https://openreview.net/forum?id=yezWGJmODg](https://openreview.net/forum?id=yezWGJmODg)

[14] Qiyuan Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv:2503.14476, 2025. [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)

[15] Jian Yao et al. *Diversity-Aware Policy Optimization for Large Language Model Reasoning*. arXiv:2505.23433, 2025. [https://arxiv.org/abs/2505.23433](https://arxiv.org/abs/2505.23433)

[16] Xingyu Dang et al. *Assessing Diversity Collapse in Reasoning*. OpenReview, 2025. [https://openreview.net/forum?id=AMiKsHLjQh](https://openreview.net/forum?id=AMiKsHLjQh)

[17] Jiarui Yao et al. *Optimizing Chain-of-Thought Reasoners via Gradient Variance Minimization in Rejection Sampling and RL*. arXiv:2505.02391, 2025. [https://arxiv.org/abs/2505.02391](https://arxiv.org/abs/2505.02391)

[18] Hieu Trung Nguyen et al. *Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards*. arXiv:2602.01601, 2026. [https://arxiv.org/abs/2602.01601](https://arxiv.org/abs/2602.01601)

[19] Wei Xiong et al. *Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training*. arXiv:2510.04996, 2025. [https://arxiv.org/abs/2510.04996](https://arxiv.org/abs/2510.04996)
