---
title: '2021秋招-NVIDIA-AI工程师-面试总结'
date: 2020-09-02
permalink: /posts/2020/09/nvidia_interview/
categories:
  - Interview
tags:
  - Interview
toc: true
---

---

---


<script async src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>

<div>
<div class="button01">
            <visited_a href="#" display:inline><span id="busuanzi_container_site_pv">你是第<span id="busuanzi_value_site_pv"></span>位访客~</span></visited_a>

      <visited_p class="top">٩(๑^o^๑)۶</visited_p>
      <visited_p class="bottom">Σ(っ °Д °;)っ被你发现了！</visited_p>
</div>
<img align="center" width="100" src="{{ site.url }}/images/static/take_me.gif" alt="" display:inline>
</div>
---

## NVIDIA AI 工程师面试总结

部门概况：主要是负责 ToB 业务，帮助客户优化模型，提高模型训练速度，一些最新的模型的实现，可能会涉及到一些 CUDA 的编程，算子的实现等。部门分布在北京、上海、深圳。不加班 5 点下班，偶尔可能会需要和国外的团队合作，基本不需要早起和美国开会。面试官说顺利的话可能会有 4-6 轮的面试。没有 KPI，晋升和奖励主要靠主管打分，然后无法解决北京户口。

工作主要是项目制，帮助 BAT 这样的大客户优化他们的模型速度，包括 deep learning 的 training 或者 inference 阶段，以及一些非机器学习的任务在 GPU 上的性能。涉及到一些模型实现的优化，包括底层的 cuda 或者 op 的实现，算法上有模型剪枝、蒸馏、量化、混合精度训练。应用级会给客户提供一些解决方案或者预研的工作。可能会有出差，部门目前国内总共 30 多人，全球估计 70 人左右，北京 12 人左右。

## 一面（2020 年 7 月 20 日）

一面主要面的是都是一些基础的问题，深度学习相关的 trick 和经验什么的。

- 深度学习训练的时候超参数，learning rate，参数初始化，batchsize 等如何设置，以及对模型最终收敛结果的影响，为什么？

  - batchsize 一般是按照显存大小尽可能大设置，然后将显存填满。batchsize 较大的话收敛会更快，但是也不是越大越好，大的 batchsize 更可能会收敛到一个不太好的局部最优解。batchsize 较小的话收敛会更慢，因为梯度的随机性更强，更有可能会产生震荡等情况，但是可能能够收敛到更好的 local optimal。
  - learning rate 主要跟 optimizer 有关，通常 sgd 初始 0.01，Adam 初始 1e-3，特定任务例如 finetune bert 等模型需要设置 1e-5。 在训练的过程中通常是随着训练的过程会逐步减小 lr，一开始较大是为了快速收敛，后面变小是为了在最优解附近防止震荡从而得到一个较好的结果。 例如训练 image net 的时候通常每 30 个 epoch 将 lr 减小 10 倍。 但是也有一些使用 lr scheduler 的操作，比如 training loss 多少个 epoch 下降小于阈值的时候就减小 lr。 也有一些任务会使用 warm up 或者 cosine lr 策略，主要想法是在陷入局部最优解的时候通过较大的 lr 跳出 local optimal 从而可以收敛到更好的效果。
  - 参数初始化主要是使用框架自带的默认的初始化，可能对于某些任务特定的初始化策略会有效，了解到有一些类似于 kaiming_normal,这种初始化策略但是了解的不多。

- 对 batch Normalization 的计算过程，训练和 inference 的时候的策略，优点缺点，为什么？

  - 计算过程：$y= \gamma \frac{x-E(x)}{\sqrt{var[x]}} + \beta$
  - 训练的时候 $E(x), var[x]$都是根据每个 batch 的 feature 及时计算的，$\gamma , \beta$根据梯度进行学习和调整
  - Inference 的时候$E(x) , var[x]$是全 epoch 的 feature 的均值和方差，$\gamma, \beta$是最后一次迭代后的值
  - BN 的优点：
  - 保证每一层的输入的分布偏差不会太大，能够一定程度增加训练的稳定性，加快训练速度和收敛速度。
  - 在使用 sigmoid 等激活函数的时候能够一定程度缓解和消除梯度消失的问题，同时能够加深网络深度。
  - 在一定程度上能够防止过拟合，BN 每次的 mini-batch 的数据都不一样，但是每次的 mini-batch 的数据都会对 moving mean 和 moving variance 产生作用，可以认为是引入了噪声，这就可以认为是进行了 data augmentation，而 data augmentation 被认为是防止过拟合的一种方法。因此，可以认为用 BN 可以防止过拟合。

- 介绍了一个最近做的项目，在其中担任的主要职责
- 模型的性能优化有哪些策略

  - 剪枝，量化，知识蒸馏的[介绍和差别](https://blog.csdn.net/Fannie_Peng/article/details/106397222)
  - 剪枝可以丢掉一些对结果影响较小的 kernel 等同于说减少一些 channel，fc 层的话可以丢掉一些不太重要的权重，然后对剪枝后的模型进行 fine-tune。具体的策略有哪些，不太清楚，貌似有用强化学习做这个的。

- NMS 如何进行计算优化，假设你有无穷的计算资源如何设计并行的线程进行计算。
  - 假设有 N 的 box，先对其按照 confidence score 进行排序分别编号 1-n
  - 然后可以使用 N^2 个线程计算两两 box 之家的 IOU 分数，计算的时候可以判断 j 是否大于 i，如果不是直接返回 0，否则话计算 IOU 并根据阈值进行判断是否抑制的，进而填充到一个 N\*N 的二维矩阵中，M[i][j]=0 表示 i 对 j 不抑制，M[i][j]=1 表示 i 对 j 进行抑制
  - 结果设置为一个一维的长度为 N 的向量初始化为全 0 即都保留，然后依次对矩阵中的每一行进行判断如果当前行对应的 box 在结果向量中已经值为 1 直接跳过，否则将当前行中为 1 的 box 编号在结果向量中也置为 1。
  - 最终的结果向量中 0 对应的 box 保留，1 对应的 box 抑制
  - 在第一步排序的过程中也可以使用多个线程先做局部的排序，然后再使用归并排序，在 N 较大的时候应该有提升。

## 2、3、4、5、6 面（2020 年 09 月 02 日）

一天时间一个下午连续面了 5 个小时，其中还有一位韩国小哥作为面试官英文面试。。。

- 对传统的机器学习有多少了解
- Resnet 的主要贡献是啥？

  - residual Connection，梯度的回传更加顺畅，避免了模型过深之后的梯度消失的问题。[Link](https://blog.csdn.net/shijing_0214/article/details/78475372)

- 缓解模型过拟合的方法有哪些？

  - 数据增强，BN，drop out，lr warm-up，early stop, 参数正则化。

- 为什么 drop out 能够缓解过拟合问题？

  - 类似 Ensemble Learning 或者 bagging 的策略
  - 让模型能够依据部分特征就能得到结果，提升模型的鲁棒性

- drop out 在底层如何实现，在 inference 的时候需要对结果做哪些操作？

  - 底层可以对每个神经元的输出设置 一个 0-1 mask，mask 乘上神经元的输出作为结果。
  - 在 inference 的时候所有的神经元都会被使用，每个神经元的输出需要乘上概率 P。避免所有的结果都是原始的然后相加起来更大。

- 为什么 bagging 策略能够提升模型的性能？

  - 减小模型的方差。

- Bagging 和 Boosting 的区别？

  - [Link](https://blog.csdn.net/qq_36523839/article/details/82490802)

- 为什么说偏差和方差是个 trade off？

  - [Link](https://blog.csdn.net/qq_30490125/article/details/52401773)

- 有哪些 Loss Function

- python 中 tuple 和 list 有啥区别？

- python 中函数对参数 x 进行修改之后，在底层是发生了什么？

- 为什么分类问题使用交叉熵的 Loss，而不是使用 MSE loss？

  - 因为 MSE 针对分类问题的时候他是一个非凸优化问题，而对于交叉熵来说是一个凸优化的问题。
  
- MSE和Cross Entropy Loss的梯度的计算。
  - 令$sig= \frac{1}{1-e^{wx+b}}$
  - $MSELoss= (y-sig)^2$
  - $\frac{\partial{MSE}}{\partial{w}} = -2(y-sig)*\frac{\partial{sig}}{\partial{w}} = -2 (y-sig)*sig*(1-sig)*x $
  - 在使用MSE作为分类的loss的时候，无论sigmoid函数输出是接近0还是接近1，loss都会很小，不利于训练。
  - $CrossEntropyLoss= -ylog(sig) - (1-y)log(1-sig)$
  - $\frac{\partial{CE}}{\partial{W}}=(\frac{-y}{sig} + \frac{(1-y)}{1-sig})* \frac{\partial{sig}}{\partial{W}} = \frac{sig(1-y)- y(1-sig)}{sig(1-sig)}* sig*(1-sig)*x =(sig- sig*y -y + sig*y)*x  =(sig-y) * x $

- 极大似然估计讲一下，交叉熵 Loss 和极大似然估计的关系。

  - [极大似然估计](https://zhuanlan.zhihu.com/p/26614750)
  - [交叉熵和最大似然估计](https://www.jianshu.com/p/191c029ad369)

- 讲解一下 SGD，Momentum，Adam 的公式，以及他们的优缺点

- Adam 的缺点是啥？

  - 可能优化的最优点不如 SGD
  - 在训练的时候消耗更多的内存，大概 SGD 的 3 倍

- Transformer 模型介绍一下，self-attention 机制为什么好？

- Transformer 的 encoder 和 decoder 计算的时候是否都可以并行计算，训练和测试的时候有啥不同？

  - encoder 和 decoder 在训练的时候都是可以直接并行计算的。
  - decoder 在测试的时候只能像 RNN 一样一步一步算，但是训练的时候是可以并行计算的。

- 画一下 RNN 的示意图，写一下 RNN 的公式，为什么大家不用 RNN 了而是使用 transformer 的结构。

- CNN 网络中的那些层可以用矩阵乘法实现。

  - 卷积层和 mean pooling 层。 feature map 拉成二维矩阵，卷积核拉成向量。计算完 resize 回去。mean pooling 可以类似卷积，只是参数固定都是 1/k。

- 精度问题，浮点数 9.9e9 + 1e-9 最后的结果是多少？float32 的结构了解吗？

  - 结果是 9.9e9，小的数会被吞掉。

- 对应到使用 fp16 进行混合精度训练上，为了避免这种大数吞小数的问题，哪些步骤不能使用 fp16？

  - 参数利用梯度更新的时候，对于参数需要保存一份 fp32 的结果，避免小的梯度加不上去。

- 数学题给定一个向量 <a,b>，另外有一个点 c，从 c 向 ab 做垂线，计算垂点 d 坐标。

  - 连接 ac 得到向量<a,c>，计算两个向量的夹角$cos \theta = \frac{<a,b> <a,c>}{\|ab\| \|ac\|}$
  - 然后可以计算出\|ad\|长度，然后根据相似三角形原理可以根据 a，c 坐标计算 d 的坐标。

- 有两个 N\*N 的矩阵 A B 需要进行矩阵乘法，然后有四台机器，每台机器都只有 N\*N 的内存,但是他们可以互相通信，怎么放置数据并设计计算流程。 如果每台机器只有 7/8 N\*N 的内存怎么办。

  - 大致思路是每台机器放置 1/4 行 和 1/4 列的数据，然后对列数据进行计算然后依次交换，最后将结果整合。

- 写出 MLP 的公式，并计算他的梯度并更新参数。
  - 先计算出梯度g，然后w_t= w_(t-1) - a*g

- 全数据集作为batch，mini-batch，单个样本的SGD 分别有什么优点和缺点？
  - batch越大收敛越快，但是大到一定程度更可能收敛到比较sharp的局部最优解，导致泛化能力下降。同时消耗更多的内存。
  - 单个样本的SGD的梯度随机性太大，可能会导致来回震荡，收敛慢，甚至无法收敛。
  - mini SGD是比较合理且常用的策略。
  - [Link](https://www.zhihu.com/question/32673260)

- 数学问题：

```matlab
S = Q * L * Q^T
其中L是对角矩阵，例如：

L=l0 0 0 0
  0 l1 0 0
  0 0 l2 0
  0 0 0 l3

如果将L变换成L1,Q矩阵如何调整可以满足S的值不变

其中L1的形式如下:
L1= l3 0 0 0
    0 l1 0 0
    0 0 l0 0
    0 0 0 l2

答案：
对Q进行按列调整，原先的列的序号调整成 【3 1 0 2】

原因是 S_xy= \sum L_j * Q_xj * Q_yj，每个点的坐标只跟j有关。

```

- C++问题，C 语言的二级指针主要用在什么情况下，C++中的重载和重写有啥区别？

---

<div data-hk-top-pages="5"> </div>
