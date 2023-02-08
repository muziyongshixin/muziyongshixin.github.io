---
title: '2021秋招提前批-快手多媒体内容理解部门-面试总结'
date: 2020-07-30
permalink: /posts/2020/07/kuaishou_mmu/
categories:
  - Interview
tags:
  - Interview
toc: true
---

---

---

<div>
<div class="button01">
            <visited_a href="#" display:inline><span id="busuanzi_container_site_pv">你是第<span id="busuanzi_value_site_pv"></span>位访客~</span></visited_a>

      <visited_p class="top">٩(๑^o^๑)۶</visited_p>
      <visited_p class="bottom">Σ(っ °Д °;)っ被你发现了！</visited_p>
</div>
<img align="center" width="100" src="{{ site.url }}/images/static/take_me.gif" alt="" display:inline>
</div>
---

## 快手 多媒体内容理解部门 面经

部门概况： 部门主要是做多媒体的内容理解，包括图像，文本，语音，搜索推荐等多个业务方向。大部门总体的人员大概有 100 多人。

## 一面 （2020 年 7 月 30 日）

面试主要是三个部分：项目相关，基础机器学习的知识，算法题

- 自我介绍，主要是介绍了一下自己做的论文，聊了一下当前实习所做的事情。
- 传统机器学习的问题，你对传统机器学习有哪些了解
- 逻辑回归和 XGBoost 有什么异同？

  - 都是做分类的模型，逻辑回归的话是线性模型，通过对特征的线性组合然后通过 sigmoid 激活函数来预测一个 0-1 之间的概率值
  - XGboost 是树模型，通过 Boosting 策略不断提升。XGboost还可以拿来做回归问题。

- 逻辑回归的 Loss 函数

  - 交叉熵损失函数

- 交叉熵函数和极大似然估计的关系

  - 交叉熵对应着最小化 KL 散度，最小化 KL 散度等价于最大似然估计
  - [详解](https://zhuanlan.zhihu.com/p/37917476)

- L1 和 L2 的 regularization 有什么作用

  - L2 的正则化导致参数都是较小的值，能够一定程度上起到抑制过拟合的效果
  - L1 的正则化能够起到一定的特征选择的作用，会导致稀疏解

- L1 和 L2 的正则化对先验数据的分布假设分别是什么样子？

  - L1 是拉普拉斯先验，而 L2 是高斯先验。
  - [详解](https://www.cnblogs.com/USTC-ZCC/p/10123610.html)

- LSTM 和 ResNet 有什么相同的点

  - 他们都有一个直通的路径方便梯度的回传以及特征信息的通路。

- LSTM 到 文本一维的 CNN 卷积 再到 Bert 这一步步的发展有一个什么样的趋势

  - 单词和单词之间的交互从 O(N) 到 O(log(N)) 到 O(1)，能够更加有效的挖掘 word 之间的相关性，挖掘 context 信息。

- Bert 训练的时候有什么缺点

  - 训练的过程中的输入是有 Mask 的，但是做下游任务的时候他的输入是没 mask 的，这会导致数据的分布不同
  - Bert 的最长输入默认是 512，再长的输入就不支持了。

- 算法题一

```python
'''
给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
nums = [1, 3, 5, -5, 6, -9, 20, 17]
'''
def func(nums):
    dp=[[0]*2 for i in range(len(nums))]
    result=nums[0]
    pre_max=nums[0]
    pre_min=nums[0]
    for i in range(1,len(nums)):
        cur=nums[i]
        if cur>=0:
            pre_max=max(pre_max*cur,cur)
            pre_min=min(pre_min*cur,cur)
        else:
            pre_max=max(pre_min*cur,cur)
            pre_min=min(pre_max*cur,cur)
        result=max([result,pre_min,pre_max])
    return result

if  __name__=='__main__':
    nums=[int(x) for x in input().split(',')]
    result=func(nums)
    print(result)

```

- 算法题 2

```python
'''
给你一个整数数组 nums ，请你找出数组中的位置k，使得数组前k个元素的方差和后n-k个元素的方差之和最小。
'''

def func(nums):
    n=len(nums)
    total_sq_sum=sum([x**2 for x in nums])
    total_sum=sum(nums)

    result=100000000000000
    pre_sq_sum=0
    pre_sum=0
    for i in range(n):
        pre_sum+=nums[i]
        pre_sq_sum+=nums[i]**2
        left=pre_sq_sum/(i+1) - (pre_sum/(i+1))**2

        tail_sq_sum=total_sq_sum-pre_sq_sum
        tail_sum=total_sum-pre_sum
        right=tail_sq_sum/(n-i-1) - (tail_sum/(n-i-1))**2
        result=min(result,left+right)
    return result
```

## 二面 （2020 年 7 月 30 日）

二面主要是开放性的系统设计题，加上自己一片论文的介绍。

- 你对搜索引擎有没有什么了解？
- 如果叫你设计一个视频搜索的系统怎么设计，用 query 文本去搜索短视频
- 目前的系统主要还是基于传统的搜索引擎的策略，基于文本和 tag 进行搜索和排序。但是一些用户会弄一些虚假的 tag 或者视频描述来恶意使自己的视频搜索排名较高，如何通过视觉的特征解决这个问题。
- 针对一些口语化的 query 如何进行建模
- 如果叫你自己去构建一个跨模态的语义匹配的数据集，你可以怎么构建，你可以去爬取任何你想要的数据
  - 可以使用 rephrase 的策略对文本数据进行扩充，可以使用 image retrieval 的策略进行图片数据的扩充。
  - 面试官说 百度他们会用 query 文本使用传统的搜索引擎技术返回的 top 视频作为和 query 文本匹配的数据进行训练。我感觉有点类似于 AQE
- 如果是在搜索的时候，通常会包含粗排和精排的过程，那么在粗排的时候对 query 进行 encoding 是使用轻模型好还是重模型好？
  - 单纯的对 query 进行 encoding 的话可以使用重模型，在使用 encoding 得到的 embedding 进行大规模数据库相似度计算和排序的时候应该使用轻模型。

## 三面 （2020 年 8 月 11 日）

三面总体感觉是在闲聊，没有考察很细节的东西，40 分钟左右就结束了。

- 自我介绍
- 介绍了一下视频隐写的现状和效果
- 讲解了一下 CVPR 的论文
- 谈论了一下论文的技术可以在哪些场景的应用
- 讨论了一下工业界的工作和学术工作的区别，以及工业界的通常面临的一些问题以及解决方法。
- 个人的未来工作的兴趣爱好，是希望偏技术中台一点，还是希望偏业务一点。
- 反问部门概况，户口等情况。
  - 面试官表示只有少量户口。部门在快速扩张，目前大概快 200 人，之后可能每年扩张 50-100 人左右。

## HR 面（2020 年 8 月 15 日）

HR 面试大概 30 分钟左右，主要是自我介绍，之前实习的做的不同的事情以及对不同公司的感受，平时怎么学习了解资讯的，未来两三年的规划，对 offer 有什么考虑。认为自己有什么缺点。三个词评价一下自己。有没有用过快手。有没有其他公司的 offer。

HR 表示如果有幸能够被评估上快 Star 的话，一周左右的时间会有进一步的联系，后面还会有两轮面试，应该是部门的总监面以及面试委员会的面试。
去年全公司的大概 20 个左右能够保证户口，剩下可能还会有几十个快 star 能够有差不多的薪资，但是无法保证户口，可能要看后面的工作表现。
工作节奏的话，MMU 中大概有七八个小的团队，不同的团队也不太一样，有的 10-10-5，有的 996 主要看业务的紧急程度。周末加班的话双倍工资。

---

<div data-hk-top-pages="5"> </div>
