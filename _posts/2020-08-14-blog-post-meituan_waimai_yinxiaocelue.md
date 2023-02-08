---
title: '2021秋招-美团-外卖营销策略组-面试总结'
date: 2020-08-14
permalink: /posts/2020/08/meituan_waimai/
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

## 美团外卖-营销策略组 面经

部门概况：团队主要做美团外卖的外部推荐，用户优惠券包的发放，交易额度提升等策略。部门分成算法和工程两部分。主要业务分为两块，一块是红包或者营销这种金额的确定，以及发放用户的确定，希望能够尽可能高的提升转化率；另外一块是类似于推荐，通过短信或者 APP 的推送通知来给用户推送一些活动信息，吸引用户下单。

<!-- 算法的话大概有 12 个人左右，工程团队七八个人左右。 -->

北斗计划的话需要最少 4 轮技术面试，否则就需要 5 轮或者更多，去年因为北斗的人进的比较多，所以今年可能会更加严格一点。投的视觉岗位说人招满了没有坑位了。。。

## 一面（2020 年 8 月 14 日）

- 自我介绍
- 讲一片论文，针对论文中的一些问题进行提问
- 实际业务场景中如果需要给 1000 万用户发 100 万张金额一样的优惠券，目标是需要让最终的交易总额最大，怎么做。
- 如果优惠券总额是 100 万，不同用户得到的钱可以不一样，优化目标不变，这个该怎么建模。
- 通常我们认为发的优惠券金额越大，该用户成交额也会越大，但是如果根据历史数据训练出来的模型输出的预测结果不是这样的，有什么办法进行优化或者怎么解决这个问题。
- 你通常在遇到一个新的领域的话怎么去学习？
- 有没有用过传统的机器学习，为啥视觉领域基本都用深度学习了，比传统的优点在哪？
- Kaggle 为什么很多人还是用传统的机器学习方法，这个是为什么，你怎么看？
- 你通常在设计这种模型的时候，各种 layer，激活函数的这种设置是怎么样的考虑，或者参考什么
- 为什么选择 CV+NLP 这个方向。

- 算法题

```python
'''
如何高效计算一个完全二叉树的节点个数
'''

def func(root):
    if root is None:
        return 0
    l=get_deepth(root, flag=1)
    r=get_deepth(root, flag=-1)
    if l==r:
        return 2**(l)-1
    else:
        l=func(root.left)
        r=func(root.right)
        return l+r+1

def get_deepth(root,flag):
    if root is None:
        return 0
    cur_deepth=0
    if flag==1:
        cur_deepth= 1+get_deepth(root.left, flag)
    elif flag==-1:
        cur_deepth= 1+get_deepth(root.right, flag)
    return cur_deepth
```

## 二面 (2020 年 8 月 18 日)

- 自我介绍
- 介绍两篇论文
- 平时训练深度学习的网络怎么调试，如果模型不收敛怎么办
  - 看 loss 曲线，看梯度信息。
  - 减小 lr，如果是没有梯度信息二分法找 bug。
- 模型参数初始化怎么设置
  - 一般直接使用默认初始化，因为 BN 能够一定程度上解决这种问题
- 为什么 BN 能够解决这种问题
  - 因为 BN 能够让每层的输入分布变得比较合理，所以参数的分布能够在一两轮迭代后自动调整好
- BN 的那两个参数的作用是什么，那两个参数又让数据分布偏移了那怎么办。
  - 保留 feature 的特征表达能力，通过梯度回传自动调整到合适的值。
- 复杂模型一般怎么设计。

- 如果业务场景需要根据不同的用户的喜好和风格给他推送的广告信息图片需要进行修改和重新生成这个怎么做。

- 两道简单算法题

```python
#coding=utf-8
import sys
'''
找到一个给定数组中第二大的数
'''
def func(nums):
    if len(nums)<2:
        return None
    v1= -sys.maxsize
    v2= -sys.maxsize
    for n in nums:
        v2=max(v2,n)
        v1,v2=max(v1,v2),min(v1,v2)
    return v2

'''
返回一个数组的最小连续字串和
'''
def func1(nums):
    if len(nums)==0:
        return None
    min_val=sys.maxsize
    pre_min=sys.maxsize
    for n in nums:
        cur_min=min(pre_min+n,n)
        min_val=min(min_val,cur_min)
        pre_min=cur_min
    return min_val

```

- 反问面试官 业务中目前遇到的最大的问题是啥
  - 建模问题，不同的用户如何确定发放的金额的多少，同一个用户所在位置的不同能够供应的商家也会影响到用户的下单率。同一个用户在不同时间看到营销信息所被吸引的概率也不同。对于短信营销的场景，因为存在成本问题，如何确保短信发送到真正需要的用户身上。
  - 目前主要还是 NLP 进行营销文案的生成，对使用 CV 技术对广告图片等生成还没做。


## 三面 （2020年8月20日）
三面也是介绍了一下论文然后和面试官讨论了一些开放性的问题。

- 比如如何计算两个文档的一个相似度？
  - Blue-Score，n-gram的这种方式？或者bert encoding，之后算embedding的相似度。

- 那如何在一个极大的数据库中搜索是否有小修改的重复的文章？
  - 说了个基于词频统计模式的初筛，类似于视觉里的直方图的模式，然后做初筛，后面再用ngram这种计算相似度。面试官表示这依然要O(N)的搜整个库，效率太低。
  - 后来了解到可以有KD-Tree这种数据结构，可以以近似log(N)的复杂度做大规模的向量相似度计算（但是貌似是不精确的）
  - 或者可以先对文章按照tag或者类别进行分桶，或者根据embedding 进行group，然后来了新的query先和不同group的中心计算，选择几个最相近的中心所对应的group里的文章再计算相似度。
  - 感觉大规模的检索或者相似度计算的问题，通常就是hash分桶或者聚类，或者使用树结构进行分层查询。
  - 额外的一个延展，如果是完全相同的文章怎么判断是否在库中存在呢，可以对文章做hash？

- 其他的话就是关于业务的一些讨论，比如目前业务上的困难以及未来需要提升的方向等。


## 四面（2020年8月25日）
四面的面试官好像也是让介绍了一下论文，然后是对目前实习工作的一个讨论。
其他好像没有问什么很细节的问题。




<div data-hk-top-pages="5"> </div>
