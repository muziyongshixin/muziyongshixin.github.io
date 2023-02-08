---
title: '2021秋招-京东数科-面试总结'
date: 2020-09-17
permalink: /posts/2020/09/jingdong_shuke/
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

## 京东数科 面经

部门概况：大部门是做京东数科的商业变现的（类似于阿里妈妈），小部门是准备搭建京东数科的技术部的算法中台。主要业务分为两块一个是先上营销（京东商城的广告推荐，京东金融的理财产品推荐等），一块是线下的营销（比如线下商场的互动大屏，人流量统计、交互游戏、行人关注度等场景的算法搭建）。目前部门处于初步组建的阶段，正式员工 3 人，还有 3 人在入职途中。办公地点在亦庄。

## 一面 （2020 年 9 月 17 日）

- 自我介绍。
- 简单介绍了下实习工作和一篇论文。
- 算法题

  - 给定一个包含 z 个不同字符的字典，每次可以有放回的随机抽一个字符，总共抽 m 次组成一个字符串。 这么操作两次得到两个长度为 m 的字符串 X 和 Y。 问判断 X 和 Y 是否相同所需要比较的 次数的期望是多少？

  - 直观的解法是如果两个字符串相同的话一定要比较 m 次，则这个情况的比较期望$E_1= \frac{m}{z^m}$
  - 如果两个字符串不同的话，设第 k 个字符不同，则说明前 k-1 个字符相同，且当前这个不同这种时候的期望是 $E_{2,k}= k * \frac{1}{z^{k-1}} * (1-\frac{1}{z}) $。
  - 两个字符串不相同的总体期望 $E_2= \sum_{k=1}^{m} E_{2,k}$
  - 整体期望为 $E= E_1+ E_2$

---

- 还有个比较直观的 dp 思路：
- 设 E(m)为长度为 m 的字符串的期望，则第一个一定要比较，第一个相同的情况下才需要比较后面的，所以状态转移方程是：$E(m) = 1+ \frac{1}{z}E(m-1)$， 然后 $E(1)=1$。

## 二面

- 二面应该是交叉面，面试时间大概 35 分钟。
- 简单介绍了一下在腾讯实习的工作
- 算法题

```python
'''
给定一个单链表，请实现随机返回其中一个结点的函数。 需要保证每个节点被选取的概率是相同的。

解法就是蓄水池抽样问题
'''
import random
m=1
def func(head):
    p=head
    cache=[]
    cnt=1
    while p is not None:
        if len(cache)>=10:
            tp=random.randint(0,cnt)
            if tp<m:
                i=random.randint(0,m-1)
                cache[i]=p.val
        else:
            cache.append(p.val)
        p=p.next
        cnt+=1
    idx=random.randint(0,len(cache)-1)
    return cache[idx]
```

蓄水池抽样问题的详解：[Link](https://www.jianshu.com/p/a4ab07f5752e)

<div data-hk-top-pages="5"> </div>
