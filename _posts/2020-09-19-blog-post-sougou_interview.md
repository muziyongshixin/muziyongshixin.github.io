---
title: '2021秋招-搜狗-面试总结'
date: 2020-09-19
permalink: /posts/2020/09/sougou/
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

## 搜狗 智能图像识别-图像算法研究员 面经

部门概况：部门主要是负责输入法以及多模态融合相关业务，具体面试官好像没怎么说。视觉团队大概 30-40 人。
大视觉团队包含三个大的方向：感知组(人脸、手势识别，分身)，OCR 组（印刷体，图表等文字识别，录音笔、词典笔、同传等场景），手写识别组（输入法里的手写输入，公式）
正常 3 轮技术面+HR 面。有机会解决北京户口，搜狗要被腾讯收购了，后期可能拿到的是腾讯 PCG 的 offer。

## 一面 （2020 年 9 月 19 日）

- 自我介绍
- 两篇论文的介绍
- 项目介绍
- RNN 的求导
  - [Link](https://blog.csdn.net/weixin_41923961/article/details/87885322)
- SENet
- Faster-RCNN 的结构，RPN 的详细流程
- Cascade-RCNN
- ROI-Align 是在哪里提出来的
- 对 Mask-RCNN 有了解吗
- Attention 机制怎么实现
- RNN 到 LSTM 到 Transformer 的发展脉络
- LSTM 里的激活函数为啥不用 ReLU
- Resnet 的基础结构，残差连接的作用，从梯度计算上说明
- Feature map 大小跟 kernel，stride padding 的关系

## 二面（2020 年 9 月 23 日）

- 介绍了一下腾讯的实习经历
- 基础知识
- ROC 曲线和 PR 曲线的区别
- ROC 曲线的横纵坐标分别是啥？
  - 横坐标是 False Positive Rate，纵坐标是 True Positive Rate。
- 负样本明显数倍于正样本的时候的分类问题通常使用哪种评价指标，为什么？

  - 用 ROC 曲线，因为 PR 曲线存在不稳定问题

- 进程和线程的区别
- 死锁的概念
- LRU 的原理和作用
- RNN 中使用的是什么激活函数？
- LSTM 中为什么不使用 ReLU 作为激活函数？

  - 实际上 RNN 和 LSTM 中有人使用 ReLu 作为激活函数，而且貌似效果还不错。

- 算法题

```python
'''
链式A+B
限定语言：Python、C++、C#、Java
有两个用链表表示的整数，每个结点包含一个数位。这些数位是反向存放的，也就是个位排在链表的首部。编写函数对这两个整数求和，并用链表形式返回结果。

给定两个链表ListNode* A，ListNode* B，请返回A+B的结果(ListNode*)。

测试样例：
{1,2,3},{3,2,1}
返回：{4,4,4}
'''
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Plus:
    def plusAB(self, a, b):
        # write code here
        if a is None:
            return b

        if b is None:
            return a
        head=ListNode(0)
        p=a
        q=b
        k=head
        t=0
        while p is not None and q is not None:
            cur=p.val+q.val+t
            t=cur//10
            cur-=t*10
            cur_node=ListNode(cur)
            k.next=cur_node
            k=k.next
            p=p.next
            q=q.next

        if q is not None:
            p=q
        while p is not None:
            cur=p.val+t
            t=cur//10
            cur-=t*10
            cur_node=ListNode(cur)
            k.next=cur_node
            k=k.next
            p=p.next
        if t==1:
            cur_node=ListNode(1)
            k.next=cur_node

        return head.next
```

算法题 2

```python
'''
顺时针打印矩阵
限定语言：Python、C++、C#、Java
对于一个矩阵，请设计一个算法从左上角(mat[0][0])开始，顺时针打印矩阵元素。

给定int矩阵mat,以及它的维数nxm，请返回一个数组，数组中的元素为矩阵元素的顺时针输出。

测试样例：
[[1,2],[3,4]],2,2
返回：[1,2,4,3]
'''
# -*- coding:utf-8 -*-
class Printer:
    def clockwisePrint(self, mat, n, m):
        # write code here
        directions=[(0,1),(1,0),(0,-1),(-1,0)]
        idx=0
        visited=[[0]*m for i in range(n)]
        cnt=0
        x,y=0,0
        result=[]
        while cnt<n*m:
            result.append(mat[x][y])
            visited[x][y]=1
            dx,dy=directions[idx]
            nx=x+dx
            ny=y+dy
            if not (nx>=0 and nx<n and ny>=0 and ny<m and visited[nx][ny]==0):
                idx=(idx+1)%4
                dx,dy=directions[idx]
                nx=x+dx
                ny=y+dy
            x=nx
            y=ny
            cnt+=1
        return result
```

## 三面 （2020年10月23日）

面试官应该是组里的大leader，聊了接近2个小时，主要是聊了一些实习的经历，以及每片文章的一个主要的contribution之类的。聊得问题比较宽泛，没有具体到很细节的问题上。
面试官表示整个公司可能就只有十几个户口指标，后续的话是否有户口需要看面试排名，暂时无法确定。

## HR面（2020年10月28日）

HR面主要聊的是项目经历以及做了什么事情，总体30分钟左右。

<div data-hk-top-pages="5"> </div>
