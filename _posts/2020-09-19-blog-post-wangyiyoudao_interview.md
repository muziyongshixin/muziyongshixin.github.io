---
title: '2021秋招-网易有道-面试总结'
date: 2020-09-19
permalink: /posts/2020/09/wangyiyoudao/
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

## 网易有道 视觉算法工程师 面经

部门概况：部门主要是负责网易有道的 ORC 技术，手写和印刷体的识别，作业批改等业务。在北京大概几十人的视觉团队。有机会解决北京户口，工作节奏 10-6-5. 办公地点在网易大厦。

## 一面 （2020 年 9 月 19 日）

- 自我介绍
- 实习工作
- 项目介绍

- 算法题

```python
'''
二维的矩阵中保存着0 和 1 两种值，问最大的实心全1的正方形边长是多少？
'''
def func(matrix):
    n=len(matrix)
    m=len(matrix[0])

    dp=[[0]*m for i in range(n)]
    result=0
    for j in range(m):
        dp[0][j]=1 if matrix[0][j]==1 else 0
        result=max(result,dp[0][j])
    for i in range(1,n):
        dp[i][0]=1 if matrix[i][0]==1 else 0
        result=max(result,dp[i][0])

    for i  in range(1,n):
        for j in range(1,m):
            if matrix[i][j]==1:
                dp[i][j]=min([dp[i-1][j],dp[i][j-1],dp[i-1][j-1]])+1
                result=max(result,dp[i][j])
    return result

if __name__=='__main__':
    t=input()
    n,m=[int(x) for x in t.split()]
    matrix=[]
    for i in range(n):
        t=input()
        row=[int(x) for x in t.split()]
        matrix.append(row)
    result=func(matrix)
    print(result)
```

- 算法题 2

```python
'''
判断一个点是否在一个凸多边形内部


主要思路是通过该点向右做射线，如果和多边形有奇数个交点说明是在内部，如果是有偶数个交点说明是在外部。
'''


def is_in_segment(p,p1,p2): # 判断共线的三个点，p是否在p1,p2组成的线段内
    if p[0]>=min(p1[0],p2[0]) and p[0]<=max(p1[0],p2[0]) and p[1]>=min(p1[1],p2[1]) and p[1]<=max(p1[1],p2[1]):
        return True
    else:
        return False

def get_cross(k1,b1,k2,b2): # 计算两条直线的交点
    if k1==k2:
        if b1==b2:
            return '#','#'
        else:
            return None,None
    if k1=='#':
        x=b1
        y=k2*x + b2
    elif k2=='#':
        x=b2
        y=k1*x+b1
    else:
        x=(b2-b1)/(k1-k2)
        y=(k1*x)+b1
    return x,y


def func1(p,plist):
    a,b=p
    lines=[] # 构造直线参数
    for i in range(len(plist)):
        x1,y1=plist[i-1]
        x2,y2=plist[i]
        if x1==x2:
            k='#'
            b=x1
        else:
            k=(y1-y2)/(x1-x2)
            b=y1-k*(x1)
        lines.append((k,b))

    cnt=set() # 记录过点p向右做的射线 和多边形的交点个数，之所以用set是因为可能射线和多边形的交点在多边形的角上，这样可能会被统计两次。
    for i, (k,b) in enumerate(lines):
        tx,ty=get_cross(0, b, k, b)
        p1=plist[i-1]
        p2=plist[i]
        if tx is not None and ty is not None:
            if tx=='#' and ty=='#' and is_in_segment(p,p1,p2):
                return True
            if tx<a:
                continue
            flag=is_in_segment((tx,ty),p1,p2)
            if flag:
                set.add((tx,ty))
    return True if len(cnt)&1==1 else False

```

## 二面 （2020 年 9 月 23 日）

总体时间比较短大概 30 分钟。

- 自我介绍
- 讲解 CVPR 论文
- 和面试官聊了一下部门的结构（网易有道的 AI 产品部，总体 60-70 人，分为 CV（16 人），NLP，语音三个组。主要负责教育场景下的一些 AI 技术，包括 OCR 前的检测，OCR 识别，直播网课的视频的前背景分离，嵌入式设备比如有道翻译笔，改卷机器等的视觉算法。）

## 三面（2020 年 9 月 27 日）

面试官应该是网易有道的 AI 产品部的总监，主要是项目经历的问答，没有啥很细节的问题。

- 自我介绍
- 按照简历依次问项目经历
- 简单介绍了一篇论文，主要问的是论文选择这个问题的过程以及思考。
- 问面试官关于部门的一些情况。（面试官表示去年大概 80%都解决了北京户口）

## HR 面（2020 年 9 月 29 日）

总体 30 分钟左右，主要是一些个人经历的交流。

- 自我介绍。
- 对实习过的不同公司的感受，优缺点。
- 认为最大的困难和挑战是什么。
- 认为做的最成功的，最有成就感的一件事情是什么。
- 对什么样的事情会比较有兴趣去做？举个例子
- 对工作的期待，薪资，其他的考虑？
- 应届生 offer 没有股票。

## 加面（2020 年 10 月 14 日）

- 感觉应该是整个 AI 部门的负责人面试的。总体时间 1 个小时左右。
- 面试主要内容是介绍了一下 CVPR 的论文，同时针对落地场景进行了一些讨论和探索。
- 同时也讲了一下在阿里和腾讯的实习工作。
- 后面负责人也简单介绍了一下有道这边 AI 技术的一个应用场景，有道主要是负责教育业务，然后 AI 技术的应用主要是分为两个线：
- 第一条线是 教育资源的数字化，比如 OCR 技术，涉及到题目、手写体、单词等的识别和翻译，把纸质数据变成电子化数据。
- 第二条线是 希望用 AI 赋能人的学习，比如实现个性化的教育，让人能够从 AI 的模式中去学习一些知识。

<div data-hk-top-pages="5"> </div>
