---
title: '2021秋招-Pony.AI-面试总结'
date: 2020-09-02
permalink: /posts/2020/09/pony-ai/
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

## 2021 秋招 小马智行 面经

部门概况：根据背景面试官表示可能回去感知组。

## 一面（2020 年 9 月 02 日）

- 自我介绍
- 简单介绍了一下 CVPR 的论文

- 算法题

```python

'''
二维平面上有N个点，每个点用二维坐标表示，找到三个点让他们组成三角形，使得其他所有的点都不在三角形内部，返回这样的三个点的坐标。
'''

###### 想了个Nlog（N）的，先对所有的点按照x轴坐标排序，然后连续的三个点就有可能是满足条件的点。但是需要判断一下多个点是否共线这种情况。
def func(points):
    points.sort(key=lambda x:x[0])
    for i in range(len(points)-2):
        j=i+1
        k=i+2
        xs=[p[0] for p in points[i:i+3]]
        ys=[p[1] for p in points[i:i+3]]
        if len(set(xs))==1:
            continue
        elif len(xs)==2:
            return points[i:i+3]
        if len(set(ys))==1 or (ys[1]-ys[0] )/(xs[1]-xs[0]) == (ys[2]-ys[0])/(xs[2]-xs[0]):
            continue
        else:
            return points[i:i+3]
    return []

# PS:面试官表示还有O(N)的做法，首先选三个不共线的点组成原始的三角形，然后依次扫描其他的点，如果他们是在三角形的外部则直接跳过，否则的话该点替换掉其中的一个点，最后所有的点都判断完毕之后得到的三角形上的三个点也满足题目要求。
```

## 二面（2020 年 9 月 14 日）

- 自我介绍
- 简单介绍了一下在阿里的实习工作
- 算法题

```python
'''
给定一个包含int数据的数组，和一个整数k。 每一次可以对数组中的一个数进行加一操作，最多可以做k次操作，k可以不用完。
希望能够使数组中相同的数字数目最多（也就是众数的数目最大），返回这个最大值

例如：
{1,2,4,4}  k=2 可以操作得到->{1,4,4,4}, 因此结果为3
'''


## 思路是先对数组排序，然后用一个左右指针表示的区间内的数字调整到区间内的最大值
# 如果区间内的和+k >= 区间最大值* 区间数目，则当前区间可以在不多于k次的操作下调整到相同的值，此时更新结果并，左移left指针
# 否则的话左移right指针
# 直到左指针小于0时终止循环。
def func(nums,k):
    nums.sort()

    pre_sum = [nums[0]]
    for i in range(1, len(nums)):
        cur_sum = pre_sum[-1] + nums[i]
        pre_sum.append(cur_sum)

    left = len(nums) - 1
    right = len(nums) - 1
    result = 1
    while left >= 0 and right >= 0:
        if right < result - 1:
            break
        t_sum = pre_sum[right] - pre_sum[left] + nums[left]  # why add nums[right]
        if t_sum + k >= (right - left + 1) * nums[right]:
            result = max(result, (right - left + 1))
            left -= 1
        else:
            right -= 1

    return result
```

- 面试官是感知组的，主要做激光雷达三维点云的，在自动驾驶领域，比如下雨的时候雨会对激光测距造成干扰如何处理是一个比较困难的问题。

## 三面 （2020 年 9 月 14 日）

- 自我介绍
- 介绍在腾讯实习的时候做的工作
- 介绍在阿里实习的时候做的工作
- 算法题

```python
'''
给定一个n*m的二维格点地图, 每个位置要么是字符’.’表示空地, 要么是’@’表示有敌人在这里. 规定给定一个d(1 <= d <= min(m, n)), 如果一个d*d的区域内没有任何敌人, 则认为这片区域是安全的.
问给定的地图中有多少个这样安全的区域.
'''

## 二维前缀和，将原始的地图转换成01矩阵，然后t_sum[i][j] 保存从（0,0） 到 （i，j）的矩形中所有元素的和
## 然后遍历二维矩阵 利用前缀和 计算以(i,j)作为左上角的d*d的区域内的元素和，如果是0，说明是安全区+1
## 总体时间复杂度 O(N*M)

def convert(data): # 转换成01矩阵
    n=len(data)
    m=len(data[0])
    matrix=[[0]*m for i in range(n)]
    for i in range(n):
        for j in range(m):
            if data[i][j]=='@':
                matrix[i][j]=1
    return matrix


def func(matrix,d):
    n=len(matrix)
    m=len(matrix[0])

    t_sum=[[0]*m for i in range(n)]

    for j in range(m):
        t_sum[0][j]=t_sum[0][j-1]+matrix[0][j]

    for i in range(1,n):
        t_sum[i][0]=t_sum[i-1][0]+matrix[i][0]

    for i in range(1,n):
        for j in range(1,m):
            t_sum[i][j]=t_sum[i][j-1]+t_sum[i-1][j]-t_sum[i-1][j-1]+matrix[i][j]

    result=1
    for i in range(n):
        if i+d>=n:
            break
        for j in range(m):
            if j+d>=m:
                break
            ni=i+d
            nj=j+d
            if i==0 and j==0:
               	tmp=t_sum[ni][nj]
            elif i==0:
                tmp=t_sum[ni][nj]-t_sum[ni][j-1]
            elif j==0:
                tmp=t_sum[ni][nj] -t_sum[i-1][nj]
            else:
            	tmp=t_sum[ni][nj]-t_sum[ni][j-1]-t_sum[i-1][nj]+t_sum[i-1][j-1]

            if tmp==0:
                result+=1
    return result
```

## 四面 （2020 年 9 月 24 日）

基础面，确实很多基础知识，很多不会。。。。

- segment fault 什么时候出现

  - [Link](https://blog.csdn.net/qq_22238021/article/details/79872978)

- 操作系统分段和分页有什么区别
- 为什么要分页，分页有哪些优点
- 分页会不会有内存碎片问题
- 进程和线程的区别
- 多线程能够共享进程的什么内部资源
- 多线程怎么调度
- 什么时候用多线程，什么时候用多进程
- 如何实现一个可以扩容的连续数据结构，也就是 c++里的 vector？
- TCP 和 UDP 的区别，什么时候使用 UDP
- HTTP 打开一个网页之后 TCP 连接会断开吗
- 动态链接库和静态链接库的区别
- 算法题如何找到一个数组中第 K 大的数
- SVD 分解解释一下，有哪些应用场景？
- 数学题，掷一个 6 面的骰子 100 次，所以点数的和大于 400 的概率是多少
  - 基于大数定理和中心极限定理
- 数学题，如何在一个单位圆里均匀采样
- 如何用两个栈实现一个队列

## 五面（2020 年 10 月 14 日）

- 这一面主要是和公司相互了解的一个过程。 时间大概 30 分钟。

- 自我介绍
- 介绍在阿里的实习工作。具体的做法，在公司和实验室的区别和感受。
- 对找工作的有啥考虑，业务方向什么的有啥偏好

<div data-hk-top-pages="5"> </div>
