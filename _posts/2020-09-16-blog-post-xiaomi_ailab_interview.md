---
title: '2021秋招-小米AI Lab-面试总结'
date: 2020-09-16
permalink: /posts/2020/09/xiaomi_ailab/
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

## 小米 AI Lab 面经

部门概况：部门是属于小米 AI 实验室下面的二级部门，在北京和武汉都有团队，目前总体人员规模在 100 多人左右。部门主要做的是视觉相关的技术在小米手机上的应用，包括人脸识别，手机拍照的相关算法， 比如自拍虚化，背景换天，夜景增强。 相册内的人脸聚类等应用。

## 一面 （2020 年 9 月 16 日）

面试主要是三个部分：

- 聊了一下各个阶段实习所做的事情。
- 算法题

```python
'''
在行列都排好序的矩阵中找指定的数
限定语言：C、Python、C++、Javascript、Python 3、Java、Go
给定一个N×M的整形矩阵matrix和一个整数K, matrix的每一行和每一列都是排好序的。
实现一个函数，判断K是否在matrix中
[要求]
时间复杂度为O(N+M)，额外空间复杂度为O(1)。
'''
def func(matrix,k):
    n=len(matrix)
    m=len(matrix[0])
    i=n-1
    j=0
    while i>=0 and j<m:
        if matrix[i][j]==k:
            return True
        elif matrix[i][j]<k:
            j+=1
        else:
            i-=1
    return False
```

<div data-hk-top-pages="5"> </div>
