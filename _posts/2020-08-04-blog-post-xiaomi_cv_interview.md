---
title: '2021秋招提前批-小米未来星-视觉算法研究员-面试总结'
date: 2020-08-04
permalink: /posts/2020/08/xiaomi/
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

## 小米未来星-视觉算法研究员 面经

部门概况：部门是属于小米 AI 实验室下面的二级部门，在北京和武汉都有团队，目前总体人员规模在 100 多人左右。部门主要做的是视觉相关的技术在小米手机上的应用，包括人脸识别，手机拍照的相关算法， 比如自拍虚化，背景换天，夜景增强。 相册内的人脸聚类等应用。

## 一面 （2020 年 8 月 04 日）

面试主要是三个部分：项目相关，算法题

- 聊了一下当前实习所做的事情。
- 以及之前做的项目的内容。
- 讲了一篇论文。
- 传统的 style transfer 的做法有哪些
- 算法题

```python
'''
输入一张二维的灰度图，然后逐个pixel用sobel算子计算灰度图像的梯度
sobel算子如下：
dx=[
[-1, 0, 1],
[-2, 0, 2],
[-1, 0, 1]
]
dy=[
[-1, -2, -1],
[0, 0, 0],
[1, 2, 1]
]
为了简化运算，最后的梯度输出为
grad=|dx|+|dy|
'''
import numpy as np

def sobel(img,dx,dy):
    n,m=img.shape

    padding=np.zeros((n+2,m+2)) # padding with zero
    padding[1:n+1,1:m+1]=img
    img=padding

    x_cache=np.zeros(img.shape)
    y_cache=np.zeros(img.shape)

    dx=np.reshape(1,-1)
    dy=np.reshape(1,-1)

    for i in range(1,n+1):
        for j in range(1,m+1):
            ''' for循环版本
            tx=0
            ty=0
            for x in range(-1,2):
                for y in range(-1,2):
                    cur_i=i+x
                    cur_j=j+y
                    if cur_i>=0 and cur_i<n and cur_j>=0 and cur_j<m:
                        tx+=img[cur_i][cur_j] * dx[x+1][y+1]
                        ty+=img[cur_i][cur_j] * dy[x+1][y+1]
            '''
            data=img[i:i+3,j:j+3].reshape(-1,1)
            tx=np.matmul(dx,data).squeeze()
            ty=np.matmul(dy,data).squeeze()
            x_cache[i][j]=tx
            y_cache[i][j]=ty
    return np.abs(x_cache)+np.abs(y_cache)

```

## 二面 （2020 年 8 月 4 日）

二面面了一个半小时，面试官会问很多项目上的细节

- 当前实习的工作，以及实现的方法，考虑

- 霍夫变换直线检测的原理。
  - 只依稀记得是转换到极坐标系下然后进行投票
  - [霍夫变换直线检测](https://blog.csdn.net/leonardohaig/article/details/87907462)

* 多条线段怎么判断是否组成一个矩形。

* 讲了一下论文

* 会不会安卓开发

* SLAM 系统的位姿矩阵如何和 OpenGL 的相机投影矩阵转换

* 深度学习抑制过拟合的方法有哪些

  - dropout，BN，regularization，early stop， data augmentation，lr warming up。

* BN 的作用，BN 在 inference 的时候均值和方差是怎么用的？

* 如何实现 pytorch 里的 data loader？多进程还是多线程实现？

* 进程和线程的区别

* python 为啥多线程无法利用多核计算资源？

  - python 解释器的全局锁问题

* python 多线程和多进程一般开到多少？

* 线程间的通信怎么做

* 多线程的读写怎么保证安全性，为什么要注意线程的安全性，java 里的线程信号量怎么实现。

* 进程的内存分配，堆？栈？

* 算法题

```python
'''
给定一个pixel转换函数输入是一个pixel，输出是某个转换的值，如果有一个很大的图片4K*5K 大小，想要对每个pixel依次操作得到返回的结果如何计算。

设给定的转换函数是 def PixelTransform(pix): return something
'''

def func(img):
  n,m=img.shape
  mp={i:PixelTransform(i) for i in range(256)}
  for i in range(n):
    for j in range(m):
        img[i][j]=mp[img[i][j]]
  return img

```

- 算法题 2

```python

'''
如果PixelTransform函数接受两个参数pix和gamma，对一张给定的图片需要对每个pixel计算多种不同的gamma值得到结果，假设有k个gamma选项，最终得到k个处理后的图片，求着k个图片所有的pixel的均值。

如果pixelTransform函数接受的输入是一个区域，这个算法还适用吗？
'''

def PixelTransform(pix,gamma):
    return None

def func(img,gammas):
    cont={}
    n,m=img.shape
    for i in range(n):
        for j in range(m):
            cur=img[i][j]
            cont[cur]=cont.get(cur,0)+1

    g_cache={i:0 for i in cont}
    for g in gammas:
        for pix,cnt in cont.items():
            g_cache[pix]+=PixelTransform(pix, g)

    total_pix_cnt=0
    total_sum=0
    for pix,cnt in cont.items():
        t_g = g_cache[pix]
        total_sum += t_g*cnt
        total_pix_cnt += cnt

    return total_sum/(total_pix_cnt*len(gammas))
```

<div data-hk-top-pages="5"> </div>
