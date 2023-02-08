---
title: '2021秋招提前批-百度PaddlePaddle平台部-面试总结'
date: 2020-06-29
permalink: /posts/2020/06/baidu_paddle_interview/
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


## Baidu，深度学习平台部(PaddlePaddle)

- 部门概况：主要负责百度深度学习平台 PaddlePaddle 的开发，生态的构建相关工作。包括分布式开发，底层框架开发完善，工程优化，最新的模型的复现。总体团队规模 100 多人。
- 感觉一面涉及到很多底层的东西，问的很多东西感觉都是知识盲区。。。。。

### 一面

- 平时用的是 Pytorch 嘛，有没有做过相关性能优化，有没有发现过 Bug？

- Python 中生成器和迭代器的区别？

  - 生成器好像没太听说过。
  - [正解](https://www.cnblogs.com/chaojiyingxiong/p/9831579.html)

- 给定一个超长的字符串，可能需要好几个文件分开才能存储，然后一个用户修改了他，怎么定位出修改的是哪个字符？
  - 针对每个不同的文件，可以先通过文件描述符的修改时间判断最近修改的文件；在文件内部定义类似 log 机制，修改的时候自动记录修改的位置的偏移量。
- （如果不使用 log 机制呢？下载一个大文件怎么判断他是否下载完了？）

  - 通过 MD5 校验来判断文件是否下载完成。
  - 通过类似于纠错码的方式来判断修改的位置？

- 快速查询有哪些数据结构可以实现？

  - Hash 表，树（红黑树，B 树，B+树）

- Hash 表的原理

- python 中 Xrange 和 range 的区别？

  - xrange 好像只有在 python2 中有，python3 中好像没这东西，具体啥区别不知道，只知道功能一样。
  - [正解](https://blog.csdn.net/wtwcsdn123/article/details/89329403)

- Cache 和 Buffer 的区别？

  - Cache 用于访问加速，Buffer 作为缓冲区
  - cache 是为了弥补高速设备和低速设备的鸿沟而引入的中间层，最终起到**加快访问速度**的作用。
  - buffer 的主要目的进行流量整形，把突发的大数量较小规模的 I/O 整理成平稳的小数量较大规模的 I/O，以**减少响应次数**（比如从网上下电影，你不能下一点点数据就写一下硬盘，而是积攒一定量的数据以后一整块一起写，不然硬盘都要被你玩坏了）。

- cache 为啥能加速访问

  - 数据访问的局部性原理

- 深度学习训练的时候的速度瓶颈除了计算外还有啥？

  - IO，内存和显存之间的通信，跨机器之间的网络传输

- 如何判断通信是瓶颈？

  - 查看 GPU 计算的利用率，越低说明瓶颈在 IO 和通信上

- 用 python 实现一个多进程的生成器，输入是一个生成器，输出是按顺序的结果（多进程的 map 函数）

  - ？？？ 不太理解这个题啥意思，触及了我的知识盲区

- 平时是否有做过计算的优化，是否手写过 OP，CUDA？
  - 只做过将循环换成矩阵计算来加速。其他没搞过

* 算法题 1

  - 数据量貌似很小，直接暴力算法。可以使用字典树效率更高，但是内存消耗更大

```python
'''
#编写一个函数来查找字符串数组中的最长公共前缀。
#如果不存在公共前缀，返回空字符串 ""。
#输入: ["flower","flow","flight"]
#输出: "fl"
# 单词数目<100
# 单词长度<100
'''
def func(ws):
    result=''
    if len(ws)==0:
        return result

    min_len=min([len(w) for w in ws])
    if min_len==0:
        return result

    for i in range(1,min_len+1):
        prefix=ws[0][:i]
        flag=True
        for w in ws:
            if w[:i]!=prefix:
                flag=False
                break
        if flag:
            result=prefix
        else:
            break
    return result

if __name__=="__main__":
    data=input()
    ws=data.split(",")
    result=func(ws)
    print(result)

```

- 算法题 2

```python
'''
给定一个二叉树和一个单链表
判断二叉树上是否存在一个从上到下的路径和单链表的值恰好相同，有返回True， 没有返回False
'''

def func(root,head):
    if head is None:
        return True
    if root is None:
        return False

    if root.val==head.val:
        if head.next is None:
            return True
        if root.left is None:
            if root.right is None:
                return False
            if root.right.val!=head.next.val:
                return False
            else:
                return func(root.right,head.next)
        elif root.right is None:
            if root.left is None:
                return False
            if root.left.val!=head.next.val:
                return False
            else:
                return func(root.left,head.next)
        else:
            if root.left.val==head.next.val:
                return func(root.left,head.next)
            elif root.right.val==head.next.val:
                return func(root.right,head.next)
            else:
                return False
    else:
        l=func(root.left,head)
        r=func(root.right,head)
        return l or r

```

- 问面试官关于部门和业务相关的问题。

## 二面

- 介绍了自己的两篇论文和一个项目

- 算法题 1

```python
'''
1. 给你一个正整数的数组 A（其中的元素不一定完全不同），请你返回可在 一次交换（交换两数字 A[i] 和 A[j] 的位置）后得到的、按字典序排列小于 A 的最大可能排列。
如果无法这么操作，就请返回原数组。

示例 1：
输入：[3,2,1]
输出：[3,1,2]
解释：
交换 2 和 1

示例 2：
输入：[1,1,5]
输出：[1,1,5]
解释：
这已经是最小排列

示例 3：
输入：[1,9,4,6,7]
输出：[1,7,4,6,9]
解释：
交换 9 和 7

示例 4：
输入：[3,1,1,3]
输出：[1,3,1,3]
解释：
交换 1 和 3
'''

#coding=utf-8
def func(nums):
    if len(nums)==0:
        return nums
    pre=1000000000000000
    i=len(nums)-1
    while i>=0 and nums[i]<=pre:
        pre=min(pre,nums[i])
        i-=1
    if i==-1:
        return nums
    j=len(nums)-1
    while j>=0 and nums[j]>=nums[i]:
        j-=1
    while j-1>=0 and nums[j-1]==nums[j]:
        j-=1
    nums[i],nums[j]=nums[j],nums[i]
    return nums

if __name__=="__main__":
    data=input()
    data=[int(x) for x in data.split()]
    result=func(data)
    print(result)
```

- 基础题 2（选做）
  - C++不太熟没做

```cpp
// 试给出下列C++代码的输出，并解释原因

#include <iostream>
#include <utility>

struct X {
    X() { std::cout << "1"; }
    X(X &) { std::cout << "2"; }
    X(const X &) { std::cout << "3"; }
    X(X &&) { std::cout << "4"; }
    ~X() { std::cout << "5"; }
};

struct Y {
    mutable X x;
    Y() = default;
    Y(const Y &) = default;
};

int main() {
    Y y1;
    Y y2 = std::move(y1);
}
```

## 三面

总体感觉很轻松，20 多分钟就结束了，主要是一些宽泛的个人经历的讲解，以及工作内容，未来发展的一些沟通。

- 自我介绍
- 简单讲解了一下一篇论文
- 简单介绍了一下在腾讯的实习工作
- 学院主要是干啥的，导师是谁
- 闲聊，对于未来工作内容的沟通，北京户口貌似要抽签，工作节奏差不多 1095 或者 995。

---

<div data-hk-top-pages="5"> </div>
