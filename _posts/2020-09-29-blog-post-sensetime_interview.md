---
title: '2021秋招-商汤科技-面试总结'
date: 2020-09-29
permalink: /posts/2020/09/sensetime/
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

## 商汤科技 视觉研究员 面经

部门概况：商汤研究院，daijifeng 手下的一个组。leader 之前是 MSRA 的，部门的业务和算法研究各占 50%。部门整体大概 20 个人左右，北京上海各七八人，其中正式员工七八人，实习生十几个。半年一次考核，按照 OKR 主要是主管打分。户口可能优先会给博士生。

## update : 公司联系说可能泄露题库，暂不公开了 （2021年8月）

<!--

## 一面 （2020 年 9 月 29 日）

- 自我介绍

- 算法题 1

```python
'''
给定一个包含很多个非负整数的数组，按照某种顺序组合这些数，使得最终的结果值最大
'''
from functools import cmp_to_key

 def cmp(a,b):
     a=str(a)
     b=str(b)
     if a+b>b+a:
         return -1
     else:
         return 1

 def func(nums):
     if len(nums)==0:
         return ''
     nums.sort(key=cmp_to_key(cmp))
     if nums[0]==0:
         return '0'
     result=[str(x) for x in nums]
     result=''.join(result)
     return result

if __name__ == "__main__":
    t=[565, 56567, 56564567, 56167, 54167, 54164999, 5, 564, 56, 54]
    result=func(t)
    print(result)

```

- 算法题 2

```python
'''
设计一个word里使用的单词检查器，并设计相应的函数接口

给定一个词典，和一个待检查的单词。  待检查的单词可能是错误的，所以需要返回可能正确的单词。

这里只考虑正确的单词包含和输入单词一样的字符，但是可能不同的排列顺序。
'''
class wordCheck:
    def __init__(self,word_dict):
        self.max_len=0
        self.cache={}
        self.word_dict=set(word_dict)
        for w in word_dict:
            self.max_len=max(self.max_len,len(w))
            k=[0]*26
            for c in w:
                k[ord(c)-ord('a')]+=1
            k=tuple(k)
            if k not in self.cache:
                self.cache[k]=[]
            self.cache[k].append(w)
    
    def get_candidates(self,word):
        if word in self.word_dict:
            return 0,[]
        if len(word)==0 or len(word)>self.max_len:
            return -1,[]
        k=[0]*26
        for c in word:
            k[ord(c)-ord('a')]+=1
        k=tuple(k)
        result=self.cache.get(k,[])
        if len(result)==0:
            return -2,result
        else:
            return len(result),result

if __name__ == "__main__":
    word_dict=['dog', 'act', 'cat', 'mouse', 'kitty']
    wc=wordCheck(word_dict)
    query=['cat','tca','tcaa']
    for q in query:
        status_code,result=wc.get_candidates(q)
        print(status_code,result)
    # dictionary = {}
    # word = 'correct' -> '0,[]'  正确单词
    # word = 'asdfasdfsadfsadfasdfsdffsfaf' ->  '-1,[]' 过长/过短的单词
    # word1= 'aaaaaaaa' -> '-2,[]'  找不到合理的重排序后的单词
    # word2= 'abcd' -> 'num,[]'     能够找到合理的重排序的单词
```

- 讲解了一下 CVPR 的论文
- 内存中堆内存和栈内存有啥区别
- 进程和线程的区别

## 二面（2020 年 10 月 12 日）

- 二面是部门的主管戴继峰，面试时间总共 30 分钟左右，上来面了两个物理问题。

- 空间站中的宇航员为啥会失重

  - 重力提供圆周运动的向心力，受力平衡所以失重

- 火星探测器的轨道是什么样子的，并分析其受力情况。

- 在两个 Visual studio IDE 中运行相同的一份代码，他们的操作结果和输出是否会相互影响？

  - 不会因为是独立的进程，有自己独立的内存空间。

- 如果两个相同的代码某个指针输出的地址空间相同是什么原因？

  - 因为操作系统的分页机制，所以地址空间是虚拟空间。

- 手推 softmax 的前向和反向传播的公式。

- 反问面试官问题，得知北京户口比较困难，先锋计划基本无望。工作内容比较灵活，可以有纯 research，也可以有业务相关的结合，和其他的业务团队合作。另外一个优势是可以方便以后读博，有比较多的学术合作的资源。

## 三面（2020 年 10 月 14 日）

- 自我介绍
- 根据 CVPR 的论文进行深入的讨论，比如领域的发展方向什么的。
- 怎么判断一个导数为 0 的点是鞍点还是极值点？
-->

<div data-hk-top-pages="5"> </div>
