---
title: '2021秋招提前批-百度搜索策略部LTR部门-面试总结'
date: 2020-07-06
permalink: /posts/2020/07/baidu_search_ltr_interview/
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

## 百度搜索策略部，LTR 部门面试总结

部门概况，搜索的推荐自然排序算法。LTR（learning to rank）。主要负责自然结果的排序，然后后续会有加入一些商业结果和其他结果变成最终展现出来的样子。
团队目前 15 人左右，目前也会用一些 NN 的方法进行 feature 提取然后排序。搜索和推荐不太一样，搜索的时候很多是有人工的 tag 标注的。

## 一面

- 自我介绍
- 介绍了两篇论文
- 你是什么专业的

- 对数理统计熟悉吗，说一下大数定理

  - 大数定理大概就是 当样本量足够大的时候 频率等于概率，具体公式不太记得。。。。
  - [大数定理](https://baike.baidu.com/item/%E5%A4%A7%E6%95%B0%E5%AE%9A%E5%BE%8B/410082?fromtitle=%E5%A4%A7%E6%95%B0%E5%AE%9A%E7%90%86&fromid=9679413&fr=aladdin)

- 中心极限定理记得吗？

  - 。。。不记得
  - [中心极限定理](https://baike.baidu.com/item/%E4%B8%AD%E5%BF%83%E6%9E%81%E9%99%90%E5%AE%9A%E7%90%86/829451?fr=aladdin)

- 讲一下贝叶斯分类的原理？

  - 。。。依稀记得朴素贝叶斯分类器是利用贝叶斯公式计算后验概率。
  - [贝叶斯分类器](https://www.jianshu.com/p/6728b311338c)

- 讲一下 logistic regression

  - 全连接层输出+sigmoid 激活函数 得到一个 0-1 之间的概率值

- 你这个是深度神经网络里的，讲解一下最原始的逻辑回归

  - 就是一个权重 w 对特征进行加权求和然后加上一个 bias，然后送到 sigmoid 函数里面预测出一个 0~1 之间的概率。
  - [逻辑回归](https://www.jianshu.com/p/6728b311338c)

- 写一下逻辑回归的 loss

  - 交叉熵 loss： $-(ylog(h(x))+ (1-y)log(1-h(x)))$

- 为什么是 log？
  - 没太理解这个问题啥意思

* 对排序了解吗，写一下快速排序的代码

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.nums=nums
        self.quick_sort(0,len(nums)-1)
        return self.nums

    def quick_sort(self,left,right):
        if left>=right:
            return
        cur=self.nums[left]
        l=left
        r=right
        while r>l:
            while r>l and self.nums[r]>=cur:
                    r-=1
            self.nums[l]=self.nums[r]
            while l<r and self.nums[l]<=cur:
                    l+=1
            self.nums[r]=self.nums[l]
        self.nums[l]=cur
        self.quick_sort(left,l-1)
        self.quick_sort(l+1,right)
        return
```

- 给一个数组怎么找到其中较大的 k 个数

  - 使用一个小根堆，每个数依次放入堆里，当堆的元素大于 k 的时候弹出堆顶元素，对所有的数依次操作，堆中保存的结果即为较大的 k 个数

- 时间复杂度多少

  - Nlog(k)

- 还有其他方法实现吗

  - 排序一下取前 k 个。。。

- 能用之前快排的思路做吗
  - 好像可以，每次判断当前的 l 将数组划分为左右两个部分，长度分别为 l_cnt, r_cnt

```python
if k==r_cnt:
  直接返回右半部分所有数字

elif k>r_cnt:
  返回右半部分所有数字 + func(nums[:l])中较大的k-r_cnt个数

else:
  返回func(nums[l+1:])中较大的k个数
```

- 实现 sqrt 函数
  - 可以使用二分法或者牛顿法

```python
'''
二分法:
'''

def sqrt(x):
    l=0
    r=x
    pre_mid=x
    while True:
        mid=(l+r)/2.0
        if abs(mid-pre_mid)<10**(-6):
            break
        if mid**2==x:
            return mid
        if mid**2>x:
            r=mid
        else:
            l=mid
        pre_mid=mid
    return pre_mid


'''
牛顿法：
'''
def sqrt(num: int) -> bool:
    pre_x = num
    while True:
      cur_x=0.5*(pre_x+ num/pre_x)
      if abs(cur_x-pre_x)<10**(-6):
        break
      else:
        pre_x=cur_x
    return cur_x
```

## 二面

二面主要是讲了自己的两篇论文，然后展示了一些 demo，然后就论文里的一些内容和面试官做了一些交流。
问了一下自己未来的工作方向的打算，然后聊了一下部门正在做的事情等。总体比较轻松 45 分钟左右结束。

## 三面（本次面试时间 2020 年 7 月 13 日）

- 简单的自我介绍，项目和科研经历
- 对搜索有多少了解

- 搜索的评价指标应该有哪些

  - 相关性（搜出来的东西应该和 query 相关性足够高）
  - 多样性（搜出来的结果不能是一样的，需要能够全面解决 query 的需求）
  - 紧凑性（希望能够定位关键结果的位置，而不是直接返回一大堆文本而很难找到关键的位置）
  - [More](https://www.zhihu.com/question/19624746)

- 给定一个 query 搜索得到结果，如何设计一个搜索系统的流程

  - 先针对文本和 title 进行分词和关键词提取，根据不同的关键词分到不同的桶里，构建倒排索引
  - 针对 query 也进行关键词提取，然后每个关键词从对应的桶里进行相关页面检索，并对所有关键词取交集，得到相对小范围候选集
  - 针对候选集基于语义信息进行重排序。例如使用 bert 等 encoder 得到 title 和正文内容的 embedding，同时对 query 也进行 embedding，然后计算 cosine 相似度进行排序。
  - [More](https://www.zhihu.com/question/19937854/answer/61704536)

- 搜索和推荐的区别

  - 推荐优化点击率，只要能够吸引用户点进去就 ok；搜索需要优化相关性，即使 title 相关但是内部的内容不想相关的话也是不好的结果。
  - 推荐的话一定程度上能够做到离线，因为用户画像是相对固定的数据，可以先离线将用户可能感兴趣的商品做一个初步的筛选和排序准备；而搜索的话会需要考虑到实时性，因为无法预估用户的搜索意图，所以只能来了 query 之后再进行检索和排序。
  - 面试官还说： 推荐的话更多的面对的数据是相对同构的，比如都是文本或者都是图片或者都是表格;但是搜索的话所面对的数据更加多样性，不同的网页内的内容有各种各样的组织形式，甚至还会涉及到一些跨模态的内容。

  - 在设计搜索排序算法里，需要想尽办法让最好的结果排在最前面，往往搜索引擎的前三条结果聚集了绝大多数的用户点击。简单来说，“好”的搜索算法是需要让用户获取信息的效率更高、停留时间更短。
  - 衡量推荐系统是否足够好，往往要依据是否能让用户停留更多的时间（例如多购买几样商品、多阅读几篇新闻等），对用户兴趣的挖掘越深入，越“懂”用户，那么推荐的成功率越高，用户也越乐意留在产品里。

- 对 Attention 的了解讲一下

  - 视觉里的 attention 机制最原始的就是 heatmap 的形式，对 feature map 上的每个 pixel 附上一个权重。
  - NLP 领域例如 bert 或者 transformer 等模型使用 multi-head self-attention 机制，对于每个单词会将其他和他有相关性的单词的 embedding 加权求和融合到其自身 embedding 中，从而得到 context 信息。
  - 跨模态等领域会有 cross attention，去寻找成对数据中最相关最重要的部分等。
  - Attention 机制本质上就是去得到一个可学习可自动生成的权重，从而区别出不同的 feature 重要性。

- 对未来的工作地点和工作内容有没有啥偏好和考虑
- 如果符合 AIDU 的要求，会有后续加面

---

<div data-hk-top-pages="5"> </div>
