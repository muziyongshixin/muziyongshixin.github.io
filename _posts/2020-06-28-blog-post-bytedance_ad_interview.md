---
title: '2021秋招提前批-字节跳动国际化广告部门-面试总结'
date: 2020-06-28
permalink: /posts/2020/06/bytedance_ad_interview/
categories:
  - Interview
tags:
  - Interview,bytedance
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

## 字节跳动，国际化广告变现部门

- 部门概况：主要是 TicTok 的全球广告算法模型相关，有一些上游的团队负责特征提取等工作。团队规模大概 18 人。

### 一面

- 自我介绍
- 对简历的一个项目的细节的问答

- 算法题 1

```python
'''
删除字符串中多余的空格
input = "   I am a bytedancer    "
output  = "I am a bytedancer"
'''
def func(s):
    if len(s)==0:
        return ''

    i=0
    while i<len(s) and s[i]==' ':
        i+=1
    if i==len(s):
        return ''
    s=s[i:]
    j=len(s)-1
    while j>=0 and s[j]==' ':
        j-=1
    s=s[:j+1]

    cache=[]
    for c in s:
        if c !=' ':
            cache.append(c)
        else:
            if cache[-1]!=' ':
                cache.append(c)
    result=''.join(cache)

    return result


if __name__=="__main__":
    input_str=input()
    result=func(input_str)
    print(result)
```

- 算法题 2
  - 实现用的 DFS，主要思路就是每次删除一个字符，直到长度和 target 相同时进行比较。不过如果输入的字符串过长的话，这个的时间复杂度会很高，这样的话按照每次选择一个字符来增加字符长度的思路应该效率会更高 $C^6_{N}$。

```python
'''
好多牛牛
限定语言：C、Python、C++、Javascript、Python 3、Java、Go
给出一个字符串S，牛牛想知道这个字符串有多少个子序列等于"niuniu"
子序列可以通过在原串上删除任意个字符(包括0个字符和全部字符)得到。
为了防止答案过大，答案对1e9+7取模
示例1
输入
"niuniniu"
输出
3
说明
删除第4，5个字符可以得到"niuniu"
删除第5，6个字符可以得到"niuniu"
删除第6，7个字符可以得到"niuniu"
'''
result=0
def solve( s ):
    if len(s)<6:
        return 0
    if len(s)==6:
        return 1 if s=='niuniu' else 0
    func(s,0)

def func(s,begin_idx):
    global result
    if len(s)<=6:
        if s=='niuniu' :
            result+=1
        return
    for i in range(begin_idx,len(s)):
        sub_str=s[:i]+s[i+1:]
        #print(i,sub_str)
        func(sub_str,i)
    return

input_str=input()
solve(input_str)
print(result)

```

- 问面试官关于部门和业务相关的问题。
- 负责全球的广告算法推荐，工作压力会很大吗（广告这个东西也不能光靠算法，也要看商务卖的多不多，所以这个压力的问题主要还是看个人）
- 全球的广告算法是一样的嘛还是每个国家不同（会按照洲际进行不同，比如北美洲一个模型，亚洲一个模型。但是不会精细到国家）

- 广告部门相对于其他部门薪资会更高一些吗？（校招不知道，**社招的话会相对来说更高一些，毕竟现金流部门**）

## 二面

- 医疗 AI 领域的数据缺乏问题怎么解决？

  - 数据增强，Loss 加权重，找医院合作标数据

- 虚函数了解吗？

  - 用于实现多态,[详解](https://www.zhihu.com/question/23971699?sort=created)

- 对推荐系统有没有了解？FM 算法听说过没有？

  - 简单知道一些矩阵分解，协同过滤等古老算法，现在有一些用强化学习进行推荐。
  - FM 算法好像没听说过。

- 广告线上系统如果出现了过高预估的问题怎么解决？

  - 不太懂，瞎答的。针对高估的部分数据训练一个分类器，新来的数据先判断是否容易被高估，如果是容易被高估的数据，再额外训练一个纠偏的模型预测偏移量。来了新的 sample 先判断是否是容易高估的数据，如果不是直接用原始系统输出，如果是容易高估的数据，线上系统的结果减去预估偏移量得到的结果作为最终结果。
  - 有点打补丁或者 Boosting 思想的意思。

- 推荐系统的冷启动问题怎么解决？
  - 好像是业界难题，貌似目前没有什么很好的解决方法吧？
  - 缓解方案[如下](https://www.jianshu.com/p/193fea0a7004)

* self-attention 的机制解释一下，为啥叫 self-attention

  - self-attention 最开始是在 NLP 领域提出的，self 主要是针对于句子内部的单词相互之间计算 attention 信息。而相对的不同句子之间的 attention 通常叫做 cross-attention。

* 视觉的话 segmentation 的这问题是怎么建模的？

  - 用 CNN 作为特征提取器，在 feature map 上做 pixel-wise 的分类问题。

* 是否了解过传统的机器学习算法

  - 了解过一些决策树，SVM 相关的算法，但是平时不怎么用。

* 视觉中会要使用传统的那些特征嘛？
  - 目前来说，基本都是深度学习一把梭，直接用 NN 提取特征进行端到端训练。

- 算法题 1

```python
'''
给定一个二叉树，原地将它展开为链表。
例如，给定二叉树

    1
   / \
  2   5
 / \   \
3   4   6
将其展开为：
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
'''
#coding=utf-8
import sys
#str = input()
#print(str)
class TreeNode:
    def __init__(val):
        self.val=val
        self.left=None
        self.right=None

stack=[]
def func(root):
    global stack
    if root is None:
        return None
    if root.right is not None:
        stack.append(root.right)

    if root.left is not None:
        nxt=root.left
    elif len(stack)>0：
        nxt=stack.pop(-1)
    else:
        nxt=None
    root.right=nxt
    self.func(nxt)
    return root

if __name__=="__main__":
    input_node=input()
    result=func(input_node)
    return result
```

- 算法题 2

```python
'''
给定一个由正整数组成且不存在重复数字的数组，找出和为给定目标正整数的组合的个数。
示例:
nums = [1, 2, 3]
target = 4
所有可能的组合为：
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
'''
#coding=utf-8
result=0
target=0
nums=[]
def func(pre_sum):
    global nums,result,target
    if pre_sum==target:
        result+=1
        return
    if pre_sum>target:
        return
    for i in range(len(nums)):
        cur_sum=pre_sum+nums[i]
        func(cur_sum)
    return

if __name__=='__main__':
    global nums,target,result
    input_data=input()
    tmp=input_data.split(',')
    tmp=[int(x) for x in tmp]
    target=int(input())
    tmp.sort()
    tmp=[x for x in tmp if x<=target]
    func(0)
    print(result)

```

## 三面

上来就是算法题，感觉面试官是学数学的，频繁叫我推公式。。。。

- 算法题
  - 有一个长度为 n 的数组，求一个数 k，k 的取值区间为[1, n-1]，使得数组的前 k 个数和后 n-k 个数的方差和最小。
  - 先推方差公式,然后前缀和公式计算 O(N)时间复杂度：
  - $C= \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x})^2$
  - $ = \frac{1}{n} \sum_{i=1}^{n} x_i^2 - \frac{2 \hat{x}}{n} \sum_{i=1}^{n}x_i + \hat{x}^2 $
  - $ = \frac{1}{n} \sum_{i=1}^{n} x_i^2 - \hat{x}^2$

```python

def func(nums):
    n=len(nums)
    result=-1
    best=10000000000000000000000
    pre_sum=nums[0]
    sq_pre_sum=(nums[0])**2
    total_sum=sum(nums)
    sq_total_sum=sum([x**2 for x in nums])
    back_sum=total_sum-pre_sum
    sq_back_sum=sq_total_sum-sq_pre_sum
    for k in range(1,n):
        a=sq_pre_sum/k - (pre_sum/k)**2
        b=sq_back_sum/(n-k) - (back_sum/(n-k))**2
        cur=a+b
        if cur<best:
            best=cur
            result=k
        i=k
        pre_sum+=nums[i]
        sq_pre_sum+=(nums[i])**2
        back_sum=total_sum-pre_sum
        sq_back_sum=sq_total_sum-sq_pre_sum

    return result

```

- 讲解两篇论文和项目

- 平时深度学习模型训练怎么 debug

  - 看 Loss 曲线，看梯度信息，看参数分布，看 ReLU 激活值的概率（如果大量的激活都是 0，说明网络大量节点死亡，实际很少用过）

- ReLU 在 0 点处的梯度怎么处理？

  - 直接取 0 或者用$ln(1+e^x) $近似

- 平时使用什么 optimizer？

  - SGD，ADAM，通常使用 ADAM 因为收敛更快，SGD 往往能收敛到更优解但是较慢

- ADAM 的特点是啥为啥能更快的收敛

  - 因为有 momentum 机制同时结合了 RMSPROP 的优点提高了稳定性
  - [详解](https://www.jianshu.com/p/aebcaf8af76e)

- 平时使用过哪些 Normalization

  - 主要是用 BN，偶尔用过 Instance Normalization

- 二分类的时候，使用 sigmoid 激活函数为啥不能用 MSE loss？从梯度的角度解释一下
  - 手推公式的题目
  - 令 $sig =  \frac{1}{1-e^{-wx+b}}$
  - $loss= (y-sig)^2 $
  - $ \frac{\partial{loss}}{\partial{x}} = -2(y-sig)* \frac{\partial{sig}}{\partial{w}}$
  - $ =-2(y-sig)*(1-sig)(sig)x$
  - 存在的问题在于首先 sigmoid 输出趋近与 0 或者趋近 1 是梯度信息很小，容易产生梯度消失的问题。
  - 当 y=1 时，即使 sig 输出是 0 其 loss 导致的惩罚梯度也很小，惩罚不够。

* 问面试官一些问题和关于推荐系统的一些咨询：
  - 是不是使用的模型都比较简单？（是的，模型只是一部分，特征的选择，组合也是很重要的一块，特征工程）
  - 18 个人负责 tictok 全球的广告算法投放模型，涉及到视频广告，广告和文字的排版等很多问题是否工作压力很大（确实每个人需要负责很多事情，工作时间是弹性的，有时候需要早起和美国团队开会。理论上有机会 transfer 到美国）
  - 广告部门的年终奖会不会比其他的部门更多（不一定，不同部门的预期是不一样的，所以不是赚钱的越多奖金越多）
  - TicTok 广告的投放量大概多少（tictok 的 DAU 和国内的抖音差不多了）
  - 为啥 tictok 和抖音要两个团队（因为国际化的广告玩法和国内不一样，而且像抖音很赚钱，那么大家肯定都会 focus 到抖音业务上 ，tictok 就没人愿意搞了，所以需要一个专门的团队来 focus 在上面）
      <!-- （所以意思是 tictok 不是很赚钱？？？） -->
   
---
   <div data-hk-top-pages="5"> </div>
