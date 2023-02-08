---
title: '2021秋招-依图科技-面试总结'
date: 2020-09-15
permalink: /posts/2020/09/yitu/
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

## 2021 秋招 依图科技 面经

部门概况：面试的部门主要是做安防的，包括 human pose，人脸识别，身份识别等。

## 一面（2020 年 9 月 15 日）

- 自我介绍
- 简单介绍了一下 CVPR 的论文
- 写代码实现 BN 的 forward 过程

```python
import torch

def BN_forward(x):
    B,C,H,W=x.shape
    mean_val=x.mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
    var_val=((x-mean_val)**2).mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
    normed_x=(x-mean_val)/(var_val**0.5)
    gamma=1
    bais=0
    result=gamma*normed_x + bais
    return result

if __name__=='__main__':
    x=torch.randn(10,3,5,5)
    bn = nn.BatchNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)
    official_result=bn(x)
    my_result=BN_forward(x)
    diff=(official_result-my_result).sum()
    print(diff)

```

- Layer Normalization, Group Normalization 等的区别？ -[Link](https://zhuanlan.zhihu.com/p/91965772)

- 实现一下 resnet 的 basic block

- Git 有哪些操作？

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
```

- 算法题实现 topk

```python
def topk(nums,k):
    if len(nums)<=k or len(nums)<=1:
        return nums
    cur=nums[0]
    left=0
    right=len(nums)-1
    while left<right:
        while left<right and nums[right]>=cur:
            right-=1
        nums[left]=nums[right]
        while left<right and nums[left]<=cur:
            left+=1
        nums[right]=nums[left]
    nums[left]=cur
    if len(nums)-left==k:
        return nums[left:]
    elif len(nums)-left>k:
        return topk(nums[left+1:],k)
    else:
        return nums[left:] + topk(nums[:left], k-(len(nums)-left))

if __name__=='__main__':
    nums=[1,3,2,4,5,8,0,19]
    k=3
    result=topk(nums,k)
    print(result)
```

## 另外一个部门一面（2020 年 10 月 9 日）

- 自我介绍
- 算法题

```python
'''
给一个二维矩阵，其中有0-1两种值，1表示该像素是人脸，0表示是背景，画面中可能会有多张人脸信息，返回最大的人脸所占的像素个数。


经典BFS，或者DFS
'''
matrix=None
n=None
m=None

def bfs(i,j):
    global matrix,n,m
    q=[(i,j)]
    matrix[i][j]=-1
    cnt=0
    directions=[(-1,0),(1,0),(0,-1),(0,1)]
    while len(q)>0:
        tq=[]
        for x,y in q:
            cnt+=1
            for dx,dy in directions:
                nx=x+dx
                ny=y+dy
                if nx>=0 and nx<n and ny>=0 and ny<m and matrix[nx][ny]==1:
                    matrix[nx][ny]=-1
                    tq.append((nx,ny))
        q=tq
    return cnt

def main(data):
    global  matrix,n,m
    n=len(data)
    if n==0:
        return 0
    m=len(data[0])
    if m==0:
        return 0
    matrix=data

    result=0
    for i in range(n):
        for j in range(m):
            if matrix[i][j]==1:
                cur=bfs(i,j)
                result=max(result,cur)
    return result

if __name__=='__main__':
    #data=[[0,1,1,0],[0,1,1,0],[0,0,0,1],[0,0,0,1]]
    data=[]
    for i in range(3):
        t=input()
        t=[int(x) for x in t.split()]
        data.append(t)
    result=main(data)
    print(result)
```

- C++中 vector 会初始分配一块空间，当 push_back 达到一定程度的时候会进行扩容，这个过程是怎么样的，平均下来每次 push_back 的时间复杂度是多少。

  - 当初始空间满了的时候会申请一块 2 倍大小的新空间，然后将原始的数据一个个拷贝到新的空间里。平均下来每次 push_back 的时间复杂度是 O(1),证明如下：
  - 设长度为 N 的空间需要移动的元素次数是 T(N)，则其其中一半是由上一步所有的元素拷贝过来的次数，以及剩下的空间的 push_back 的次数，所以有递推公式：
  - T(N)=N+T(N/2) = N + N/2 + N/4 + ... + 1 , 总共有 log_2(N) + 1 项。
  - 则这是一个以 1 为初始，2 为倍数的等比数列求和。
  - T(N) = 1 \* (1-2^(log_2(N)+1)) / (1-2) = 2N - 1 = O(N)
  - 所以平均下来每个元素的 push_back 操作的时间复杂度是 O(1)的。

- 简单介绍了一下论文。

## 二面（2020 年 10 月 13 日）

- 自我介绍
- 算法题

```python
'''
输入： N 个数和 target
输出：找出 N 个数中所有连续区间和等于 target
case:
输入：1 1 2 2 1 1 3 3 4 5 6 ， target=4
输出
5
0 2
2 3
3 5
5 6
8 8
'''
def func(nums,target):
    if len(nums)==0:
        return []
    result=[]
    cache={}
    pre_sum=0
    for i,n in enumerate(nums):
        cur_sum=pre_sum+n
        key=cur_sum-target
        if cur_sum==target:
            result.append((0,i))
        if key in cache:
            for begin in cache[key]:
                result.append([begin+1,i])
        if cur_sum not in cache:
            cache[cur_sum]=[]
        cache[cur_sum].append(i)
        pre_sum=cur_sum
    return result

if __name__=='__main__':
    nums=[0,0,4,0,0]
    target=4
    result=func(nums,target)
    print(len(result))
    for u,v in result:
        print(u,' ',v)
```

- 讲解了一下 CVPR 的论文

## 三面 （2020 年 10 月 19 日）

三面的面试官是依图的前 10 号员工主要是做语音技术的，面试的时间大概一个小时，感觉有点像是压力面。

- 讲解 CVPR 的论文并根据论文进行提问，包括遇到的问题，怎么解决的，期间的主要的 road map 有哪些，未来的可以扩展的方向之类的。

- 其他的就是有点像 HR 面，比如遇到了什么问题，遇到的最大的挫折是什么之类的。

<div data-hk-top-pages="5"> </div>
