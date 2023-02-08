---
title: '2021秋招-华为诺亚-面试总结'
date: 2020-08-19
permalink: /posts/2020/08/huawei_nuoya/
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

## 华为诺亚北京-终端视觉部门

## 笔试

- 给定一个 M 行，N 列的二维矩阵上每个位置站了一个人，左上角坐标是(0,0),右下角坐标是(M-1,N-1),按照顺时针的方向逐层报数，报数从 1 开始，如果报数的编号个位数是 7，且十位数是奇数，则该坐标上的人会被选做特种兵，请按顺序输出特种兵的坐标。

输入： M N

输出：按顺序输出所有特种兵的坐标

M 和 N 取值在 [10,1000]，不合理的输入返回空列表

例如：

输入：10 10

输出：[[7,9],[1,1],[8,2],[7,5],[4,4]]

```python
'''
主要的坑在不合理的输入的处理上
'''
import sys
def func(M,N):
    result=[]
    idx=1
    visited=[[0]*N for i in range(M)]
    directions=[(0,1),(1,0),(0,-1),(-1,0)]
    p=0
    x,y=0,0
    while idx<= M*N:
        str_idx=str(idx)
        if len(str_idx)<2:
            shiwei=0
        else:
            shiwei=int(str_idx[-2])
        gewei=int(str_idx[-1])
        if gewei==7 and shiwei&1==1:
            result.append([x,y])
        visited[x][y]=1
        idx+=1

        dx,dy=directions[p]
        nx=x+dx
        ny=y+dy
        if not(nx>=0 and nx<M and ny>=0 and ny<N and visited[nx][ny]==0):
            p=(p+1)%4
            dx,dy=directions[p]
            nx=x+dx
            ny=y+dy
        x=nx
        y=ny
    return result
if __name__ == "__main__":
    for data in sys.stdin:
        try:
            fm,fn=[float(x) for x in data.split()]
            M=int(fm)
            N=int(fn)
            if fm-M!=0 or fn-N!=0:
                print("[]")
            elif M>=10 and M<=1000 and N>=10 and N<=1000:
                result=func(M,N)
                output=str(result)
                output=output.replace(" ",'')
                print(output)
            else:
                print("[]")
        except:
            print("[]")
```

- 第二题

给定二叉树中每个节点的深度，判断能够构成多少种不同的二叉树

输入：第一行是 N,表示二叉树节点的个数。 第二行是 N 个数用空格分隔，表示每个节点的深度

输出：有多少种不同的二叉树的构成，结果对（10^9+7）取模

例如：

输入：

4

1 0 2 2

输出：

2

```python
'''
排列组合问题,主要的坑在于判断是否有的层的节点数 大于2**n,或者节点数为0
'''

def cmn(n,m): # 计算C^m_n的组合数
    down=1
    for i in range(1,m+1):
        down*=i
    top=1
    while m>0:
        top*=n
        n-=1
        m-=1
    return top//down

def func(nums):
    cache={}
    max_deepth=max(nums)
    for d in nums:
        cache[d]=cache.get(d,0)+1
    for deepth, cnt in cache.items():
        if cnt> 2**deepth:
            return 0
    all_depth=cache.keys()
    if len(all_depth)!=max_deepth+1:
        return 0
    result=1
    pre_layer_number=0
    for cur_depth in range(max_deepth+1):
        cur_layer_number=cache[cur_depth]
        if pre_layer_number==0:
            pre_layer_number=cur_layer_number
        else:
            cur_result=cmn(pre_layer_number*2,cur_layer_number)
            result*=cur_result
            result%=(10**9+7)
            pre_layer_number=cur_layer_number
    return result

if __name__ == "__main__":
    N=int(input())
    data=input().split()
    data=[int(x) for x in data]
    result=func(data)
    print(result)
```

- 第三题

俄罗斯方块问题，给定一个游戏中间状态用 frame 表示，和下一个方块用 brick 表示，问最优放置消除后还有多少层。

frame 用一个字符串表示，数字表示方块数量方向朝上。例如 2202 表示一个宽度为 4 的画面，第 1,2,4 列有两个方块，第 3 列没有方块。

```
|## #|
|## #|
|————|
```

brick 也用一个字符串表示，数字表示方块数量，方向朝下，例如 121 表示一个宽度为 3 朝下的倒三角方块。

```
###
 #
```

输入两行字符串，第一行表示 frame，第二行表示 brick。其中字串中的每个字符是一个数字，值在[0,9]之间。

例如：

输入：

2202

2

输出：

0

```python
'''
暴力模拟过了
'''
import copy
def func(frame,brick):
    fw=len(frame)
    fb=len(brick)
    grid=[[0]*fw for i in range(20)]
    for j in range(fw):
        cur_high=frame[j]
        for i in range(cur_high):
            grid[i][j]=1

    result=20
    i=0
    while i+fb <= fw:
        tmp_grid=copy.deepcopy(grid)
        max_down=20
        for j in range(fb):
            cur=i+j
            cur_down=20-frame[cur]-brick[j]
            max_down=min(max_down,cur_down)
        # print(i,max_down)
        for j in range(fb):
            cur_brick=brick[j]
            row=19-max_down
            while cur_brick>0:
                tmp_grid[row][i+j]=1
                cur_brick-=1
                row-=1
        # print(tmp_grid)
        cur_result=0
        for row in range(20):
            row_sum=sum(tmp_grid[row])
            if row_sum==0 or row_sum==fw:
                continue
            else:
                cur_result+=1
        result=min(result,cur_result)
        i+=1
    return result

if __name__ == "__main__":
    frame=input()
    frame=[int(x) for x in str(frame).strip()]
    brick=input()
    brick=[int(x) for x in str(brick).strip()]

    result=func(frame,brick)
    print(result)
```

## 一面（2020 年 8 月 26 日）

因为一面的面试官在实习的时候已经面过我了所以总体上没问很多问题。华为流水线式的一天面完了三面。

- 为啥笔试的第一题最开始超内存， 后面又 AC 了主要改了啥？

  - 把 visited 从保存二维坐标的 set 改成了二维数组，节省了一倍的内存。

- 讲 CVPR 的论文

- 一道算法题

```python
'''
下一个字典序  134->143    321->123     145765321->146123557
leetcode原题
'''
def func(nums):
    n=len(nums)
    # print(nums)
    idx=-1
    for i in range(n-1,0,-1):
        if nums[i]<=nums[i-1]:
            continue
        else:
            idx=i-1
            break
    if idx==-1:
        nums.reverse()
        return nums
    for i in range(n-1,idx,-1):
        if nums[i]>nums[idx]:
            nums[i],nums[idx]=nums[idx],nums[i]
            nums[idx+1:]=nums[idx+1:][::-1]
            break
    return nums
nums=input()
nums=[int(x) for x in nums.split()]
result=func(nums)
print(result)
```

## 二面（2020 年 8 月 26 日）

- 算法题

```python
'''
给你一个数组，里面保存着N个点的二维坐标，计算这些点两两之间的曼哈顿距离的和
曼哈顿距离定义为 dist=|x1-x2|+|y1-y2|

按照横向和纵向分离之后，排序。类似前缀和问题。
'''
p=[[1,2],[2,3],[3,4],[5,6]]

def func1(p): # 暴力解法O(N^2)
    n=len(p)
    result=0
    for i in range(n):
        for j in range(i+1,n):
            cur_dist=abs(p[i][0]-p[j][0])+abs(p[i][1]-p[j][1])
            result+=cur_dist
    return result
def func2(p): # 排序后前缀和，O(Nlog(N)+N)
    x_cache=[x[0] for x in p]
    y_cache=[x[1] for x in p]
    x_cache.sort()
    y_cache.sort()
    n=len(x_cache)
    result=0
    pre_dist=0
    for i in range(n):
        if i>=1:
            cur_dist=pre_dist+ i*(x_cache[i]-x_cache[i-1])
        else:
            cur_dist=0
        result+=cur_dist
        pre_dist=cur_dist
    pre_dist=0
    for i in range(n):
        if i>=1:
            cur_dist=pre_dist+ i*(y_cache[i]-y_cache[i-1])
        else:
            cur_dist=0
        result+=cur_dist
        pre_dist=cur_dist
    return result
```

- 论文是怎么发表的，Idea 是从哪里来的，科研工作是怎么分工的。

## 三面（2020 年 8 月 26 日）

三面有点像 HR 面，没有问太多技术上的问题。

- 对底层的算法优化有没有了解过，比如在移动端上的部署。
- 对未来的工作方向有没有什么倾向和偏好
- 觉得自己影响力最大的一片工作是哪个？
- 有没有觉得什么工作以前做的是不太好的，原因是什么。
- 实习期间和在学校科研有什么不同？
- 实习期间有没有遇到什么困难，怎么解决的，需不需要和上下游进行对接什么的。
- 论文发表和其他作者的一个分工是什么样子，科研方向通常是怎么决定的？

<div data-hk-top-pages="5"> </div>
