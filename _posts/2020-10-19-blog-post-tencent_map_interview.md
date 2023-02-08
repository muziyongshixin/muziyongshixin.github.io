---
title: '2021秋招-腾讯地图-面试总结'
date: 2020-10-19
permalink: /posts/2020/10/tencent_map/
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

<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-8633559477921171"
     crossorigin="anonymous"></script>
---

## 腾讯地图 应用研究 面经

部门概况：北京腾讯地图，地图平台部，主要负责地图的检索以及信息流的推荐相关业务。部门大概 30 多个人，地点在北京总部。面试官说上班时间是 10-8,9 项目紧急的时候偶尔周末会加班。

## 一面 （2020 年 10 月 19 日）

- 自我介绍

- 讲解 CVPR 的论文。

- 讲解了一下 AAAI 的论文。

- 讲了一下在阿里的实习经历。

- 算法题 1

```python
'''
实现split函数
'''
def split(s):
    result=[]
    t=[]
    for c in s:
        if c==' ':
            if len(t)>0:
                result.append(''.join(t))
            t=[]
        else:
            t.append(c)
    if len(t)>0:
        result.append(''.join(t))
    return result
```

- 算法题 2

```python
'''
链表反转
class ListNode:
    def __init__(self,val):
        self.val=val
        self.next=None
'''

def reverse_link_list(head):
    if head is None:
        return None
    pre=None
    cur=head
    nxt=cur.next
    while nxt is not None:
        cur.next=pre
        pre=cur
        cur=nxt
        nxt=cur.next
    cur.next=pre
    return
```

- 算法题 3

```python
'''
计算一个数组的逆序对数
'''

def reversePairs(nums,begin,end):
    if end<=begin:
        return 0

    mid=(begin+end)>>1
    l_cnt=func(nums,begin,mid)
    r_cnt=func(nums,mid+1,end)

    t=[]
    cur_cnt=0
    l=begin
    r=mid+1
    while l<=mid  and r<=end:
        if nums[r]<nums[l]:
            cur_cnt+= (mid-l+1)
            t.append(nums[r])
            r+=1
        else:
            t.append(nums[l])
            l+=1
    while l<=mid:
        t.append(nums[l])
        l+=1
    while r<=end:
        t.append(nums[r])
        r+=1

    for i in range(len(t)):
        nums[begin+i]=t[i]
    return l_cnt+r_cnt+cur_cnt
```

## 二面（2020 年 10 月 21 日）

- 自我介绍
- 讲解 CVPR 论文
- 四道算法题，一道问答题。。。

- 算法题 1

```python
'''
删除一个字符串中重复的字符，忽略大小写，只能在当前字符串上修改，且只能使用O(1)的额外空间。
s='abACeBF' ->  abCeF
'''

def funcA(chars):
    cache=0
    for i,c in enumerate(chars):
        t=c.lower()
        cur= 1 << (ord(t)-ord('a'))
        if cur&cache==cur:
            chars[i]=''
        cache |= cur
    l=0
    r=0
    while l<len(chars) and r < len(chars):
        if chars[r]=='':
            r+=1
        else:
            chars[l]=chars[r]
            l+=1
            r+=1
    return chars[:l]
```

- 算法题 2

```python
'''
找出排序二叉树第K大的节点
'''
global cnt,result,k
def help_funcB(root):
    global cnt,result,k
    if root is None or result is not None:
        return
    help_funcB(root.right)
    cnt += 1
    if cnt == k:
        result=root
        return
    help_funcB(root.left)
    return

def funcB(root,input_k):
    global cnt,result,k
    result=None
    cnt=0
    k=input_k
    help_funcB(root)
    return result

```

- 算法题 3

```python
'''
一个链表有两个指针，一个next指针，一个random指针，深度复制这个链表，返回复制后的链表的头结点
'''
class ListNode:
    def __init__(self,val):
        self.val=val
        self.next=None
        self.random=None
cache={}
def funcC(head):
    global cache
    if head is None:
        return None
    if head in cache:
        return  cache[head]
    cur=ListNode(head.val)
    cache[head]=cur
    cur.next=funcC(head.next)
    cur.random=funcC(head.random)
    return cur
```

- 算法题 4

```python
'''
返回两个字符串的最长公共子串的长度 （注意和最长公共子序列有点不一样）
'''
def funcD(strA,strB):
    if len(strA)==0 or len(strB)==0:
        return 0

    n=len(strA)
    m=len(strB)
    dp=[[0]*(m+1) for i in range(n+1)]

    result=0
    for i in range(n):
        for j in range(m):
            if strA[i]==strB[j]:
                dp[i+1][j+1]=dp[i][j]+1
            else:
                # dp[i+1][j+1]=max(dp[i][j+1],dp[i+1][j]) # 最长公共子序列 用这个
                dp[i+1][j+1]=0 # 最长公共子串 用这个
            result=max(result,dp[i+1][j+1])
    return result

```

<div data-hk-top-pages="5"> </div>
