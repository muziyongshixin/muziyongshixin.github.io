---
title: '二叉树的三种遍历（递归与非递归）'
date: 2020-09-29
permalink: /posts/2020/09/algorithm-binary-tree-Traversal/
categories:
  - Algorithm
tags:
  - Algorithm, Binary Tree

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

## 二叉树的三种遍历 递归和非递归实现

- 三种遍历的递归实现

```python
'''
递归遍历,实现起来比较简单，主要就是调整root.val放入结果的时机
'''
result=[]
def preorder(root):
    if root is None:
        return

    result.append(root.val)
    preorder(root.left)
    preorder(root.right)
    return

def inorder(root):
    if root is None:
        return
    inorder(root.left)
    result.append(root.val)
    inorder(root.right)
    return


def postorder(root):
    if root is None:
        return
    postorder(root.left)
    postorder(root.right)
    result.append(root.val)
    return
```

- 非递归实现

```python
'''
主要思想是使用栈来模拟递归调用的过程

其实先序和后序的代码非常相似，只是后续每次优先往右边深入，最后逆序一下结果

中序的结果是在while循环外面才保存到result中的，这个需要注意。
'''


def pre_order(root):
    result=[]
    stack=[]
    cur=root
    while cur is not None or len(stack)>0:
        while cur is not None:
            stack.append(cur)
            result.append(cur.val)
            cur=cur.left

        cur=stack.pop(-1)
        cur=cur.right
    return result

def in_order(root):
    result=[]
    stack=[]
    cur=root
    while cur is not None or len(stack)>0:
        while cur is not None:
            stack.append(cur)
            cur=cur.left
        cur=stack.pop(-1)
        result.append(cur.val)
        cur=cur.right
    return result

def post_order(root):
    result=[]
    stack=[]
    cur=root
    while cur is not None or len(stack)>0:
        while cur is not None:
            result.append(cur.val)
            stack.append(cur.right)
            cur=cur.right

        cur=stack.pop(-1)
        cur=cur.left
    result.reverse()
    return result

```

<div data-hk-top-pages="5"> </div>
