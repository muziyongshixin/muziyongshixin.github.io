---
title: '2021秋招-幻方量化-深度学习-面试总结'
date: 2020-08-09
permalink: /posts/2020/08/huanfanglianghua/
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

## 幻方量化-深度学习研究员/工程师 面经

部门概况：研发团队貌似只有杭州有。

## update : 公司联系说可能泄露题库，暂不公开了 （2021年4月）

<!--

## 笔试 （2020 年 8 月 09 日）

- 算法题 1

```python

'''
压缩字符串，如果一个字符出现的次数大于等于3次的话，将他表示成 数字+字符

例如：   aaabccccd**zzz   -> 3ab4cd883z
'''
# coding=utf-8
import sys
def encode(s):
    # 返回编码后的字符串
    final = []
    pre = '0'
    cnt = 0
    for c in s:
        if c != pre:
            if cnt < 3:
                for i in range(cnt):
                    final.append(pre)
            else:
                final.append(str(cnt))
                final.append(pre)
            pre = c
            cnt = 1
        else:
            cnt += 1
    if cnt >= 3:
        final.append(str(cnt))
        final.append(pre)
    else:
        for i in range(cnt):
            final.append(pre)
    result = ''.join(final)
    return result

for l in sys.stdin:
    print(encode(l.strip()))
```

- 算法题 2

```python

# coding: utf-8

'''
设计一个滑动窗口类，窗口的大小为10，每次输入是多个连续交易日的价格数据，返回滑动窗口内的最大回撤

最大回撤定义为  最大的a[i]-a[j]  其中i，j均在窗口内，且满足 j>i
例如当前窗口为： [1,2,3,4,10,3,0,6,7,7] 最大回撤为10 = 10-0

返回每个输入最后的滑动窗口的最大值。

输入示例：
1,2,13,12,1,3,0,4,10,3,0,6,7,7
3,-3,4,5,6,7,1

示例输出：
10
6
'''
import sys
class RollingWindow(object):
    def __init__(self):
        # 请完成此功能
        self.data = []

    def append(self, n):
        # 请完成此功能
        self.data.append(n)

    def get_max_drawdown(self):
        # 请完成此功能
        final_data = self.data[-10:]
        tail_min = final_data[-1]
        result = 0
        for i in range(len(final_data) - 2, -1, -1):
            cur = final_data[i] - tail_min
            result = max(result, cur)
        return result

if __name__ == '__main__':
    for line in sys.stdin:
        w = RollingWindow()
        for x in line.split(','):
            w.append(int(x))
        print(w.get_max_drawdown())

```

- 算法题 3

```python
'''
判断一个输入的9*9 的二维数组是否能够构成有效的数独的解。
'''
# coding=utf-8

import sys


def check_sudoku(grid):
    # grid是list of lists, 使用grid[0][0]访问其中数字
    # 请完成此功能
    row = len(grid)
    if row != 9:
        return False
    row_cache = {i: [] for i in range(9)}
    col_cache = {i: [] for i in range(9)}
    grid_cache = {i: [] for i in range(9)}

    for i in range(row):
        cur_row = len(grid[i])
        if cur_row != 9:
            return False

    for i in range(9):
        for j in range(9):
            cur = grid[i][j]
            row_cache[i].append(cur)
            col_cache[j].append(cur)
            a = i // 3
            b = j // 3
            gn = a * 3 + b
            grid_cache[gn].append(cur)

    for i, nums in row_cache.items():
        nums=set(nums)
        if len(nums)!=9:
            return False
        for i in range(1,10):
            if i not in nums:
                return False
    for i, nums in col_cache.items():
        nums = set(nums)
        if len(nums) != 9:
            return False
        for i in range(1, 10):
            if i not in nums:
                return False
    for i, nums in grid_cache.items():
        nums = set(nums)
        if len(nums) != 9:
            return False
        for i in range(1, 10):
            if i not in nums:
                return False
    return True

if __name__ == '__main__':
    grid = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            print(check_sudoku(grid))
            grid = []
            continue
        grid.append([int(x) for x in line.split(',')])
    print(check_sudoku(grid))
```

- 问答题 1:

  - 对于逻辑回归（logistics regression），给出赋予每个样本不同权重的 L-2 正则化逻辑回归的交叉熵损失函数（含 sigmoid 函数），以及参数更新过程。

- 问答题 2:

  - 给出 SVM 推导过程，包含拉格朗日优化式子，以及样本权重表达式，截距表达式等。
  - 通过计算说明 SVM 可以通过核函数快速进行非线性推广的基本前提，并说明高斯核和多项式核的最本质差别是什么。

- 问答题 3

  - 对于 EM 算法，给定样本{x1,x2...xm},样本间独立，引入隐含变量 z，使得 p(x,z)最大化，先给出 p(x,z)的最大似然估计，然后由该似然推导出 EM 算法的 E-step 和 M-step。

- 问答题 4:
  - 给出卷积神经网络 CNN 的误差传播公式，以及权值更新公式（提示，反向传播的过程中分全连接层，卷积层和池化层，参数更新只计算卷积层和全连接层）

## 一面

一面主要是问的是论文和当前实习的项目相关的问题。

- 自我介绍
- 介绍 CVPR 的论文
- 介绍实习时候的工作，遇到的问题，解决方案。

## 二面

二面主要问的是相对开放一些的问题

- 自我介绍
- 胸部 X 光片诊断报告的生成的具体做法，遇到的问题，怎么解决样本不平衡的问题等
- 如果在 X 光片中有一些先验信息比如病变通常在 5\*5 的区域内，有什么方法可以改进你的模型和算法。
- 如果在量化领域有大量的时序数据是很低信噪比的类似股票曲线这种，无法直接使用 CNN 或者 lstm 进行特征提取，有没有什么好的方法去做特征提取。

  - 先将数据按区间切片，然后利用 transformer 这种 model 的 self attention 机制去在时序上做 context 信息提取。用[CLS] token 的 embedding 作为整个时间段的 feature 然后在下游任务进行端到端的训练。

- 如果无法做到端到端的训练，先要必须将特征提取出来之后才能做下游的任务，有什么策略可以做特征的提取，信号处理相关的知识有了解吗。

- 反问面试官
  - 幻方量化的 AI Lab 目前人员规模七八人的样子。
  - 工作内容主要是用深度学习相关技术做股票投资，目前还没使用强化学习等策略
  - 工作的评价标准的话更多是模型在实盘的效果，一个模型可能过一段时间就不 work 了，也没有固定的数据集来判断提升多少个点什么的。
  - 工作相比于 BAT 可能要更自由一点。

## 三面

- 三面主要是就隐写的论文进行了一些技术的探讨，以及实验的设置，应用场景的讨论等。总体算是比较开放性的讨论，没有很细节的问题。


## 结果
凉了

-->

<div data-hk-top-pages="5"> </div>
