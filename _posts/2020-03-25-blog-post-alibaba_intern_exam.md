---
title: '2020暑期实习-阿里巴巴笔试2020年3月25日场'
date: 2020-03-25
permalink: /posts/2020/03/alibaba_intern_exam/
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

## 2020 暑期实习，阿里巴巴笔试题 2020 年 3 月 25 日场

## 第一题

```python
'''
给定一个数组 n，然后给三个长度为 n 的数组，可以从这三个数组中选出一个长度为 n 的数组，第 i 个位置需要是从给出的三个数组第 i 个位置选择的，然后要求使这个数组后一项减前一项的绝对值之和最小。
输入示例:
5 9 5 4 4
4 7 4 10 3
2 10 9 2 3
这里可以选择 5 7 5 4 4，所以输出等于|7-5|+|5-7|+|4-5|+|4-4|=5。所以输出就是 5。
'''

def func1(data, n):
    dp = [[0] * n for i in range(3)]
    for j in range(3):
        dp[0][j] = 0
    for i in range(1, n):
        for j in range(3):
            dp[j][i] = min(
                [dp[0][i - 1] + abs(data[j][i] - data[0][i - 1]),
                 dp[1][i - 1] + abs(data[j][i] - data[1][i - 1]),
                 dp[2][i - 1] + abs(data[j][i] - data[2][i - 1])])
    result = min([dp[0][-1], dp[1][-1], dp[2][-1]])
    return result

if __name__=="__main__":
    n=int(input())
    data=[]
    for i in range(3):
        t=[int(x) for x in input().split()]
        data.append(t)
    result=func1(data,n)
    print(result)

```

## 第二题

```python
'''
给出一个二维矩阵，这个矩阵的每一行和每一列都是一个独立的等差数列，其中一些数据缺失了，现在需要推理隐藏但是可以被唯一确定的数字，然后对输入的查询进行回答。

输入描述：
第一行，n,m,q分别表示矩阵的行数，列数和查询的条数。
接下来的n行，每行m个数表示这个矩阵，0表示缺失数据。-10^9≤A_{ij}≤10^9

接下来q行，每行两个数字i,j表示对矩阵中第i行第j列的数字进行查询。

输出描述：
如果可以确定该位置的数字，则输出数字，如果不能确定则输出UNKNOWN。

输入示例：
2 3 6
1 0 3
0 0 0
1 1
1 2
1 3
2 1
2 2
2 3

输出示例：
1
2
3
Unknown
Unknown
Unknown
'''


def func2(data, query, n, m):
    result = []

    row_num_cont = {x: [] for x in range(n)}
    col_num_cont = {x: [] for x in range(m)}

    for i in range(n):
        for j in range(m):
            if data[i][j] != 0:
                row_num_cont[i].append((i, j, data[i][j]))
                col_num_cont[j].append((i, j, data[i][j]))
            else:
                data[i][j] = '#'
    flag = True
    while flag:
        flag = False
        for i in range(n):
            if len(row_num_cont[i]) == m:
                continue
            if len(row_num_cont[i]) >= 2:
                i1, j1, v1 = row_num_cont[i][0]
                i2, j2, v2 = row_num_cont[i][1]
                d = (v2 - v1) / (j2 - j1)
                for j in range(m):
                    if data[i][j] == '#':
                        cur = (j - j1) * d + v1
                        data[i][j] = cur
                        row_num_cont[i].append((i, j, cur))
                        col_num_cont[j].append((i, j, cur))
                        flag = True
        for j in range(m):
            if len(col_num_cont[j]) == n:
                continue
            if len(col_num_cont[j]) >= 2:
                i1, j1, v1 = col_num_cont[j][0]
                i2, j2, v2 = col_num_cont[j][1]
                d = (v2 - v1) / (i2 - i1)
                for i in range(n):
                    if data[i][j] == '#':
                        cur = (i - i1) * d + v1
                        data[i][j] = cur
                        col_num_cont[j].append((i, j, cur))
                        row_num_cont[i].append((i, j, cur))
                        flag = True

    for q in query:
        x, y = q
        if data[x][y] != '#':
            result.append(int(data[x][y]))
        else:
            result.append('Unknown')

    return result


if __name__ == '__main__':
    input_str = input()
    n, m, q = [int(x) for x in input_str.split()]

    data = [[0] * m for i in range(n)]
    query = []
    for i in range(n):
        input_str = input()
        tmp = [int(x) for x in input_str.split()]
        data[i] = tmp
    for i in range(q):
        input_str = input()
        tmp = [int(x) - 1 for x in input_str.split()]
        query.append(tmp)
    result = func2(data, query, n, m)
    for i in result:
        print(i)
```

以上代码能够 AC。

---

<div data-hk-top-pages="5"> </div>
