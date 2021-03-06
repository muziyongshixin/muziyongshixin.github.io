---
title: '2021秋招提前批-百度小度云平台部-面试总结'
date: 2020-07-21
permalink: /posts/2020/07/baidu_xiaoducloud_interview/
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
      <visited_a href="#" display:inline>你是第<span data-hk-page="current"> - </span>位访客~</visited_a>
      <visited_p class="top">٩(๑^o^๑)۶</visited_p>
      <visited_p class="bottom">Σ(っ °Д °;)っ被你发现了！</visited_p>
</div>
<img align="center" width="100" src="{{ site.url }}/images/static/take_me.gif" alt="" display:inline>
</div>
---

## 百度 小度云平台部 视觉部门面试总结

部门概况： 主要是做小度音响上的摄像头相关的算法，例如人脸相关的，儿童年龄检测，手势相关的，human pose 相关，以及一些安防相关的视觉算法。部门目前有 5 个人，3 个人负责算法，2 个人负责工程 sdk 的开发和封装等。算法一部分在端上，一部分在云上。

## 一面 （2020 年 7 月 21 日）

一面面试官是大数据的，面到一半直接说方向不匹配然后换了一个视觉的面试官来面试。

## 二面 （2020 年 7 月 21 日）

- 自我介绍
- 介绍在腾讯实习的做的项目，技术实现等
- 介绍最近的一个项目，以及在项目中承担的角色，用到的算法等
- 简单介绍了一下发表的论文
- 视觉小样本的问题，怎么解决
  - 数据增强，flip，rotation，random crop，对比度饱和度等调整
  - 针对样本不平衡的问题，重采样，weighted loss 等
  - 外部数据做弱监督
- 缓解过拟合的方法

  - Dropout，BN，数据增强，参数正则化，验证集上的 early stop

- 算法题一

```python
''' n行m列的棋盘格，每次只能向下或者向右走一步，问从左上角到右下角有多少种走法。'''
def func(n, m): # 方法一，动态规划
    dp = [[0] * m for i in range(n)]
    dp[0][0] = 1
    for j in range(1, m):
        dp[0][j] = 1
    for i in range(1, n):
        dp[i][0] = 1

    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]

def func_by_math(n,m): # 方法二，数学上的组合问题，总共需要走n+m-2步，其中有n-1步是向下走的，所以组合的种类就是从n+m-2中选择n-1，即C_{n+m-2}^{n-1}
    '''
    return C^{n-1}_{n+m-2}
    '''
    a=1
    b=1
    for i in range(n-1):
        a*=(n+m-2-i)
        b*=(i+1)
    return a//b

if __name__ == '__main__':
    n = int(input())
    m = int(input())
    result = func(n, m)
    print(result)

```

- 算法题二

```python
'''给定5个数，每次只能给其中的四个数+1，问最少操作多少次最终五个数会相同'''

'''每次给四个数加一相当于每次只给一个数减一，所以最后结果是所有数和最小的数的差的和'''

def func(nums):
    a=min(nums)
    cache=[n-a for n in nums]
    return sum(cache)
```

## 三面 （2020 年 7 月 24 日）

- 小度音响的很多功能需要视觉技术支持，但是会消耗大量的资源，针对这个问题怎么优化？

  - 针对一些使用频次不高同时对时延要求不高的需求，类似于年龄检测，距离检测这种可以通过匿名化之后，将数据返回给后台进行计算，从而降低端侧的性能需求
  - 针对一些实时性需求较高的应用，例如手势识别等，可以采用模型压缩、蒸馏、剪枝等策略优化神经网络的计算量，从而

- 小度音响需要做儿童的年龄的识别，从而进行个性化的内容推荐，但是现在不是很准，有没有什么优化策略？

  - 首先基于初步的年龄段分类结果进行内容的推送，然后可以根据儿童自己的感兴趣的内容进行点击结果的反馈进行修正。
  - 或者根据音响的语音功能可以收录儿童的声音，可以根据音色等信息进行一些多模态的融合。还有一些说话的逻辑以及流利程度也一定程度上能够判断儿童的年龄。
  - 或者可以设计一些简单的题目，根据不同年龄段儿童的智力可以设计一些识物、算术题或者成语意义分辨等方式做到间接的年龄估计。

- 对未来的职业规划是怎样的，3-5 年甚至更久的。

- 你以后想创业的话，你觉得你会做什么方向？

  - AI 教育和 AI 医疗相关

- 谈谈 AI 医疗领域除了一些看片子的应用，还有哪些应用？

  - 动脉血管 3D 建模，自动测量血管直径
  - 基于手指姿态估计的帕金森诊断系统
  - 基于人体姿态估计的运动康复评价系统

- 对于一个给聋哑人使用的 AI 程序，它包含手语识别，语音识别，文字转换，语音或者手语合成，你觉得哪个部分最难

  - 我觉得手语的识别应该是最难的部分，因为涉及到光照，遮挡，颜色，不同人的体型大小不同，手语的速度也不同，所以对这个的手势的识别应该是最困难的部分。
  - 而手语识别之后根据文本语义进行手语或者语音合成应该是较为简单的步骤，可以根据特定的硬编码规则类似于电影或者游戏里的人物动作渲染来实现。

- 那如果手势识别这个做的不好，你有什么改进的思路吗？

  - 在针对每个单独语素的识别之后可以考虑一下语义的前后的 context 做一些后处理的纠错，类似于打字的时候我们可能会按错一些按键，但是基于前后的语义信息我们可以根据一些正确的结果去推测出不准确或者错误的部分，从而提高整个程序的鲁棒性。

- 那这个策略是不是就无法做到实时的操作了？

  - 我觉得在对于聋哑人交流这个场景下，能够在每句话的手势做完一两秒内返回结果应该还是可以接受的。毕竟连实时的机器同声传译也也一定的延迟和纠正操作，所以我觉得这个延迟应该是可以接受的。

* 问面试官问题
  - 小度音响现在带屏幕的大概会占到 50%的比例，然后日活在几百万的量级，和天猫精灵，小爱同学基本三分天下
  - 面试官表示在北京买房也就五六年的问题，不用太紧张，要乐观一点(￣(∞)￣)
  - 他主要负责小度的 OS，大数据分析，视觉算法团队，大概有 50 人左右向他汇报，目前视觉团队大概 10 个人左右。
  - 部门三块的话今年整体 HC 大概 20 多人，AIDU 的 offer 的话只占很小一部分。HR 会在一周后通知结果。
  - 面试官建议还是需要自己学习下 C++，工作中需要从训练到部署到 SDK 开发。

---

<div data-hk-top-pages="5"> </div>
