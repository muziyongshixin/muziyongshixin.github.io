---
title: '2021秋招-微信-技术架构部-面试总结'
date: 2020-08-14
permalink: /posts/2020/08/weixin_jishujiagou/
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

## 微信-技术架构部-视觉应用组 面经

部门概况：团队主要是作为微信的计算机视觉的技术中台，涉及到视频特效，文字识别， 图片分类，手势识别等主要用在视频号，表情等业务下。
部门算法加上工程团队总共 20 多人左右，团队都在深圳，但是可以发北京的 offer，需要在深圳工作一年熟悉一下团队之后可以回北京。工作时长一般是 10-10-5。在版本上线前可能会有短暂加班。

## 一面（2020 年 8 月 14 日）

- 自我介绍
- 讲一片论文，针对论文中的一些问题进行提问
- 实际业务场景中有年龄识别需求，但是面临数据的长尾分布问题，比如小孩和老人的图片较少，如何解决。
  - 数据方面，可以用 GAN face edition 这种方法生成一些假的数据进行数据扩充，或者比如婴幼儿网站或者老年人网站这种爬取数据。还可以利用类似带屏幕智能音箱这种可以带语音输入的设备，通过语音信息进行 labeling。
  - 训练方面，可以使用数据增强策略，重采样的策略以及 Boosting 这种策略去针对困难样本进行学习
  - Loss 方面可以使用 weighted loss 或者 Focal Loss 这种来提升模型性能。
- Focal Loss 为啥能够起到这样的效果？

  - [FocalLoss](https://blog.csdn.net/like_study_cat/article/details/107750540)

- 算法题

```python

'''
实现softmax，需要注意溢出的问题
'''
import numpy as np

def softmax(data):
    data = data - data.max()
    n_exp = np.exp(data)
    sum_exp = n_exp.sum()
    return n_exp / sum_exp

if __name__ == "__main__":
    data=np.array([1,2,3,4])
    print(softmax(data))

```

- 算法题

```python
'''
实现一个鲁棒的开根号函数
'''

def sqrt(n):
    if n < 0:
        return -1
    if n == 0:
        return 0

    pre = n
    while True:
        if abs((pre**2 + n) / (2 * pre) - pre) <= (10**-6):
            break
        pre = (pre**2 + n) / (2 * pre)
    return pre

```

<div data-hk-top-pages="5"> </div>
