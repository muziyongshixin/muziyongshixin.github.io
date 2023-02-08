---
title: '2021秋招提前批-百度视觉技术部-面试总结'
date: 2020-07-07
permalink: /posts/2020/07/baidu_visiongroup_interview/
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

## 百度 视觉技术部 面试总结

部门概况，视觉技术部在百度内部主要是作为一个技术中台的作用，有一些对内的技术支持，以及一些对外的 ToB 业务。视觉技术部总体规模 150 多人左右，基本覆盖计算机视觉的各个方面。一面面试官是视频编辑的部门，主要是做一些视频的 style transfer 等，以及一些 low level 的超分辨，去噪等工作，小部门大概 10 个人左右。部门在北京，上海，深圳都有分布。

## 一面

- 自我介绍

- 介绍了一篇论文

- 介绍了在谷歌冬令营的项目，主要做了哪些工作，怎么实现的。

- 聊了一下搜索引擎里的大规模的图片检索的技术和存在的问题，改进的空间。

- 了解 GAN 嘛，讲一下 cycle GAN 能够成功的原因

  - 有一些浅显的了解，cycle GAN 是相当于有一个 A -> B -> A' 的过程，其中 A'应该和 A 足够相似，相当于起到了一个自监督的效果。所以能够有一些不错的效果。

- 一道算法题，写一下 detection 里的 NMS，可以直接写一维的情况。
  - 矩阵的操作不太会写，直接写的 naive 的版本。

```python
import numpy as np

def iou_score(box1, box2):
    min_1, max_1 = min(box1), max(box1)
    min_2, max_2 = min(box2), max(box2)

    i = max(min(max_1, max_2) - max(min_1, min_2), 0)
    u = max(max(max_1, max_2) - min(min_1, min_2), 0)
    return i / u

def nms(boxes, confidence, threshold):

    cids = np.argmax(confidence, reverse=True)
    suppressed = [0] * len(boxes)

    for i, bid in enumerate(cids):
        if suppressed[bid] == 1:
            continue
        cur_box = boxes[bid]
        for j in range(i + 1, len(cids)):
            if suppressed[cids[j]] == 1:
                continue
            tmp_box = boxes[cids[j]]

            iou = iou_score(cur_box, tmp_box)
            if iou > threshold:
                suppressed[cids[j]] = 1

    keep = [i for i in range(len(boxes)) if suppressed[i] == 0]
    return keep
```

- 问面试官一些问题，视觉技术部面向的业务方向，人员规模等。

面试总体很快，大概 40 分钟结束。

## 二面

二面主要是在聊天，没有啥很细节的问题，大概 30 分钟就完了。

- 对 GAN 的了解多不多，知道有哪些 loss？
- 对视频相关的工作了解的多不多
- 目前有没有在投其他的公司
- 来百度的意愿有多大
- 对以后做什么方向有没有很强的要求

## 三面

三面的面试官应该是部门经理，基本也是聊天的形式，主要了解了一下以前做过的哪些事情，以后的工作意愿。
面试官说北京户口可能概率不大，但是每年情况不一样，以 HR 的沟通为准。面试大概 20 多分钟，比较快。(面试时间 2020 年 7 月 10 日)

---

<div data-hk-top-pages="5"> </div>
