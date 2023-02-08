---
title: '2021秋招提前批-腾讯微信开放平台-面试总结'
date: 2020-07-21
permalink: /posts/2020/07/weixin_kaifangpingtai/
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

## 腾讯 微信开放平台 公众平台中心面试总结

部门概况： 微信的开放平台，负责小程序公众号等里面的一些视觉技术。
例如扫一扫里的识图，扫二维码，OCR 技术等。（识图大概月活到淘宝拍立淘的 1/3，去年 12 月才上线的功能）
还有一些以图搜索，给定图片可以找到该图片在公众号等地方的出处。
部门在北京和广州都有，北京的话大概有 4 个员工左右是做 cv 相关的，广州的大概有 2 个左右的正式员工做 cv 相关的技术，其他的大概有 10 多个人是做后台开发。面试的话正常会有一面，组长面，总监面，微信面试委员会面，总共五六轮。

## 一面 （2020 年 7 月 21 日）

- 自我介绍，主要是介绍了一下自己做的论文以及一些项目，大概四十分钟左右，没有问啥具体的问题。
- 腾讯的大咖计划是要看 HC 的，部门推荐过去能不能评上还要看有没有名额和审核。

## 二面 （2020 年 7 月 23 日）

- 主要也是先介绍了两篇最近自己的工作，然后针对其中的一些实验结果进行提问。感觉面试官对实验的训练细节等东西比较关注，感觉面试官对深度学习的细节知识掌握的比较全面。

- BN 的作用，为什么能够防止过拟合

  - BN 使得每个 sample 的 feature 受到 batch 内部其他数据的影响，相当于加入了一些噪声，有一定的数据增强等的效果，所以能够一定程度提高泛化能力。

- 如果使用了 BN 仍然有过拟合问题，怎么调整。

  - 面试官说可以调整的方面有两个方面，一个是 batch size 的大小，一个是针对 BN 的 lr 和 momentum 的设置
  - batch size 的话应该是减小，从而增大 BN 后的 feature 的 variance，从而使得模型能够探索更多的位置
  - lr 应该是增大，momentum 应该减小，同样也是为了增大 BN 输出的 feature map 的 variance，从而能够使得模型泛化能力增大
  - 面试官说这种策略在自己收集的小数据集上的泛化效果较好，在 imagenet 或者 cifar 这种数据集上效果并不明显。

- 为什么 BN 能够加速模型的收敛

  - 在使用是 sigmoid 激活函数的时候使用 BN 能够将数据从饱和区拉到非饱和区，从而避免出现很小的梯度。
  - 使用 BN 之后可以使用较大的 lr，模型也不会乱飞，从而一定程度加速模型的训练。

- 对 resnet 后来的发展有啥了解

  - [ResNext 和 Res2Net](https://blog.csdn.net/wh8514/article/details/106725142/)
  - [ResNest](https://blog.csdn.net/Guo_Python/article/details/105646838)

- ResnetV2 和 ResnetV1 有啥区别和改进，为啥 v1 只能到 100 多层，而 v2 能够到 1000 多层。
  - V2 的 identity 分支梯度回传更合理，可以学到的信息更多
  - [v1 和 v2 的区别](https://blog.csdn.net/chenyuping333/article/details/82344334)


---
卒。。。

---

<div data-hk-top-pages="5"> </div>
