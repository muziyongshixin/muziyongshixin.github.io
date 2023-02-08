# 本项目主要用于构建个人主页

## 左边栏的个人信息修改

修改 \_config.yml 中的对应条目

## 设置导航栏条目

- 修改 \_data\navigation.yml， 其中每个 title 表示标题栏上的一个项目
- 其中 url: /xxxx_pagename/ 对应\_pages 目录下对应的 xxxx_pagename.md 文件

## \_pages 目录

主要页面都在其中，每个 markdown 文件就是一个页面

## \_posts 目录

包含 blog 中的所有文章信息，其中的命名格式为：
YYYY-MM-DD-blog-post-xxxxxxnamexxxxx.md

## files 文件夹

其中可以包含一些文件。例如论文文件，个人简历 PDF 文件等，可以通过类似如下方式引用:

"[Click Here to Download My Resume] ({{ site.url }}/files/pdf/lyz_resume.pdf)"

## images 文件夹

其中保存页面中的图片文件。可以通过如下方式引用：

```js
<br />
<img align="center" width="1000" src="{{ site.url }}/images/static/image_ste.jpg" alt="...">
<br />
```


## 谷歌网站访问分析以及统计人数的js文件路径存放在了如下文件中，从而自动导入每一个页面

_includes\footer\custom.html
该文件中还包含了人数统计的button的css配置


## 页面访问人数统计
通过如下标签可以在页面上添加一个当前访问人数统计的button
``` js
<div class="button01">
            <visited_a href="#" display:inline><span id="busuanzi_container_site_pv">你是第<span id="busuanzi_value_site_pv"></span>位访客~</span></visited_a>

      <visited_p class="top">٩(๑^o^๑)۶</visited_p>
      <visited_p class="bottom">Σ(っ °Д °;)っ被你发现了！</visited_p>
</div>
```
