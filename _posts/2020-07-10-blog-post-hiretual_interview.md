---
title: '社招前端面试题汇总'
date: 2020-07-10
permalink: /posts/2020/07/hiretual_interview/
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

## Hiretual 前端面试算法题总结

非本人面试，仅做记录。


- 算法题一道题，大概意思是一个图上找一个最短路径问题，但是路线上的地点的海拔只能先增加后降低。dfs的时候找判断一下邻居节点的可行性就ok。
```python
''' 
You decided to run every night when you see your roomate is more charming than you because he/she works out regularly.
Now you have a dictionary of places in Beijing. It's in the form of {location: elevation}. And an array of distances you find on Baidu Map connecting each places. Please find the length of the shortest route on which you can run completely uphill then completely downhill. Assume you live in "Huilongguan".
elevations = {"Huilongguan": 5, "Chaoyang Park": 25, "National Stadium": 15, "Olympic Park": 20, "Tsinghua University": 10}
paths = {
    ("Huilongguan", "Chaoyang Park"): 10,
    ("Huilongguan", "National Stadium"): 8,
    ("Huilongguan", "Olympic Park"): 15,
    ("Chaoyang Park", "Olympic Park"): 12,
    ("National Stadium", "Tsinghua University"): 10,
    ("Olympic Park", "Tsinghua University"): 5,
    ("Olympic Park", "Huilongguan"): 17,
    ("Tsinghua University", "Huilongguan"): 10
}
For this set of data, the shortest valid path would be "Huilongguan" -> "National Stadium" -> "Tsinghua University" -> "Huilongguan", with a distance of 28.
'''
class solution:
    def solve(self,dists,highs,begin,end):
        self.g={}
        for names,d in dists.items():
            u,v=names
            if u not in g:
                self.g[u]={}
            self.g[u][v]=d

        self.h=highs        
        self.result=1000000000000

        self.begin=begin
        self.end=end
        begin_h=self.h[begin]
        for nei,d in self.g[begin].items():
            tmp_h=self.h[nei]
            if tmp_h>begin_h:
                self.dfs(nei,d,1)
        return self.result
    
    def dfs(self,cur_node, pre_dist, direction):
        if cur_node==self.end:
            self.result=min(self.result,pre_dist)
            return
        neibors=self.g[cur_node]
        if len(neibors)==0:
            return 
        cur_h=self.h[cur_node]
        if direction==1:#upfill
            for nei,d in neibors.items():
                tmp_h=self.h[nei]
                if tmp_h>cur_h:
                    self.dfs(nei,pre_dist+d,1)
                elif tmp_h<cur_h:
                    self.dfs(nei,pre_dist+d,-1)
        else:
            for nei,d in neibors.items():
                tmp_h=self.h[nei]
                if tmp_h<cur_h:
                    self.dfs(nei,pre_dist+d,-1)
        return
```

---

<div data-hk-top-pages="5"> </div>
