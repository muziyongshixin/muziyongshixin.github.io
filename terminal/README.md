# Terminal Homepage

`liyongzhi.xyz/terminal/` 的源码：一个命令行式的个人主页，访客可以敲命令浏览论文、
经历、博客，也可以直接向 AI 提问。与传统版共用同一个仓库、同一次部署。

## 它是怎么跑起来的

页面本身**零构建**：`index.html` 就是产物，不需要打包器、不需要 npm build。
所有需要预处理的东西都提前编译进了 `assets/` 下的三个数据文件：

| 文件 | 内容 | 由谁生成 | 输入 |
| --- | --- | --- | --- |
| `assets/avatar.js` | 像素肖像（亮度层 + 16 色调色板） | `tools/pixelize.py` | `tools/source.jpg` |
| `assets/content.js` | 博客元信息、摘要、章节结构 | `tools/extract_posts.py` | `../_posts/*.md` |
| `assets/papers.js` | 论文原文（摘要 + 各章节摘录） | `tools/fetch_papers.py` | arXiv / CVF / PDF |

这样做的原因是浏览器端受 CORS 限制：arXiv、Semantic Scholar 都不返回跨域头，
运行时根本取不到；而一篇论文的 HTML 有 150–400KB，即便取到，每次问答都要重新
prefill，KV cache 直接失效。构建期抓一次、抽成几 KB 的摘录，运行时零延迟零依赖。

## 日常维护

```bash
cd terminal

make posts     # 写完新博客后（读 ../_posts）
make papers    # 加了新论文后（改 tools/fetch_papers.py 里的 PAPERS 清单）
make avatar    # 换了照片后（替换 tools/source.jpg，按需调 CROP）
make test      # 提交前跑一遍
make serve     # 本地预览
```

### 加一篇论文

1. 在 `index.html` 的 `PAPERS` 数组里加一条（id / title / venue / year / authors / tags / tldr / links）。
2. 在 `tools/fetch_papers.py` 的 `PAPERS` 清单里加同样的 id，尽量给上 `arxiv` 或 `pdf`。
3. `make papers` —— 脚本会按 arXiv HTML → PDF → OpenReview/CVF → arXiv API → OpenAlex
   的顺序尝试，并打印每篇拿到了几节正文。没有 arXiv 号也没关系，它会用标题去
   OpenAlex 反查 DOI，自动发现预印本。
4. `make test`。

`dl.acm.org` 和 `techrxiv.org` 有 Cloudflare 人机验证，抓不到，这两个源的论文
请提供 arXiv 或作者主页的 PDF 作为替代。

### 换照片

替换 `tools/source.jpg`，调 `tools/pixelize.py` 顶部的 `CROP`（方形/竖版裁剪框）
和 `NW, NH`（像素网格，现在 108×144），然后 `make avatar`。脚本会在终端打印
ASCII 预览，方便确认构图。

## AI 与 Agent

模型走 ModelScope 的 API-Inference（OpenAI 兼容协议，浏览器可直连，实测放行 CORS）。

站点内置了一个 token，访客开箱即用。它在 `index.html` 的 `VAULT` 里以 XOR + Base64
混淆存储，运行时现解现用，不写进变量也不落 localStorage。

**必须清楚：这是混淆，不是加密。** 解码逻辑随代码发给浏览器，打开 DevTools 看
Network 里的 Authorization 头就能拿到明文——前端做不到真正的密钥保护。混淆挡住的是
GitHub 密钥扫描和批量爬虫的正则匹配，那才是明文 token 最常见的泄露途径。

真正起作用的是另外三层：

- **域名白名单**：只有 `VAULT.hosts` 里的域名才组装出 token，别人整站复制走也直接用不了
- **每日配额**：单个浏览器每天 40 次，防止一个人刷爆共享额度；访客配了自己的 token 就不受限
- **可随时作废**：被盗刷就去 ModelScope 撤销，换新 token 重跑一次脚本

### 模型

`MODELS` 里只保留了实测跑通的四个（`model` 命令可切换）：

| 模型 | 首字正文 | 备注 |
| --- | --- | --- |
| `Qwen/Qwen3.8-Flash-Next` | 1.3s | 默认，带思考 |
| `Qwen/Qwen3.8-27B` | 4.3s | 更大，带思考 |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | 1.8s | 带思考 |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | 0.6s | 不思考，最快 |

前三个是 reasoning 模型：流式时先吐 `reasoning_content`，`content` 要等思考结束
才出现。前端两路都接，思考过程实时显示在灰色区域里，正文一到就折叠成
「思考了 N 字」，点击可展开。**只接 `content` 会导致模型思考的那一两秒页面完全没有反馈。**
思考内容不写入对话历史。

`DeepSeek-V3.1`、`Qwen3-32B`、`GLM-4.5` 在 ModelScope 上返回
"has no provider supported"，已从列表移除；`LLM.model` 会校验 localStorage
里存的模型是否还在列表中，不在就回退到默认。

轮换 token：

```bash
python3 tools/obfuscate_key.py ms-新的token   # 写入并自检
python3 tools/obfuscate_key.py --check        # 只验证当前嵌入的能否解回
```

脚本只在 `const VAULT = {...}` 这一段内替换。早期版本用全局正则 `d:\s*"..."`，
结果命中了论文数据里的 `id:"adaptive-video-distillation"`，把论文 id 覆盖成了密文——
所以务必保留那个锚点，改完跑 `make test`。

Agent 部分不依赖 function calling（ModelScope 上各模型支持程度不一），走的是文本协议：
模型需要工具时把回复的第一个字符写成 `<tool>{"name":...,"args":{...}}</tool>`，
前端解析后执行，结果以 `[工具结果 · 工具名]` 回灌，最多连续三次。

四个工具：`site_search`（站内检索，本地）、`read_paper`（论文原文，本地）、
`read_post`（博客章节，本地）、`paper_search`（OpenAlex 学术检索）、`wiki`（维基百科）。
只收录明确放行 CORS 的外部服务。

## 上下文策略

对话历史只追加不改写，每次请求发送 `[system, ...history, newUser]`，前缀逐轮增长
且保持稳定，服务端的 prefix cache 才能命中。历史超过 32K 字符时一次砍到 60%，
而不是每轮丢一对——那样每轮前缀都变，缓存必然失效。

命令输出也会作为 assistant 消息进入上下文（截断到 1800 字符），这样访客说
「第三篇讲什么」时模型知道他在看什么。但真正发送给模型的原文（比如 `explain`
注入的论文摘录）不截断，否则历史与实际发送内容不一致，同样会破坏缓存。

## 测试

`make test` 会在 jsdom 里真实加载页面，跑一遍全部命令，覆盖 reader 视图、
中文输入法组字、四个工具，并用 mock 掉的流式接口把 agent 循环完整走一遍
（工具调用链路、坏 JSON、网络故障），还会断言界面上的统计数字与数据一致、
第二次请求复用了第一次的消息前缀。

`node --check` 只验证语法，抓不到未定义引用这类错误，改完务必跑 `make test`。

## 部署

无需任何额外配置。这个目录随 Jekyll 站点一起发布，`terminal/tools` 已在
`_config.yml` 的 `exclude` 里，不会进入 `_site`。push 到 master 即上线。
