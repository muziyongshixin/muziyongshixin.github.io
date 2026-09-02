# 交互式终端主页 —— 数据构建与本地预览
#
# 页面本身零构建：index.html 直接就是产物。需要「构建」的只有 assets/ 下
# 那三份数据，它们由 tools/ 里的脚本从照片、_posts、论文来源编译而来。

PY  ?= python3
PORT ?= 8420

.PHONY: help data avatar posts papers papers-force test serve check clean

help:
	@echo "make data          重新生成全部数据（头像 + 博客 + 论文）"
	@echo "make avatar        换了照片后重跑（tools/source.jpg -> assets/avatar.js）"
	@echo "make posts         写了新博客后重跑（../_posts -> assets/content.js）"
	@echo "make papers        加了新论文后重跑（抓原文 -> assets/papers.js，带缓存）"
	@echo "make papers-force  忽略缓存重抓全部论文"
	@echo "make test          冒烟测试：真实 DOM 里跑一遍所有命令"
	@echo "make serve         本地预览 http://localhost:$(PORT)"
	@echo "make check         提交前跑一遍：数据 + 测试"

data: avatar posts papers

avatar:
	$(PY) tools/pixelize.py

posts:
	$(PY) tools/extract_posts.py

papers:
	$(PY) tools/fetch_papers.py

papers-force:
	$(PY) tools/fetch_papers.py --force

test:
	@command -v node >/dev/null || { echo "需要 node"; exit 1; }
	@[ -d node_modules ] || npm install --no-save jsdom@24
	node tools/smoke-test.mjs

serve:
	@echo "http://localhost:$(PORT)  (Ctrl+C 退出)"
	@$(PY) -m http.server $(PORT)

check: posts test

clean:
	rm -rf tools/.paper-cache node_modules
