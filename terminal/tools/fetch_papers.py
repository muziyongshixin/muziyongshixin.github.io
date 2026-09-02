#!/usr/bin/env python3
"""构建期抓取论文全文，编译成前端可直接读的结构化数据。

为什么放在构建期：arXiv / CVF / OpenReview 都不返回 CORS 头，浏览器根本
取不到；而且一篇论文的 HTML 有 150-400KB，运行时拉下来光 prefill 就废掉
了 KV cache。构建期抓一次、抽成几 KB 的章节摘录，运行时零延迟零依赖。

抓取优先级：arXiv HTML 全文 → CVF / OpenReview 摘要页 → arXiv API 摘要
→ OpenAlex 摘要 → 放弃（前端会提示该篇需要手工补要点）。

用法：python3 tools/fetch_papers.py [--only <paper-id>] [--force]
"""
import argparse, json, pathlib, re, sys, time
import requests
from bs4 import BeautifulSoup

HERE  = pathlib.Path(__file__).resolve().parent.parent
OUT   = HERE / "assets" / "papers.js"
CACHE = HERE / "tools" / ".paper-cache"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HEADERS = {"User-Agent": UA}
TIMEOUT = 40

SECTION_CHARS = 900     # 每节保留的正文字符数
MAX_SECTIONS  = 14
ABSTRACT_CHARS = 1600

# 论文清单与 index.html 的 PAPERS 保持一致。
# pdf 链接来自作者核对过的 Google Scholar 详情页；dl.acm.org 与 techrxiv 有
# Cloudflare 反爬（403），这几篇靠 arXiv / OpenAlex 兜底。
PAPERS = [
    dict(id="adaptive-video-distillation", title="Adaptive Video Distillation: Mitigating Oversaturation and Temporal Collapse in Few-Step Generation",
         arxiv="2603.21864", pdf="https://arxiv.org/pdf/2603.21864"),
    dict(id="narrative-weaver",            title="Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning",
         arxiv="2603.06688", pdf="https://arxiv.org/pdf/2603.06688"),
    dict(id="autocut",                     title="AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation",
         arxiv="2603.28366", pdf="https://arxiv.org/pdf/2603.28366"),
    dict(id="generative-recommendation-ads", title="Generative Recommendation for Large-Scale Advertising",
         arxiv="2602.22732"),   # dl.acm.org/doi/pdf/10.1145/3770855.3818424 反爬
    dict(id="acpo",                        title="ACPO: Adaptive Credit Policy Optimization via Fine-Grained Surrogate Entropy",
         arxiv="2607.03126", pdf="https://arxiv.org/pdf/2607.03126"),
    dict(id="rec-as-generation",           title="Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Scale",
         arxiv="2606.25496", pdf="https://arxiv.org/pdf/2606.25496"),
    dict(id="fbos-rl",                     title="FBOS-RL: Feedback-Driven Bi-Objective Synergistic Reinforcement Learning",
         arxiv="2605.20256", pdf="https://arxiv.org/pdf/2605.20256"),
    dict(id="universal-discrete-tokenizers", title="Universal Discrete Tokenizers: Principles, Applications, and Future Directions"),
                                            # www.techrxiv.org/doi/pdf/... 反爬
    dict(id="imagine-multi-agent",         title="Imagine: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning",
         arxiv="2510.14406", pdf="https://arxiv.org/pdf/2510.14406"),
    dict(id="ecom-instance-pretrain",      title="Learning Instance-Level Representation for Large-Scale Multi-Modal Pretraining in E-commerce",
         arxiv="2304.02853", pdf="https://arxiv.org/pdf/2304.02853"),
    dict(id="stvg-consistency",            title="Embracing Consistency: A One-Stage Approach for Spatio-Temporal Video Grounding",
         arxiv="2209.13306", openreview="NzFtM5Pzvm",
         pdf="https://proceedings.neurips.cc/paper_files/paper/2022/file/bc18c538d983cea434f9281148d43e1e-Paper-Conference.pdf"),
    dict(id="catadioptric-pose",           title="Learning 3-D Human Pose Estimation from Catadioptric Videos",
         pdf="https://www.ijcai.org/proceedings/2021/0118.pdf"),
    dict(id="adv-hash-search",             title="Efficient Fine-Grained Visual-Text Search Using Adversarially-Learned Hash Codes",
         pdf="https://pkumyd.github.io/paper/icme21_liyz.pdf"),
    dict(id="multiview-crowd",             title="Learning Factorized Cross-View Fusion for Multi-View Crowd Counting",
         pdf="https://pkumyd.github.io/paper/icme21_zhenglf.pdf"),
    dict(id="visual-semantic-matching",    title="Visual-Semantic Matching by Exploring High-Order Attention and Distraction",
         cvf="https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Visual-Semantic_Matching_by_Exploring_High-Order_Attention_and_Distraction_CVPR_2020_paper.html",
         pdf="https://pkumyd.github.io/paper/CVPR2020_LYZ.pdf"),
    dict(id="spectral-medical",            title="Spectrally-Enforced Global Receptive Field for Contextual Medical Image Segmentation and Classification",
         pdf="https://pkumyd.github.io/paper/ICME20_LYZ.pdf"),
    dict(id="video-steganography",         title="High-Capacity Convolutional Video Steganography with Temporal Residual Modeling",
         arxiv="1806.02941", pdf="https://arxiv.org/pdf/1806.02941"),
                                            # dl.acm.org/doi/pdf/10.1145/3323873.3325011 反爬；
                                            # Scholar 把它的 arXiv 预印本另列为一条，实为同一工作
]

SKIP_HEADINGS = re.compile(
    r"^(references?|bibliography|acknowledg|appendix|supplementary|abstract)\b", re.I)


def get(url, **kw):
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    return r


def clean(t):
    t = re.sub(r"\[\d+(,\s*\d+)*\]", "", t)        # 引用角标
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ---------------------------------------------------------------- sources
def from_arxiv_html(arxiv_id):
    """arXiv 2023 年末起为多数论文生成 HTML 版，是唯一能拿到全文结构的免费来源。"""
    r = requests.get(f"https://arxiv.org/html/{arxiv_id}", headers=HEADERS, timeout=TIMEOUT)
    if r.status_code != 200 or "<html" not in r.text[:400].lower():
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    if soup.select_one("#no-html-message") or len(r.text) < 20000:
        return None

    abstract = ""
    ab = soup.select_one(".ltx_abstract")
    if ab:
        abstract = clean(ab.get_text(" "))[:ABSTRACT_CHARS]

    sections = []
    for sec in soup.select("section.ltx_section"):
        h = sec.select_one("h2, h3, .ltx_title")
        title = clean(h.get_text(" ")) if h else ""
        if not title or SKIP_HEADINGS.match(title):
            continue
        for bad in sec.select(".ltx_bibliography, .ltx_appendix, table"):
            bad.decompose()
        body = clean(sec.get_text(" "))
        if title and body.startswith(title):
            body = body[len(title):].strip()
        if len(body) < 120:
            continue
        sections.append({"title": title[:120], "text": body[:SECTION_CHARS]})
        if len(sections) >= MAX_SECTIONS:
            break

    if not abstract and not sections:
        return None
    return {"abstract": abstract, "sections": sections,
            "source": f"https://arxiv.org/abs/{arxiv_id}", "via": "arxiv-html"}


def from_arxiv_api(arxiv_id):
    r = get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}")
    soup = BeautifulSoup(r.text, "html.parser")
    e = soup.find("entry")
    if not e or not e.find("summary"):
        return None
    return {"abstract": clean(e.find("summary").get_text())[:ABSTRACT_CHARS], "sections": [],
            "source": f"https://arxiv.org/abs/{arxiv_id}", "via": "arxiv-api"}


def from_openreview(fid):
    r = get(f"https://api2.openreview.net/notes?forum={fid}")
    for n in r.json().get("notes", []):
        ab = (n.get("content", {}).get("abstract") or {}).get("value")
        if ab:
            return {"abstract": clean(ab)[:ABSTRACT_CHARS], "sections": [],
                    "source": f"https://openreview.net/forum?id={fid}", "via": "openreview"}
    return None


def from_cvf(url):
    soup = BeautifulSoup(get(url).text, "html.parser")
    ab = soup.select_one("#abstract")
    if not ab:
        return None
    return {"abstract": clean(ab.get_text(" "))[:ABSTRACT_CHARS], "sections": [],
            "source": url, "via": "cvf"}


SEC_LINE = re.compile(
    r"^\s*(?:(\d{1,2})[\.\)]?\s+|([IVX]{1,5})[\.\)]\s+)"
    r"([A-Z][A-Za-z][A-Za-z \-&/']{2,48})\s*$")
SEC_NAMES = re.compile(
    r"^\s*(?:\d{1,2}[\.\)]\s*)?(introduction|related works?|background|preliminar\w*|"
    r"method\w*|approach|framework|experiment\w*|evaluations?|results?|"
    r"ablation\w*|analysis|discussion|conclusions?|limitations?)\b", re.I)
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"}

# PDF 的字距经常把标题拆开（"RELA TED WORK"），按去空格后的形态还原成规范写法
CANON = ["INTRODUCTION", "RELATED WORK", "RELATED WORKS", "BACKGROUND", "PRELIMINARIES",
         "PROPOSED METHOD", "OUR PROPOSED METHOD", "METHODOLOGY", "METHOD", "APPROACH",
         "EVALUATIONS AND EXPERIMENTS", "EVALUATIONS", "EVALUATION", "EXPERIMENTS",
         "EXPERIMENTAL RESULTS", "RESULTS", "ABLATION STUDY", "DISCUSSION",
         "CONCLUDING REMARKS", "CONCLUSIONS", "CONCLUSION"]
CANON_MAP = {c.replace(" ", ""): c for c in CANON}


def canon_heading(t):
    return CANON_MAP.get(t.upper().replace(" ", ""), t)


def looks_like_heading(s):
    """PDF 抽出来的文本里，正文行很容易被误当成章节标题
    （"model [32] pretrained on MS-COCO [24]…"），所以先按形态过滤一遍。
    注意编号要先剥离再查数字，否则 "1. INTRODUCTION" 会被自己的规则毙掉。"""
    if not (3 <= len(s) <= 46):
        return False
    if s.endswith((".", ",", ";", ":")) or re.search(r"[\[\]()=+@]", s):
        return False
    core = re.sub(r"^\s*(?:\d{1,2}|[IVX]{1,5})[\.\)]?\s+", "", s).strip()
    if not core or re.search(r"\d", core) or SKIP_HEADINGS.match(core):
        return False
    words = core.split()

    if SEC_LINE.match(s) and len(words) <= 6:          # 带编号的正规标题
        return core
    if SEC_NAMES.match(s) and len(words) <= 4:         # 常见章节名，收紧词数避开表头
        return core
    if core.isupper() and len(core) >= 8 and re.fullmatch(r"[A-Z][A-Z \-&']{7,44}", core):
        return core                                     # PDF 里整行大写的小标题
    return False


def normalize_pdf_text(raw):
    for a, b in LIGATURES.items():
        raw = raw.replace(a, b)
    raw = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", raw)   # 行尾断词
    return raw


def pdf_pages(url):
    """PDF 下载后缓存到本地：这些文件动辄一二十兆，重跑时不该再拉一遍。"""
    from pypdf import PdfReader
    import io, hashlib
    cache_dir = CACHE / "pdf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / (hashlib.md5(url.encode()).hexdigest() + ".pdf")
    if f.exists():
        blob = f.read_bytes()
    else:
        blob = get(url).content
        if blob[:4] != b"%PDF":
            return None
        f.write_bytes(blob)
    if blob[:4] != b"%PDF":
        return None
    reader = PdfReader(io.BytesIO(blob))
    return [p.extract_text() or "" for p in reader.pages[:14]]


def from_pdf(url):
    pages = pdf_pages(url)
    if not pages:
        return None
    raw = normalize_pdf_text("\n".join(pages))
    if len(raw) < 800:
        return None

    flat = clean(raw)
    m = re.search(r"abstract[—\-–:\s]*(.{200,2400}?)"
                  r"(?:\b1[\.\s]+introduction|\bindex terms|\bkeywords|\bccs concepts)", flat, re.I)
    abstract = clean(m.group(1)) if m else flat[:ABSTRACT_CHARS]

    # 逐行找章节标题，把正文切开；两栏 PDF 的抽取顺序不完美，但足够定位各节
    lines = raw.splitlines()
    marks, seen = [], set()
    for i, ln in enumerate(lines):
        title = looks_like_heading(ln.strip())
        if not title:
            continue
        key = title.lower().replace(" ", "")
        if key in seen:      # 页眉重复出现的标题只认第一次
            continue
        seen.add(key)
        marks.append((i, canon_heading(title)[:60]))

    sections = []
    for j, (i, title) in enumerate(marks):
        end = marks[j + 1][0] if j + 1 < len(marks) else len(lines)
        body = clean(" ".join(lines[i + 1:end]))
        if len(body) < 150:
            continue
        sections.append({"title": title, "text": body[:SECTION_CHARS]})
        if len(sections) >= MAX_SECTIONS:
            break

    if not sections:   # 章节没识别出来时，退回整段正文，总比没有强
        sections = [{"title": "Body (extracted from PDF)", "text": flat[:SECTION_CHARS * 3]}]
    return {"abstract": abstract[:ABSTRACT_CHARS], "sections": sections,
            "source": url, "via": "pdf"}


def from_openalex(title):
    r = get("https://api.openalex.org/works", params={
        "filter": f"title.search:{re.sub(r'[,:]', ' ', title)[:120]}",
        "per-page": 1, "mailto": "yongzhili@pku.edu.cn"})
    res = r.json().get("results") or []
    if not res:
        return None
    w = res[0]
    inv = w.get("abstract_inverted_index")
    if not inv:
        return None
    pos = {}
    for word, ps in inv.items():
        for p in ps:
            pos[p] = word
    abstract = " ".join(pos[k] for k in sorted(pos))
    return {"abstract": clean(abstract)[:ABSTRACT_CHARS], "sections": [],
            "source": w.get("doi") or w.get("id"), "via": "openalex"}


def discover_arxiv(title):
    """OpenAlex 会给出 arXiv 预印本的 DOI（10.48550/arXiv.xxxx），
    据此能给那些手上没有 arXiv 号的论文自动找到全文页。"""
    try:
        r = get("https://api.openalex.org/works", params={
            "filter": f"title.search:{re.sub(r'[,:]', ' ', title)[:120]}",
            "per-page": 1, "mailto": "yongzhili@pku.edu.cn"})
        res = r.json().get("results") or []
        for w in res:
            for cand in [w.get("doi") or ""] + [
                    (loc.get("landing_page_url") or "") for loc in (w.get("locations") or [])]:
                m = re.search(r"arxiv[./](\d{4}\.\d{4,5})", cand, re.I)
                if m:
                    return m.group(1)
    except Exception as e:
        print(f"      discover_arxiv failed: {type(e).__name__}")
    return None


def fetch(p):
    """按可靠性从高到低尝试各来源。"""
    if not p.get("arxiv"):
        found = discover_arxiv(p["title"])
        if found:
            print(f"      discovered arXiv:{found} via OpenAlex")
            p = {**p, "arxiv": found}

    # 顺序即质量：能切出章节的来源排在只有摘要的来源前面
    attempts = []
    if p.get("arxiv"):
        attempts += [("arxiv-html", lambda: from_arxiv_html(p["arxiv"]))]
    if p.get("pdf"):
        attempts += [("pdf", lambda: from_pdf(p["pdf"]))]
    if p.get("openreview"):
        attempts += [("openreview", lambda: from_openreview(p["openreview"]))]
    if p.get("cvf"):
        attempts += [("cvf", lambda: from_cvf(p["cvf"]))]
    if p.get("arxiv"):
        attempts += [("arxiv-api", lambda: from_arxiv_api(p["arxiv"]))]
    attempts += [("openalex", lambda: from_openalex(p["title"]))]

    for name, fn in attempts:
        try:
            got = fn()
            if got:
                return got, name
        except Exception as e:
            print(f"      {name} failed: {type(e).__name__} {str(e)[:70]}")
        time.sleep(0.4)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    CACHE.mkdir(exist_ok=True)
    out = {}
    for p in PAPERS:
        if a.only and a.only not in p["id"]:
            continue
        cf = CACHE / f"{p['id']}.json"
        if cf.exists() and not a.force:
            out[p["id"]] = json.loads(cf.read_text(encoding="utf-8"))
            print(f"  · {p['id']:<32} cached ({out[p['id']]['via']})")
            continue

        print(f"  · {p['id']:<32} fetching…")
        got, via = fetch(p)
        if not got:
            print(f"      -> 无可用来源，需要手工补要点")
            continue
        got["title"] = p["title"]
        cf.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
        out[p["id"]] = got
        n = len(got["abstract"]) + sum(len(s["text"]) for s in got["sections"])
        print(f"      -> {via}: abstract {len(got['abstract'])}c, "
              f"{len(got['sections'])} sections, {n/1024:.1f}KB")

    # 合并已有缓存，避免 --only 时覆盖掉其它论文
    if a.only:
        for cf in CACHE.glob("*.json"):
            out.setdefault(cf.stem, json.loads(cf.read_text(encoding="utf-8")))

    OUT.write_text(
        "/* generated by tools/fetch_papers.py — do not edit by hand */\n"
        "window.PAPER_FULLTEXT = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8")

    full = sum(1 for v in out.values() if v["sections"])
    print(f"\n{len(out)}/{len(PAPERS)} papers resolved "
          f"({full} with section-level full text) -> assets/papers.js "
          f"({OUT.stat().st_size/1024:.1f}KB)")
    missing = [p["id"] for p in PAPERS if p["id"] not in out]
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
