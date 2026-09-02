/**
 * 冒烟测试：在真实 DOM 里加载 index.html，跑一遍每一条命令。
 *
 * `node --check` 只验证语法，抓不到 runAgent 这种未定义引用 —— 那种错误
 * 只有真正执行到那一行才会暴露。这个脚本就是用来在发布前执行到每一行的。
 *
 * 用法：node tools/smoke-test.mjs
 */
import { JSDOM, VirtualConsole } from "jsdom";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const errors = [];
const logs = [];

const vc = new VirtualConsole();
vc.on("jsdomError", e => errors.push(`[jsdomError] ${e.message}`));
vc.on("error", (...a) => errors.push(`[console.error] ${a.join(" ")}`));
vc.on("warn", (...a) => logs.push(`[warn] ${a.join(" ")}`));

const dom = new JSDOM(readFileSync(resolve(ROOT, "index.html"), "utf8"), {
  url: "http://localhost:8420/",
  runScripts: "dangerously",
  resources: "usable",
  pretendToBeVisual: true,
  virtualConsole: vc,
});
const { window } = dom;

/* jsdom 没有 canvas 后端，给一个够用的假 2D context */
const stub2d = {
  canvas: null, fillStyle: "", font: "", textAlign: "", textBaseline: "",
  globalAlpha: 1, imageSmoothingEnabled: false,
  clearRect() {}, fillRect() {}, fillText() {}, drawImage() {},
  setTransform() {}, getImageData: () => ({ data: new Uint8ClampedArray(4) }), putImageData() {},
};
window.HTMLCanvasElement.prototype.getContext = function () { return { ...stub2d, canvas: this }; };
window.requestAnimationFrame = cb => setTimeout(() => cb(Date.now()), 16);
window.cancelAnimationFrame = id => clearTimeout(id);
/* jsdom 不提供这两个，而流式解析依赖 TextDecoder；注入 Node 的实现，
   mock 编码时也必须用同一个，否则跨 realm 的 Uint8Array 解不出来。 */
window.TextEncoder = TextEncoder;
window.TextDecoder = TextDecoder;
window.addEventListener("error", e => errors.push(`[window.error] ${e.message}`));
window.addEventListener("unhandledrejection", e => errors.push(`[unhandled] ${e.reason}`));

const sleep = ms => new Promise(r => setTimeout(r, ms));

/* 内置 token 在 localhost 白名单内是可用的，若不拦住，命令遍历阶段的
   自然语言输入会真的请求 ModelScope，既慢又消耗真实额度。 */
window.fetch = async () => { throw new Error("network disabled in smoke test"); };

/* 等外部 script 与 boot 序列跑完 */
await new Promise(r => window.addEventListener("load", r, { once: true }));
await sleep(1600);

const $ = s => window.document.querySelector(s);
const fail = m => { errors.push(m); };

/* ---------- 1. 启动后的基本结构 ---------- */
const checks = [
  ["外部数据 avatar.js", () => window.AVATAR_DATA?.w > 0],
  ["外部数据 content.js", () => Array.isArray(window.POSTS_DATA) && window.POSTS_DATA.length > 0],
  ["外部数据 papers.js", () => Object.keys(window.PAPER_FULLTEXT || {}).length > 0],
  ["侧栏命令列表已渲染", () => $("#cmdlist")?.children.length > 0],
  ["欢迎模块已输出", () => $("#stream")?.textContent.includes("START HERE") ||
                            $("#stream")?.querySelector(".mod")],
  ["输入框存在", () => !!$("#cmdline")],
  ["状态栏三段", () => $("#sb-llm") && $("#sb-ctx") && $("#sb-agent")],
];
for (const [name, fn] of checks) {
  let ok = false;
  try { ok = !!fn(); } catch (e) { fail(`结构检查「${name}」抛错：${e.message}`); continue; }
  if (!ok) fail(`结构检查未通过：${name}`);
}

/* ---------- 1b. 界面上的统计数字必须来自数据 ----------
   写死「12 篇论文」这种文案，每加一篇论文就错一次，而且没人会记得改。 */
const T0 = window.__term;
if (!T0) fail("window.__term 测试出口不存在");
else {
  const home = $("#stream").textContent;
  const expect = [
    [/pub · (\d+) 篇论文/, T0.PAPERS.length, "首屏论文数"],
    [/blog · (\d+) 篇文章/, T0.POSTS.length, "首屏文章数"],
    [/exp · (\d+) 段经历/, T0.EXPERIENCE.length, "首屏经历数"],
  ];
  for (const [re, want, what] of expect) {
    const m = home.match(re);
    if (!m) fail(`${what}：首屏没找到对应文案`);
    else if (+m[1] !== want) fail(`${what}：页面显示 ${m[1]}，数据实际是 ${want}`);
  }
  /* reader 里「最近 N 篇被 X 接收」同样要能跟着数据走 */
  try {
    window.openReader();
    const rd = $("#rd-body").textContent;
    const m = rd.match(/RECENT · (\d+) 篇被/);
    const latest = Math.max(...T0.PAPERS.map(p => p.year));
    const want = T0.PAPERS.filter(p => p.year >= latest && !p.preprint).length;
    if (!m) fail("reader 最新成果行缺失");
    else if (+m[1] !== want) fail(`reader 最新成果显示 ${m[1]} 篇，数据实际 ${want} 篇`);
    const stats = [...$("#rd-body").querySelectorAll(".stat .n")].map(n => n.textContent);
    if (!stats.includes(String(T0.PAPERS.length))) fail("reader 统计卡缺少论文总数");
    if (!stats.includes(String(T0.SCHOLAR.citations))) fail("reader 统计卡缺少引用数");
    $("#rd-close")?.click();
  } catch (e) { fail(`统计数字检查抛错：${e.message}`); }
}

/* ---------- 2. 逐条执行所有命令 ---------- */
const stream = $("#stream");
const before = errors.length;

// 从侧栏取出真实注册的命令名，避免测试和实现脱节
const registered = [...$("#cmdlist").querySelectorAll(".cmd-item")].map(el => el.dataset.cmd);
const ARGS = {
  paper: "1", cite: "adaptive-video-distillation", explain: "acpo",
  pub: "cvpr", blog: "diffusion", model: "", theme: "dark", agent: "",
  avatar: "dither", config: "", ask: "",
};
const EXTRA = [
  "pub 2026", "pub 不存在的关键词", "paper autocut", "paper", "cite 99",
  "blog python", "explain narrative-weaver --eli5", "avatar ascii", "avatar color",
  "agent off", "agent on", "theme light", "theme dark", "reset",
  "sudo hire-me", "sudo rm -rf", "不是命令的一句话提问", "help",
];

const ran = [];
for (const name of [...registered, ...EXTRA]) {
  const line = registered.includes(name) ? `${name} ${ARGS[name] ?? ""}`.trim() : name;
  const mark = stream.childElementCount;
  const errMark = errors.length;
  try {
    window.exec(line);
  } catch (e) {
    fail(`执行 \`${line}\` 抛错：${e.message}`);
    continue;
  }
  await sleep(30);
  const produced = stream.childElementCount - mark;
  const text = [...stream.children].slice(mark).map(n => n.textContent).join("");
  if (produced === 0 && !["clear","cls"].includes(name)) fail(`执行 \`${line}\` 没有产生任何输出`);
  if (/undefined|\[object Object\]|NaN/.test(text))
    fail(`执行 \`${line}\` 的输出里出现 undefined / [object Object] / NaN`);
  ran.push({ line, produced, chars: text.length, errs: errors.length - errMark });
}

/* ---------- 3. reader 视图 ---------- */
try {
  window.openReader();
  await sleep(120);
  if (!$("#reader")?.classList.contains("show")) fail("reader 视图没有显示");
  if (!$("#rd-body")?.textContent.includes("PUBLICATIONS")) fail("reader 视图缺少论文区块");
  if (!$("#rd-pix")) fail("reader 视图缺少像素肖像画布");
  $("#rd-close")?.click();
} catch (e) { fail(`reader 视图抛错：${e.message}`); }

/* ---------- 4. 输入法与光标 ---------- */
try {
  const input = $("#cmdline");
  input.value = "pu";
  input.dispatchEvent(new window.Event("input"));
  const ghost = $("#ghost").textContent;
  if (!ghost.startsWith("pub")) fail(`Tab 补全提示异常，期望以 pub 开头，实际 "${ghost}"`);
  input.dispatchEvent(new window.CompositionEvent("compositionstart"));
  if ($("#ghost").textContent !== "") fail("组字期间 ghost 提示未清空");
  const kd = new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true });
  Object.defineProperty(kd, "isComposing", { value: true });
  const n0 = stream.childElementCount;
  input.dispatchEvent(kd);
  if (stream.childElementCount !== n0) fail("组字期间按 Enter 仍然提交了命令");
  input.dispatchEvent(new window.CompositionEvent("compositionend"));
  input.value = "";
} catch (e) { fail(`输入法检查抛错：${e.message}`); }

/* ---------- 5. 本地工具 ---------- */
const T = window.__term;
if (!T) fail("window.__term 测试出口不存在");
try {
  const r = await T.TOOLS.site_search.run({ query: "视频生成" });
  if (!r || r.length < 40) fail("site_search 返回内容过短");
  const rp = await T.TOOLS.read_paper.run({ id: "adaptive-video-distillation" });
  if (!/摘要/.test(rp)) fail("read_paper 未返回摘要");
  const miss = await T.TOOLS.read_paper.run({ id: "根本不存在的论文" });
  if (!miss.includes("没有找到")) fail("read_paper 对不存在的 id 未给出友好提示");
  const post = await T.TOOLS.read_post.run({ slug: "agentic-rl" });
  if (!post.includes("章节")) fail("read_post 未返回章节");
} catch (e) { fail(`工具检查抛错：${e.message}`); }

/* ---------- 6. agent 循环（mock 掉模型接口） ----------
   runAgent 是最复杂也最容易出错的一段（那个 runAgent/runLLM 命名不一致
   就藏在这里），必须真正执行到。 */
function sse(text, chunk = 24) {
  const parts = [];
  for (let i = 0; i < text.length; i += chunk)
    parts.push(`data: ${JSON.stringify({ choices: [{ delta: { content: text.slice(i, i + chunk) } }] })}\n\n`);
  parts.push("data: [DONE]\n\n");
  /* 必须用页面 realm 的 TextEncoder：jsdom 里 window.TextDecoder 不接受
     Node realm 造出来的 Uint8Array，解码会静默失败。 */
  const enc = new window.TextEncoder();
  let i = 0;
  return {
    ok: true, status: 200,
    text: async () => "",
    body: { getReader: () => ({
      read: async () => i < parts.length
        ? { done: false, value: enc.encode(parts[i++]) }
        : { done: true, value: undefined }
    }) }
  };
}

let queue = [], sent = [];
window.fetch = async (url, opts) => {
  sent.push(JSON.parse(opts.body));
  if (!queue.length) throw new Error("mock: 未预置响应");
  return sse(queue.shift());
};
window.localStorage.setItem("lyz.mskey", "ms-faketoken-for-test");

async function agentRun(cmd, responses, timeout = 4000) {
  queue = [...responses]; sent = [];
  const n = stream.childElementCount;
  window.__term.exec(cmd);
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    await sleep(40);
    if (!window.document.body.classList.contains("busy") && queue.length === 0) break;
  }
  await sleep(120);
  return [...stream.children].slice(n);
}

/* 6a. 普通回答，不调工具 */
try {
  const before = T.CHAT.length;
  const out = await agentRun("他最近在做什么方向", ["他目前在快手负责 AdsLLM 与 AI Agent 方向的工作。"]);
  const text = out.map(n => n.textContent).join("");
  if (!text.includes("AdsLLM")) fail("agent 普通回答未渲染到页面");
  if (sent.length !== 1) fail(`agent 普通回答应只请求 1 次，实际 ${sent.length} 次`);
  if (T.CHAT.length !== before + 2) fail(`agent 普通回答应写入 2 条上下文，实际 ${T.CHAT.length - before} 条`);
  const msgs = sent[0].messages;
  if (msgs[0].role !== "system") fail("请求首条消息不是 system");
  if (msgs[msgs.length - 1].role !== "user") fail("请求末条消息不是 user");
} catch (e) { fail(`agent 普通回答抛错：${e.message}`); }

/* 6b. 工具调用链路 */
try {
  const before = T.CHAT.length;
  const out = await agentRun("站内有哪些关于视频生成的内容", [
    '<tool>{"name":"site_search","args":{"query":"视频生成"}}</tool>',
    "站内相关的工作主要集中在视频生成与一致性控制上。",
  ]);
  const text = out.map(n => n.textContent).join("");
  if (!out.some(n => n.querySelector?.(".tool"))) fail("工具调用卡片没有渲染");
  if (!text.includes("site_search")) fail("工具卡片未显示工具名");
  if (sent.length !== 2) fail(`工具链路应请求 2 次，实际 ${sent.length} 次`);
  if (T.CHAT.length !== before + 4) fail(`工具链路应写入 4 条上下文，实际 ${T.CHAT.length - before} 条`);
  const second = sent[1].messages;
  if (!second.some(m => m.content.includes("[工具结果 · site_search]")))
    fail("第二次请求没有携带工具结果");
  /* KV cache 的前提：第二次请求必须以第一次的消息为前缀 */
  const p1 = sent[0].messages, p2 = second;
  const prefixOK = p1.every((m, i) => p2[i] && p2[i].role === m.role && p2[i].content === m.content);
  if (!prefixOK) fail("第二次请求没有复用第一次的消息前缀，KV cache 会失效");
} catch (e) { fail(`agent 工具链路抛错：${e.message}`); }

/* 6c. 模型返回坏 JSON 时不能卡死 */
try {
  const out = await agentRun("触发一个坏的工具调用", ['<tool>{"name": 不是合法JSON}</tool>']);
  if (!out.length) fail("坏工具调用没有任何输出");
  if (window.document.body.classList.contains("busy")) fail("坏工具调用后仍停留在 busy 状态");
} catch (e) { fail(`坏工具调用抛错：${e.message}`); }

/* 6d. 接口报错时的兜底 */
try {
  queue = [];
  window.fetch = async () => { throw new Error("network down"); };
  const n = stream.childElementCount;
  window.__term.exec("ask 网络挂了会怎样");
  await sleep(600);
  const text = [...stream.children].slice(n).map(x => x.textContent).join("");
  if (!/失败|错误|error/i.test(text)) fail("接口报错时没有给出提示");
  if (window.document.body.classList.contains("busy")) fail("接口报错后仍停留在 busy 状态");
} catch (e) { fail(`接口报错兜底抛错：${e.message}`); }

/* ---------- 7. 内置 token 的混淆与配额 ---------- */
try {
  const V = T.LLM;
  const builtin = V.builtin;
  if (!/^ms-[\w-]{20,}$/.test(builtin)) fail("内置 token 没能在白名单域名下解出");
  const src = readFileSync(resolve(ROOT, "index.html"), "utf8");
  if (src.includes(builtin)) fail("index.html 里出现了 token 明文");
  if (/ms-[0-9a-f]{8}-[0-9a-f]{4}/.test(src)) fail("index.html 里能正则匹配到 token 形态");
  /* 非白名单域名不应组装出 token */
  dom.reconfigure({ url: "https://evil.example.com/" });
  if (T.LLM.builtin !== "") fail("非白名单域名下仍然解出了内置 token");
  dom.reconfigure({ url: "http://localhost:8420/" });
  if (T.LLM.builtin !== builtin) fail("恢复域名后内置 token 解不出来了");
  /* 配额耗尽时应当拦下请求而不是继续调用。
     先清掉前面 agent 测试写入的自带 token —— 自带 token 本就不受配额限制。 */
  window.localStorage.removeItem("lyz.mskey");
  if (T.LLM.own !== "") fail("清除后 LLM.own 仍非空");
  window.localStorage.setItem("lyz.quota", JSON.stringify(
    { day: new Date().toISOString().slice(0, 10), n: T.QUOTA.max }));
  if (!T.LLM.blocked) fail("配额用尽后 blocked 仍为 false");
  const n0 = stream.childElementCount;
  T.exec("配额用尽时的提问");
  await sleep(80);
  const txt = [...stream.children].slice(n0).map(x => x.textContent).join("");
  if (!txt.includes("额度")) fail("配额用尽时没有给出提示");
  window.localStorage.removeItem("lyz.quota");
} catch (e) { fail(`内置 token 检查抛错：${e.message}`); }

/* ---------- 报告 ---------- */
const cmdErrs = errors.length - before;
console.log(`\n执行命令 ${ran.length} 条，DOM 节点产出正常 ${ran.filter(r => r.produced > 0).length} 条`);
const quiet = ran.filter(r => r.chars < 40);
if (quiet.length) console.log("输出偏短（请人工确认）:", quiet.map(r => r.line).join(", "));

if (errors.length) {
  console.log(`\n✗ 发现 ${errors.length} 个问题：`);
  [...new Set(errors)].forEach(e => console.log("  - " + e.slice(0, 300)));
  process.exit(1);
}
console.log("\n✓ 所有检查通过（命令执行、reader 视图、输入法、工具、上下文）");
process.exit(0);
