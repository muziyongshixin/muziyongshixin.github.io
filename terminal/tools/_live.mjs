import { JSDOM, VirtualConsole } from "jsdom";
const errs = [];
const vc = new VirtualConsole();
vc.on("jsdomError", e => errs.push("[jsdomError] " + e.message));
vc.on("error", (...a) => errs.push("[console.error] " + a.join(" ")));

const target = process.argv[2] || "https://liyongzhi.xyz/terminal/";
const html = await (await fetch(target)).text();
console.log("拉取:", target, html.length, "字符");

const dom = new JSDOM(html, {
  url: target, runScripts: "dangerously", resources: "usable",
  pretendToBeVisual: true, virtualConsole: vc });
const w = dom.window;
w.HTMLCanvasElement.prototype.getContext = () => ({clearRect(){},fillRect(){},fillText(){},drawImage(){},setTransform(){},getImageData:()=>({data:new Uint8ClampedArray(4)}),putImageData(){}});
w.requestAnimationFrame = cb => setTimeout(()=>cb(Date.now()),16);
w.addEventListener("error", e => errs.push("[window.error] " + e.message));
w.addEventListener("unhandledrejection", e => errs.push("[unhandled] " + e.reason));

await new Promise(r => w.addEventListener("load", r, { once: true }));
await new Promise(r => setTimeout(r, 3000));

console.log("外部脚本是否就位:");
console.log("  AVATAR_DATA  :", w.AVATAR_DATA ? `${w.AVATAR_DATA.w}x${w.AVATAR_DATA.h}` : "✗ 缺失");
console.log("  POSTS_DATA   :", Array.isArray(w.POSTS_DATA) ? w.POSTS_DATA.length + " 篇" : "✗ 缺失");
console.log("  PAPER_FULLTEXT:", w.PAPER_FULLTEXT ? Object.keys(w.PAPER_FULLTEXT).length + " 篇" : "✗ 缺失");
console.log("  __term 出口   :", w.__term ? "✓" : "✗ 脚本未执行完");
const stream = w.document.querySelector("#stream");
console.log("  首屏输出节点  :", stream ? stream.childElementCount : "无 #stream");
console.log("  首屏文本长度  :", stream ? stream.textContent.trim().length : 0);
console.log();
console.log(errs.length ? "运行时错误:" : "✓ 无运行时错误");
[...new Set(errs)].forEach(e => console.log("  - " + e.slice(0, 260)));
process.exit(0);
