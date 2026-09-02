#!/usr/bin/env python3
"""把 ModelScope token 混淆成可以嵌进前端的形式，并直接写回 index.html。

⚠️ 这是混淆，不是加密。解码逻辑随代码发给浏览器，任何人打开 DevTools
看 Network 里的 Authorization 头都能拿到明文。它能挡住的是 GitHub 密钥
扫描和批量爬虫的正则匹配——那才是明文 token 最常见的泄露途径。

真正的防线是另外三条，都已经在 index.html 里：
  1. 域名白名单：只在自己的域名下才组装出 token
  2. 前端配额：单个浏览器每天有限次数，刷不爆免费额度
  3. 可随时作废：在 ModelScope 撤销旧 token，用新 token 重跑本脚本

用法：
    python3 tools/obfuscate_key.py ms-xxxxxxxx-xxxx-xxxx
    python3 tools/obfuscate_key.py --check      # 只校验当前嵌入的能否解回
"""
import argparse, base64, hashlib, pathlib, random, re, sys

HERE = pathlib.Path(__file__).resolve().parent.parent
PAGE = HERE / "index.html"


def make_salt(n=24):
    """随机盐，每次轮换都不一样，避免密文形态被固定特征匹配。"""
    return [random.randint(33, 250) for _ in range(n)]


def obfuscate(token, salt):
    raw = token.encode("utf-8")
    xored = bytes(b ^ salt[i % len(salt)] for i, b in enumerate(raw))
    return base64.b64encode(xored).decode("ascii")


def deobfuscate(blob, salt):
    raw = base64.b64decode(blob)
    return bytes(b ^ salt[i % len(salt)] for i, b in enumerate(raw)).decode("utf-8")


VAULT_RE = re.compile(r"const VAULT = \{.*?\n\};", re.S)


def vault_block(text):
    """必须先框定 VAULT 这一段再替换：全局搜 `d: "..."` 会命中论文数据里的
    `id:"adaptive-video-distillation"`，把论文 id 直接覆盖掉。"""
    m = VAULT_RE.search(text)
    if not m:
        sys.exit("index.html 里找不到 const VAULT = { ... };")
    return m


def read_current(text):
    block = vault_block(text).group(0)
    m = re.search(r'\n\s*d:\s*"([^"]*)"', block)
    s = re.search(r"\n\s*s:\s*\[([0-9,\s]*)\]", block)
    if not m or not s:
        return None, None
    salt = [int(x) for x in s.group(1).split(",") if x.strip()]
    return m.group(1), salt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("token", nargs="?")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    text = PAGE.read_text(encoding="utf-8")

    if a.check or not a.token:
        blob, salt = read_current(text)
        if not blob:
            print("index.html 里还没有嵌入 token")
            return
        tok = deobfuscate(blob, salt)
        print(f"嵌入的 token 可正常解回：{tok[:6]}…{tok[-4:]}  （长度 {len(tok)}）")
        print(f"sha256[:12] = {hashlib.sha256(tok.encode()).hexdigest()[:12]}")
        return

    token = a.token.strip()
    if not token.startswith("ms-"):
        sys.exit("ModelScope 的 SDK Token 应以 ms- 开头")

    salt = make_salt()
    blob = obfuscate(token, salt)
    assert deobfuscate(blob, salt) == token, "自检失败"

    vm = vault_block(text)
    block = vm.group(0)
    block, n1 = re.subn(r'(\n\s*d:\s*)"[^"]*"',
                        lambda m: m.group(1) + '"' + blob + '"', block, count=1)
    block, n2 = re.subn(r"(\n\s*s:\s*)\[[0-9,\s]*\]",
                        lambda m: m.group(1) + "[" + ",".join(map(str, salt)) + "]",
                        block, count=1)
    if not (n1 and n2):
        sys.exit("VAULT 里没有找到 d / s 字段")
    new_text = text[:vm.start()] + block + text[vm.end():]

    PAGE.write_text(new_text, encoding="utf-8")
    print(f"已写入 index.html")
    print(f"  token : {token[:6]}…{token[-4:]}")
    print(f"  盐长度 : {len(salt)}")
    print(f"  密文   : {blob[:44]}…")
    print("\n提醒：这只是混淆。若发现被盗刷，去 ModelScope 撤销该 token，")
    print("      再用新 token 重跑本脚本即可。")


if __name__ == "__main__":
    main()
