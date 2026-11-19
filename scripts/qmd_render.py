"""skill 內附的 qmd 執行器：claude.ai 沙盒沒有 Quarto，以此 render 步驟 4 寫出的 report.qmd。

做法與 Quarto 相同：依序執行 qmd 的 Python 區塊（共用同一個命名空間，工作目錄為 qmd 所在資料夾），
把 Markdown 文字與區塊輸出組成單一 HTML 檔。只支援 report.qmd 用到的語法：
  - YAML 標頭的 title、subtitle（雙引號字串）
  - Markdown：#–#### 標題、段落、- 與 1. 清單、行內 `程式碼`、**粗體**、[連結](網址)
  - ```{python}``` 區塊與選項 include、echo、output（true／false／asis）；echo 時程式碼摺疊顯示
  - ```{r}``` 區塊：只用於在 RStudio 準備 Python 環境，執行器略過
  - output: asis 時，輸出中的 ```{=html} … ``` 為原始 HTML，其餘當 Markdown
使用者在自己的電腦以 Quarto render 同一份 qmd，內容相同，版面樣式為 Quarto 的。
"""
import contextlib
import html
import io
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_html import page  # noqa: E402

FENCE = re.compile(r"^```\{(\w+)\}\s*$")
OPTION = re.compile(r"^#\|\s*([\w-]+):\s*(.*)$")
RAW_HTML = re.compile(r"```\{=html\}\n(.*?)\n```", re.S)


def parse(text):
    """回傳 (標頭欄位, 區塊清單)；區塊為 ("md", 文字) 或 ("chunk", 語言, 選項, 程式碼)。"""
    meta = {}
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            mm = re.match(r'^(\w+):\s*"(.*)"\s*$', line)
            if mm:
                meta[mm.group(1)] = mm.group(2).replace('\\"', '"').replace("\\\\", "\\")
        text = text[m.end():]
    blocks, buf = [], []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        f = FENCE.match(lines[i])
        if not f:
            buf.append(lines[i])
            i += 1
            continue
        if buf:
            blocks.append(("md", "\n".join(buf)))
            buf = []
        j = i + 1
        opts, code = {}, []
        while j < len(lines) and lines[j].rstrip() != "```":
            om = OPTION.match(lines[j])
            if om and not code:
                opts[om.group(1)] = om.group(2).strip().strip('"')
            else:
                code.append(lines[j])
            j += 1
        blocks.append(("chunk", f.group(1), opts, "\n".join(code)))
        i = j + 1
    if buf:
        blocks.append(("md", "\n".join(buf)))
    return meta, blocks


def _inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)


def markdown(text, toc):
    """把 report.qmd 用到的 Markdown 轉成 HTML；第 2 層標題加入目錄 toc。"""
    out, para, items, kind = [], [], [], None

    def flush():
        nonlocal para, items, kind
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para = []
        if items:
            out.append(f"<{kind}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{kind}>")
            items, kind = [], None

    for line in text.split("\n"):
        h = re.match(r"^(#{1,4})\s+(.*)$", line)
        li = re.match(r"^\s*(-|\d+\.)\s+(.*)$", line)
        if h:
            flush()
            level, title = len(h.group(1)), h.group(2).strip()
            anchor = f"sec-{len(toc) + 1}" if level == 2 else ""
            if level == 2:
                toc.append((anchor, title))
            out.append(f'<h{level}{f" id={chr(34)}{anchor}{chr(34)}" if anchor else ""}>{_inline(title)}</h{level}>')
        elif li:
            if para:
                flush()
            k = "ul" if li.group(1) == "-" else "ol"
            if kind and kind != k:
                flush()
            kind = k
            items.append(li.group(2))
        elif not line.strip():
            flush()
        elif items and line.startswith("   "):
            items[-1] += " " + line.strip()          # 清單項目的續行
        else:
            if items:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


def _asis(text, toc):
    parts, pos = [], 0
    for m in RAW_HTML.finditer(text):
        parts.append(markdown(text[pos:m.start()], toc))
        parts.append(m.group(1))
        pos = m.end()
    parts.append(markdown(text[pos:], toc))
    return "\n".join(p for p in parts if p.strip())


def render(qmd_path):
    """執行 qmd 並回傳 HTML 字串。任何區塊出錯時拋出 RuntimeError，指出第幾個 Python 區塊。"""
    qmd = Path(qmd_path).resolve()
    meta, blocks = parse(qmd.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    body, toc = [], []
    old_cwd = os.getcwd()
    os.chdir(qmd.parent)
    try:
        n_chunk = 0
        for b in blocks:
            if b[0] == "md":
                body.append(markdown(b[1], toc))
                continue
            _, lang, opts, code = b
            if lang != "python":
                continue
            n_chunk += 1
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, f"{qmd.name}（第 {n_chunk} 個 Python 區塊）", "exec"), ns)
            except SystemExit as e:
                raise RuntimeError(f"{qmd.name} 第 {n_chunk} 個 Python 區塊結束執行：{e}") from e
            except Exception as e:
                raise RuntimeError(f"{qmd.name} 第 {n_chunk} 個 Python 區塊出錯：{type(e).__name__}: {e}") from e
            if opts.get("include", "true") == "false":
                continue
            if opts.get("echo", "true") != "false":
                body.append('<details class="code-fold"><summary>程式碼</summary><pre><code>'
                            + html.escape(code.strip("\n")) + "</code></pre></details>")
            text, mode = buf.getvalue(), opts.get("output", "true")
            if mode == "false" or not text.strip():
                continue
            body.append(_asis(text, toc) if mode == "asis" else "<pre><code>" + html.escape(text) + "</code></pre>")
    finally:
        os.chdir(old_cwd)
    return page(meta.get("title", qmd.stem), meta.get("subtitle", ""), "\n".join(body), toc)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dest = src.with_suffix(".html")
    dest.write_text(render(src), encoding="utf-8")
    print(f"已寫出 {dest}")
