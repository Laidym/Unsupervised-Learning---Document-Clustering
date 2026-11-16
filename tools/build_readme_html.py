"""由 README.md 產生 README.html（單一檔案，資源內嵌）。

README.md 是使用說明的唯一原始檔，寫法以 GitHub 顯示為準。本程式把 GitHub 專用語法
轉成 Quarto 語法，再以 Quarto 輸出 HTML：

- 標題第一行 `# …` → 文件標題
- ``` mermaid → ```{mermaid}（流程圖）
- ``` math 區塊 → $$ … $$；行內 $`…`$ → $…$
- > [!NOTE] 等提示框 → Quarto callout

用法：python tools/build_readme_html.py（需安裝 Quarto）
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "README.md"
OUT = ROOT / "README.html"
TMP = ROOT / "_readme_html.qmd"

FRONT = """---
title: "{title}"
lang: zh-Hant
language:               # Quarto 的 zh 介面文字為簡體，改為繁體
  callout-note-title: "附註"
  callout-tip-title: "提示"
  callout-important-title: "重要"
  callout-warning-title: "警告"
  callout-caution-title: "注意"
  copy-button-tooltip: "複製到剪貼簿"
  copy-button-tooltip-success: "已複製"
format:
  html:
    toc: true
    toc-depth: 2
    toc-title: "目錄"
    embed-resources: true
    mermaid:
      theme: neutral
---
"""

ALERTS = {"NOTE": "note", "TIP": "tip", "IMPORTANT": "important",
          "WARNING": "warning", "CAUTION": "caution"}


def convert_alerts(lines):
    out, i = [], 0
    while i < len(lines):
        m = re.fullmatch(r"> \[!(\w+)\]", lines[i])
        if not m or m.group(1) not in ALERTS:
            out.append(lines[i])
            i += 1
            continue
        out.append(f"::: {{.callout-{ALERTS[m.group(1)]}}}")
        i += 1
        while i < len(lines) and lines[i].startswith(">"):
            body = lines[i][2:] if lines[i].startswith("> ") else lines[i][1:]
            out.append(re.sub(r"^#{1,6} ", "## ", body))   # 提示框內的標題作為 callout 標題
            i += 1
        out.append(":::")
    return out


def convert_math(lines):
    out, in_math, in_code = [], False, False
    for line in lines:
        if line == "``` math":
            out.append("$$")
            in_math = True
        elif in_math and line == "```":
            out.append("$$")
            in_math = False
        elif in_math:
            out.append(line)
        else:
            if line.startswith("```"):
                in_code = not in_code
            out.append(line if in_code else re.sub(r"\$`(.+?)`\$", r"$\1$", line))
    return out


def to_qmd(text):
    lines = text.splitlines()
    title = lines[0].removeprefix("# ").strip()
    body = [("```{mermaid}" if l == "``` mermaid" else l) for l in lines[1:]]
    body = convert_alerts(convert_math(body))
    return FRONT.format(title=title) + "\n".join(body) + "\n"


def main():
    if not shutil.which("quarto"):
        sys.exit("找不到 quarto，請先安裝 Quarto。")
    TMP.write_text(to_qmd(SRC.read_text(encoding="utf-8")), encoding="utf-8")
    try:
        subprocess.run(["quarto", "render", TMP.name, "--output", OUT.name], cwd=ROOT, check=True)
    finally:
        TMP.unlink(missing_ok=True)
        shutil.rmtree(ROOT / "_readme_html_files", ignore_errors=True)
    print(f"已產生 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
