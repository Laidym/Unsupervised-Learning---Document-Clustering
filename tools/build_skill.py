"""把 skill 打包成 repo 根目錄的 text-clustering.zip，供上傳到 claude.ai。

修改 SKILL.md、scripts/、vendor/ 或 data/ 後，commit 前要重新執行本腳本，讓 zip 與原始碼一致。

zip 內為單一資料夾 text-clustering/，只含執行所需檔案：
SKILL.md、scripts/（程式與中文停用詞）、vendor/（jieba）、data/（20 Newsgroups 子集）。
不含 tests/、tools/、README.md 等開發用檔案。

用法：python tools/build_skill.py
"""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "text-clustering"
OUT = ROOT / f"{NAME}.zip"          # 放在 repo 根目錄，方便使用者下載安裝

INCLUDE = [
    "SKILL.md",
    "scripts/*.py",
    "scripts/stopwords_zh.txt",
    "vendor/README.md",
    "vendor/jieba/**/*",
    "data/20ng_subset.csv",
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".big", ""}   # "" 為 LICENSE


def collect():
    files = set()
    for pattern in INCLUDE:
        for f in ROOT.glob(pattern):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                files.add(f)
    return sorted(files)


def check_skill_md():
    text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise SystemExit("SKILL.md 缺少 YAML frontmatter")
    meta = dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line)
    if meta.get("name") != NAME:
        raise SystemExit(f"SKILL.md 的 name 應為 {NAME}，目前為 {meta.get('name')}")
    if not 0 < len(meta.get("description", "")) <= 1024:
        raise SystemExit("SKILL.md 的 description 須為 1–1024 字元")


def main():
    check_skill_md()
    files = collect()
    raw = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            data = f.read_bytes()
            if f.suffix in TEXT_SUFFIXES:
                # Windows 上 git 可能把換行轉成 CRLF；沙盒是 Linux，一律以 LF 打包（與 repo 內容一致）
                data = data.replace(b"\r\n", b"\n")
            raw += len(data)
            z.writestr(f"{NAME}/{f.relative_to(ROOT).as_posix()}", data)
    print(f"已寫出 {OUT}：{len(files)} 個檔案，原始 {raw / 1e6:.1f} MB，壓縮後 {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
