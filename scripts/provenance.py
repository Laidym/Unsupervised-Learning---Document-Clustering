"""報告中的程式碼、執行指令與執行環境，讓使用者可以驗證與重現結果。程式碼依步驟分段嵌入報告。

- 執行指令：各步驟最後一次實際執行的指令（work/stepN/command.json）。步驟 3 若以檔案提供群名稱，
  改列以 --names / --suggestions 直接寫出名稱的等效指令。
- 程式碼：scripts/ 內所有 .py 與內建中文停用詞清單，依 STEP_FILES 歸入主要使用它的步驟（每個檔案只出現一次），
  逐檔附 SHA-256，並可整包下載（zip，內嵌於報告）。
- 內附 jieba（詞典約 9 MB）不嵌入，只列版本與詞典的 SHA-256。
"""
import base64
import hashlib
import io
import platform
import re
import shlex
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
JIEBA_DIR = HERE.parent / "vendor" / "jieba"
STEPS = ("step1", "step2", "step3", "step4")
PACKAGES = ("numpy", "scipy", "pandas", "sklearn", "matplotlib")
# 每個檔案歸入主要使用它的步驟；新增程式檔時須加入此表（有測試檢查）
STEP_FILES = {
    "step1": ("step1_vectorize.py", "tokenize_zh.py", "stopwords_zh.txt", "common.py"),
    "step2": ("step2_cluster.py", "clustering.py", "kmedoids.py", "spherical_kmeans.py", "hierarchical.py"),
    "step3": ("step3_keywords.py", "textmetrics.py"),
    "step4": ("step4_evaluate.py", "wordcloud.py", "report_html.py", "provenance.py"),
}


def source_files():
    """報告要嵌入的程式碼：scripts/ 內所有 .py 與中文停用詞清單，依檔名排序。"""
    files = sorted(p for p in HERE.glob("*.py"))
    stop = HERE / "stopwords_zh.txt"
    return files + ([stop] if stop.exists() else [])


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _quote(s):
    """與 shlex.quote 相同，但中文等非 ASCII 文字不加引號，指令較易讀。"""
    return s if re.fullmatch(r"[\w@%+=:,./-]+", s) else shlex.quote(s)


def _equivalent_step3(args, names_info):
    """把 --names-file / --suggestions-file 換成直接列出名稱，讓指令不依賴暫存檔。"""
    out, skip, replaced = [], False, False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a in ("--names-file", "--suggestions-file"):
            skip, replaced = True, True
            key = "names" if a == "--names-file" else "suggestions"
            vals = names_info.get(key) or []
            if vals:
                out += [f"--{key}", *vals]
            continue
        out.append(a)
    return out, replaced


def commands(out_dir, load_json):
    """回傳 {步驟: (指令字串, 說明)}；沒有記錄的步驟略過。"""
    work = Path(out_dir) / "work"
    rows = {}
    for step in STEPS:
        f = work / step / "command.json"
        if not f.exists():
            continue
        c = load_json(f)
        args, note = list(c["args"]), ""
        if step == "step3" and (work / "step3" / "names.json").exists():
            args, replaced = _equivalent_step3(args, load_json(work / "step3" / "names.json"))
            if replaced:
                note = "原指令以檔案提供群名稱，此處改為直接列出名稱的等效指令"
        cmd = " ".join(_quote(a) for a in ["python3", f"scripts/{c['script']}", *args])
        rows[step] = (cmd, note)
    return rows


def environment():
    rows = [("Python", platform.python_version())]
    for name in PACKAGES:
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = __import__(name)
            except ImportError:
                continue
        rows.append(("scikit-learn" if name == "sklearn" else name, getattr(mod, "__version__", "—")))
    init = JIEBA_DIR / "__init__.py"
    if init.exists():
        ver = next((ln.split("=")[1].strip().strip("'\"") for ln in init.read_text(encoding="utf-8").splitlines()
                    if ln.startswith("__version__")), "—")
        rows.append(("jieba（內附）", ver))
    return rows


def jieba_dict_hash():
    d = JIEBA_DIR / "dict.txt.big"
    return sha256(d.read_bytes()) if d.exists() else None


def code_bundle(files):
    """每個檔案的內容、行數與 SHA-256，以及整包 zip 的 data URI。

    換行一律轉為 LF（與 text-clustering.zip 相同），SHA-256 才與 skill 內的檔案一致。
    """
    items, buf = [], io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            data = p.read_bytes().replace(b"\r\n", b"\n")
            text = data.decode("utf-8")
            step = next((s for s, names in STEP_FILES.items() if p.name in names), "step4")
            items.append({"name": p.name, "step": step, "text": text, "lines": text.count("\n") + (not text.endswith("\n")),
                          "sha256": sha256(data)})
            z.writestr(f"scripts/{p.name}", data)
    uri = "data:application/zip;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return items, uri


def collect(out_dir, load_json):
    items, zip_uri = code_bundle(source_files())
    return {"commands": commands(out_dir, load_json), "environment": environment(),
            "jieba_dict_sha256": jieba_dict_hash(), "files": items, "zip_uri": zip_uri}
