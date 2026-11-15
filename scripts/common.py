"""各步驟共用：路徑、隨機種子、讀檔與中間結果存取。"""
import json
import sys
from pathlib import Path

import pandas as pd

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_20NG = ROOT / "data" / "20ng_subset.csv"
DEFAULT_OUT = Path("output")

ENCODINGS = ("utf-8-sig", "utf-8", "cp950", "big5", "gb18030", "latin-1")

CJK_FONTS = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC", "Noto Sans TC",
             "Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]


def cjk_font():
    """回傳可用的中文字型名稱；沒有時回傳 None（圖中文字改用英文）。"""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in CJK_FONTS if f in names), None)


def setup_stdout():
    """Windows 主控台預設不是 UTF-8，統一改成 UTF-8 以正確輸出中文。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def work_dir(out_dir, step):
    d = Path(out_dir) / "work" / step
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_text_file(path):
    raw = Path(path).read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"無法判斷編碼：{path}")


def read_table(path):
    """讀 CSV、Excel 或 .txt 資料夾，回傳 DataFrame。txt 資料夾產生 doc_id、text 兩欄。"""
    path = Path(path)
    if path.is_dir():
        rows = []
        for f in sorted(path.glob("*.txt")):
            text, _ = _read_text_file(f)
            rows.append({"doc_id": f.stem, "text": text})
        if not rows:
            raise ValueError(f"資料夾內沒有 .txt 檔：{path}")
        return pd.DataFrame(rows)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
    if suffix in (".csv", ".tsv", ".txt"):
        from io import StringIO
        text, _ = _read_text_file(path)
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(StringIO(text), sep=sep)
    raise ValueError(f"不支援的檔案格式：{path.suffix}")


def load_docs(path, text_col, label_col=None, id_col=None, categories=None):
    """讀入文件並做基本清理。

    回傳 (docs, stats)。docs 欄位為 doc_id、text，以及（若有指定）label。
    stats 記錄原始列數、刪除的空白列與重複文件數。
    """
    df = read_table(path)
    for col in (text_col, label_col, id_col):
        if col is not None and col not in df.columns:
            raise ValueError(f"找不到欄位「{col}」。可用欄位：{', '.join(map(str, df.columns))}")

    if categories:
        if label_col is None:
            raise ValueError("指定類別時必須同時指定標籤欄位。")
        df = df[df[label_col].isin(categories)]

    docs = pd.DataFrame({"text": df[text_col].astype("string").fillna("").str.strip()})
    if id_col is not None:
        docs.insert(0, "doc_id", df[id_col].astype(str).values)
    elif "doc_id" in df.columns:
        docs.insert(0, "doc_id", df["doc_id"].astype(str).values)
    else:
        docs.insert(0, "doc_id", [f"d{i:05d}" for i in range(len(df))])
    if label_col is not None:
        docs["label"] = df[label_col].astype(str).values

    n_raw = len(docs)
    docs = docs[docs["text"] != ""]
    n_empty = n_raw - len(docs)
    docs = docs.drop_duplicates(subset="text")
    n_dup = n_raw - n_empty - len(docs)
    docs = docs.reset_index(drop=True)
    if not docs["doc_id"].is_unique:
        raise ValueError("文件編號欄有重複值。")
    return docs, {"原始列數": n_raw, "刪除空白列": n_empty, "刪除重複文件": n_dup}
