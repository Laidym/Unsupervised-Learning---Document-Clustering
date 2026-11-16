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


def result_dir(out_dir, algorithm, k):
    """結果檔目錄 <out>/<演算法>_k<k>/：不同演算法或群數的結果分開保存，同一組合重跑時覆寫。"""
    d = Path(out_dir) / f"{algorithm}_k{k}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _decode(raw, name):
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"無法判斷編碼：{name}")


def _read_text_file(path):
    return _decode(Path(path).read_bytes(), path)


def _skip(rel):
    """略過 macOS 壓縮時附帶的 __MACOSX/、._ 檔、隱藏檔，以及資料集說明檔 README.txt（不分大小寫）。"""
    return (any(part == "__MACOSX" or part.startswith(".") for part in rel.parts)
            or rel.stem.lower() == "readme")


def txt_files_table(items):
    """把 (相對路徑, bytes) 轉成 DataFrame：doc_id（相對路徑）、text，有子資料夾時另加 folder。

    所有檔案共用的最上層資料夾（例如壓縮時包住全部檔案的那一層）會先去掉。
    folder 為檔案所在的資料夾（相對路徑），可當標籤欄位；不會自動當成標籤。
    """
    items = [(Path(r), b) for r, b in items if Path(r).suffix.lower() == ".txt" and not _skip(Path(r))]
    if not items:
        raise ValueError("找不到 .txt 檔。")
    while all(len(r.parts) > 1 for r, _ in items) and len({r.parts[0] for r, _ in items}) == 1:
        items = [(Path(*r.parts[1:]), b) for r, b in items]
    items.sort(key=lambda x: x[0].as_posix())
    df = pd.DataFrame({"doc_id": [r.as_posix() for r, _ in items],
                       "text": [_decode(b, r)[0] for r, b in items]})
    folders = [r.parent.as_posix() for r, _ in items]
    if any(f != "." for f in folders):
        df["folder"] = [f if f != "." else "" for f in folders]
    return df


def is_txt_collection(path):
    """輸入是否為 .txt 資料夾或 zip（每個檔案一篇文件）。"""
    path = Path(path)
    return path.is_dir() or path.suffix.lower() == ".zip"


def read_table(path):
    """讀 CSV、Excel、.txt 資料夾或內含 .txt 的 zip，回傳 DataFrame。

    .txt 資料夾與 zip 產生 doc_id、text 兩欄（見 txt_files_table），有子資料夾時另有 folder 欄。
    """
    path = Path(path)
    if path.is_dir():
        return txt_files_table((f.relative_to(path), f.read_bytes())
                               for f in path.rglob("*") if f.is_file())
    suffix = path.suffix.lower()
    if suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(path) as z:
            return txt_files_table((i.filename, z.read(i)) for i in z.infolist() if not i.is_dir())
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

    text_col 可為多個欄位（例如標題與內文），依序以換行合併為一篇文件。
    回傳 (docs, stats)。docs 欄位為 doc_id、text，以及（若有指定）label。
    stats 記錄原始列數、刪除的空白列與重複文件數。
    """
    df = read_table(path)
    text_cols = [text_col] if isinstance(text_col, str) else list(text_col)
    for col in (*text_cols, label_col, id_col):
        if col is not None and col not in df.columns:
            raise ValueError(f"找不到欄位「{col}」。可用欄位：{', '.join(map(str, df.columns))}")

    if categories:
        if label_col is None:
            raise ValueError("指定類別時必須同時指定標籤欄位。")
        df = df[df[label_col].isin(categories)]

    parts = [df[c].astype("string").fillna("").str.strip() for c in text_cols]
    text = parts[0]
    for p in parts[1:]:
        text = (text + "\n" + p).str.strip()      # 某欄為空時不留多餘換行
    docs = pd.DataFrame({"text": text})
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
