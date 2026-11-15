"""步驟 1　前處理與向量化。

固定設定：語言自動判斷；英文先移除 email、網址與主機名稱，依空白、標點與數字切分，
去除內建英文停用詞（含縮寫殘片）；中文以內附 jieba 精確模式＋大詞典斷詞（見 tokenize_zh.py），
去除單字詞、數字與內建中文停用詞；
TF-IDF → Truncated SVD（100 維，文件數 ≤ 100 時為 n − 1）→ 長度正規化。
可選設定：詞彙篩選（min_df、max_df 或不篩選；min_df 預設依文件數：< 1,000 篇為 2，否則為 5）、詞組範圍（單詞或單詞＋雙詞）。

輸出至 <out>/work/step1/，並印出檢查點 1 摘要。標籤只原樣存檔供步驟 4 使用。

用法：
  python scripts/step1_vectorize.py --dataset 20ng
  python scripts/step1_vectorize.py --input reviews.csv --text-col 內容 [--label-col 類別]
  可加 --min-df 5、--no-filter、--ngram 2
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfTransformer,
)
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_20NG, DEFAULT_OUT, SEED, load_docs, load_json, save_json, setup_stdout, work_dir  # noqa: E402
from tokenize_zh import ZhAnalyzer  # noqa: E402

SVD_DIM = 100
ZH_THRESHOLD = 0.3

# 英文縮寫被標點切開後留下的殘片（don't → don），一併視為停用詞
_CONTRACTION_PARTS = {
    "ain", "aren", "couldn", "didn", "doesn", "don", "hadn", "hasn", "haven",
    "isn", "ll", "mustn", "needn", "re", "shouldn", "ve", "wasn", "weren",
    "won", "wouldn",
}
EN_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS | _CONTRACTION_PARTS)

# 由兩個以上字母組成的詞；數字與標點都是分隔符
EN_TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"

# email、網址與主機名稱在切詞後會留下 edu、com、mit 等殘片，切詞前先移除。
# 主機名稱只認通用頂級網域（可再接兩字母國碼），避免誤刪 image.gif 這類檔名。
_EN_NOISE = re.compile(
    r"\S+@\S+"                                            # email
    r"|(?:https?|ftp)://\S+|\bwww\.\S+"                   # 網址
    r"|\b(?:[a-z0-9-]+\.)+(?:edu|com|org|net|gov|mil)"    # 主機名稱
    r"(?:\.[a-z]{2})?\b(?:/\S*)?",
    re.IGNORECASE,
)


def clean_en(text):
    """英文前處理：移除 email、網址與主機名稱，轉小寫。"""
    return _EN_NOISE.sub(" ", text).lower()

_CJK = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_language(texts):
    """中文字元占（中文字元＋英文字母）的比例超過門檻即判為中文。"""
    joined = "".join(texts)
    n_cjk = len(_CJK.findall(joined))
    n_latin = len(_LATIN.findall(joined))
    total = n_cjk + n_latin
    ratio = n_cjk / total if total else 0.0
    return ("zh" if ratio > ZH_THRESHOLD else "en"), ratio


def fixed_settings(lang):
    """記錄於 config.json 與報告的固定前處理設定。"""
    if lang == "zh":
        return {
            "preprocess": "移除 email、網址、主機名稱；英文轉小寫",
            "tokenizer": "jieba 0.42.1 精確模式（內附 dict.txt.big）；只保留兩字以上中文詞與兩個字母以上英文詞",
            "stop_words": "內建中文停用詞（本專案整理）＋英文停用詞",
        }
    return {
        "preprocess": "移除 email、網址、主機名稱；轉小寫",
        "tokenizer": f"依空白、標點與數字切分（{EN_TOKEN_PATTERN}）",
        "stop_words": "scikit-learn ENGLISH_STOP_WORDS＋英文縮寫殘片",
    }


def make_vectorizer(lang, min_df, max_df, ngram_max):
    if lang == "zh":
        return CountVectorizer(
            analyzer=ZhAnalyzer(ngram_max, clean_en, EN_STOP_WORDS),
            min_df=min_df,
            max_df=max_df,
        )
    return CountVectorizer(
        preprocessor=clean_en,
        token_pattern=EN_TOKEN_PATTERN,
        stop_words=sorted(EN_STOP_WORDS),
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, ngram_max),
    )


def svd_dim(n_docs, n_terms):
    return max(1, min(SVD_DIM, n_docs - 1, n_terms - 1))


def vectorize(texts, lang, min_df=5, max_df=0.95, ngram_max=1, seed=SEED):
    """回傳 dict：counts（詞頻）、keep（保留的文件索引）、vocab、vectors、svd 資訊。

    篩選詞彙後完全沒有詞的文件無法計算 cosine 距離，予以排除並記錄於 keep。
    """
    cv = make_vectorizer(lang, min_df, max_df, ngram_max)
    try:
        counts = cv.fit_transform(texts)
    except ValueError as e:
        raise ValueError("詞彙篩選後沒有剩下任何詞，請放寬詞彙篩選。") from e
    vocab = cv.get_feature_names_out().tolist()

    keep = np.flatnonzero(np.asarray(counts.sum(axis=1)).ravel() > 0)
    counts = counts[keep].tocsr()

    tfidf = TfidfTransformer().fit_transform(counts)
    dim = svd_dim(counts.shape[0], counts.shape[1])
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    vectors = normalize(svd.fit_transform(tfidf))
    return {
        "counts": counts,
        "keep": keep,
        "vocab": vocab,
        "vectors": vectors.astype(np.float64),
        "svd_dim": dim,
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
    }


SMALL_CORPUS = 1000           # 文件數少於此值時，min_df 預設為 2
MIN_DF_DEFAULT, MIN_DF_SMALL = 5, 2


def default_min_df(n_docs):
    return MIN_DF_SMALL if n_docs < SMALL_CORPUS else MIN_DF_DEFAULT


def _top_terms_table(counts, vocab, idx, top_n):
    term_total = np.asarray(counts.sum(axis=0)).ravel()
    term_df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idx = np.asarray(idx)
    order = idx[np.argsort(-term_total[idx], kind="stable")[:top_n]]
    return [
        {"詞": vocab[i], "總次數": int(term_total[i]), "出現篇數": int(term_df[i]),
         "出現比例": round(term_df[i] / counts.shape[0], 3)}
        for i in order
    ]


def summarize(docs, counts, vocab, top_n=20, top_bigrams=15, lang="en"):
    if lang == "zh":   # 中文沒有空白分詞，原始長度以字數（不含空白）計
        raw_len = docs["text"].str.replace(r"\s+", "", regex=True).str.len()
    else:
        raw_len = docs["text"].str.split().str.len()
    eff_len = np.asarray(counts.sum(axis=1)).ravel()
    bigram_idx = [i for i, t in enumerate(vocab) if " " in t]
    out = {
        "文件數": int(counts.shape[0]),
        "詞彙量": len(vocab),
        "雙詞數": len(bigram_idx),
        "原始長度單位": "字" if lang == "zh" else "詞",
        "平均原始長度": round(float(raw_len.mean()), 1),
        "原始長度中位數": float(raw_len.median()),
        "平均有效詞數": round(float(eff_len.mean()), 1),
        "有效詞數中位數": float(np.median(eff_len)),
        "有效詞數少於 5 的文件": int((eff_len < 5).sum()),
        "最高頻詞": _top_terms_table(counts, vocab, np.arange(len(vocab)), top_n),
    }
    if bigram_idx:
        out["最高頻雙詞"] = _top_terms_table(counts, vocab, bigram_idx, top_bigrams)
    return out


def settings_text(settings):
    filt = ("不篩選" if settings["min_df"] == 1 and settings["max_df"] == 1.0
            else f"min_df={settings['min_df']}、max_df={settings['max_df']}")
    ngram = "單詞" if settings["ngram_max"] == 1 else "單詞＋雙詞"
    return f"{filt}；{ngram}"


def comparison_table(prev, cur):
    """上一次與本次執行的對照表；prev 為 None 時回傳 None。"""
    if prev is None:
        return None
    rows = []
    for tag, (cfg, summ) in (("上一次", prev), ("本次", cur)):
        rows.append({
            "": tag,
            "設定": settings_text(cfg),
            "詞彙量": summ["詞彙量"],
            "平均有效詞數": summ["平均有效詞數"],
            "有效詞數中位數": f"{summ['有效詞數中位數']:g}",
            "有效詞數<5": summ["有效詞數少於 5 的文件"],
            "排除文件": len(cfg.get("dropped_no_terms", [])),
        })
    return pd.DataFrame(rows)


def print_checkpoint(load_stats, lang, ratio, settings, info, summary, n_dropped, compare=None):
    lang_name = {"en": "英文", "zh": "中文"}[lang]
    print("=" * 60)
    print("檢查點 1　詞彙摘要")
    print("=" * 60)
    print(f"讀入 {load_stats['原始列數']} 列；刪除空白列 {load_stats['刪除空白列']}、"
          f"重複文件 {load_stats['刪除重複文件']}。")
    if n_dropped:
        print(f"另有 {n_dropped} 篇文件在詞彙篩選後沒有任何詞，已排除。")
    print(f"語言判斷：{lang_name}（中文字元比例 {ratio:.1%}）")
    print(f"目前設定：{settings_text(settings)}")
    if settings.get("min_df_auto"):
        print(f"（min_df 依文件數自動設定：文件數 < {SMALL_CORPUS:,} 篇用 {MIN_DF_SMALL}，"
              f"否則用 {MIN_DF_DEFAULT}）")
    print()
    print(f"文件數：{summary['文件數']}")
    bi = f"（其中雙詞 {summary['雙詞數']}）" if summary["雙詞數"] else ""
    print(f"詞彙量：{summary['詞彙量']}{bi}")
    unit = summary["原始長度單位"]
    print(f"平均文件長度：原始 {summary['平均原始長度']} {unit}（中位數 {summary['原始長度中位數']:g}）；"
          f"去除停用詞與篩選後 {summary['平均有效詞數']} 詞（中位數 {summary['有效詞數中位數']:g}）")
    print(f"SVD 降至 {info['svd_dim']} 維。")
    print()
    print("最高頻詞（依總次數）：")
    print(pd.DataFrame(summary["最高頻詞"]).to_string(index=False))
    if "最高頻雙詞" in summary:
        print()
        print("最高頻雙詞（依總次數）：")
        print(pd.DataFrame(summary["最高頻雙詞"]).to_string(index=False))
    if compare is not None:
        print()
        print("與上一次執行比較：")
        print(compare.to_string(index=False))

    notes = []
    if summary["有效詞數少於 5 的文件"]:
        notes.append(f"有 {summary['有效詞數少於 5 的文件']} 篇文件的有效詞數少於 5，分群依據較少。")
    if summary["文件數"] < 60:
        notes.append("文件數少於 60 篇，步驟 2 無法形成有效的候選群數範圍。")
    if notes:
        print()
        print("提示：")
        for s in notes:
            print(f"- {s}")


def _load_previous(wd, path):
    """讀取同一份資料上一次的設定與摘要；資料不同或不存在時回傳 None。"""
    try:
        cfg = load_json(wd / "config.json")
        summ = load_json(wd / "summary.json")
    except (FileNotFoundError, ValueError):
        return None
    return (cfg, summ) if cfg.get("input") == str(path) else None


def main(argv=None):
    setup_stdout()
    ap = argparse.ArgumentParser(description="步驟 1　前處理與向量化")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=["20ng"], help="使用內建範例資料")
    src.add_argument("--input", help="CSV、Excel 檔或 .txt 資料夾")
    ap.add_argument("--text-col", help="文字欄位")
    ap.add_argument("--label-col", help="標籤欄位（只用於步驟 4 事後比對）")
    ap.add_argument("--id-col", help="文件編號欄位（可省略）")
    ap.add_argument("--categories", nargs="+", help="只取這些類別（20ng 用）")
    ap.add_argument("--min-df", type=int,
                    help=f"省略時依文件數自動設定（< {SMALL_CORPUS} 篇為 {MIN_DF_SMALL}，否則為 {MIN_DF_DEFAULT}）")
    ap.add_argument("--max-df", type=float, default=0.95)
    ap.add_argument("--no-filter", action="store_true", help="不篩選詞彙")
    ap.add_argument("--ngram", type=int, choices=[1, 2], default=1, help="1＝單詞；2＝單詞＋雙詞")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    if args.dataset == "20ng":
        path, text_col, label_col, id_col = DATA_20NG, "text", "label", "doc_id"
    else:
        if not args.text_col:
            ap.error("使用 --input 時必須指定 --text-col")
        path, text_col, label_col, id_col = args.input, args.text_col, args.label_col, args.id_col

    docs, load_stats = load_docs(path, text_col, label_col, id_col, args.categories)
    if args.no_filter:
        min_df, max_df, auto = 1, 1.0, False
    elif args.min_df is None:
        min_df, max_df, auto = default_min_df(len(docs)), args.max_df, True
    else:
        min_df, max_df, auto = args.min_df, args.max_df, False
    settings = {"min_df": min_df, "max_df": max_df, "ngram_max": args.ngram, "min_df_auto": auto}

    lang, ratio = detect_language(docs["text"].tolist())
    info = vectorize(docs["text"].tolist(), lang, min_df, max_df, args.ngram)

    dropped = docs.drop(index=info["keep"])
    docs = docs.iloc[info["keep"]].reset_index(drop=True)
    summary = summarize(docs, info["counts"], info["vocab"], lang=lang)

    wd = work_dir(args.out, "step1")
    previous = _load_previous(wd, path)
    docs[["doc_id", "text"]].to_csv(wd / "docs.csv", index=False, encoding="utf-8")
    labels_path = wd / "labels.csv"
    if "label" in docs.columns:
        docs[["doc_id", "label"]].to_csv(labels_path, index=False, encoding="utf-8")
    elif labels_path.exists():
        labels_path.unlink()
    sparse.save_npz(wd / "counts.npz", info["counts"])
    np.save(wd / "vectors.npy", info["vectors"])
    save_json(info["vocab"], wd / "vocab.json")
    config = {
        "input": str(path),
        "text_col": text_col,
        "label_col": label_col,
        "language": lang,
        "zh_ratio": round(ratio, 4),
        **settings,
        **fixed_settings(lang),
        "tfidf": "scikit-learn TfidfTransformer 預設（smooth_idf、l2）",
        "svd_dim": info["svd_dim"],
        "svd_explained_variance": round(info["svd_explained_variance"], 4),
        "seed": SEED,
        "load_stats": load_stats,
        "dropped_no_terms": dropped["doc_id"].tolist(),
    }
    save_json(config, wd / "config.json")
    save_json(summary, wd / "summary.json")

    compare = comparison_table(previous, (config, summary))
    print_checkpoint(load_stats, lang, ratio, settings, info, summary, len(dropped), compare)
    print()
    print(f"中間結果已存於 {wd}")


if __name__ == "__main__":
    main()
