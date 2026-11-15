"""步驟 4　評估與限制。

內部指標（固定）：silhouette、關鍵詞一致性 NPMI。
事後比對（自動）：步驟 1 有指定標籤欄才計算 ARI、NMI、純度；否則略過。標籤只在本步驟讀取。
限制：固定說明四項，並依本次資料提示。

輸出：<out>/metrics.csv、<out>/report.md（文字報告，記錄所有設定與 seed）。

用法：python scripts/step4_evaluate.py [--out output]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, load_json, setup_stdout  # noqa: E402
from clustering import ALGORITHMS, silhouette  # noqa: E402
from hierarchical import cluster_linkage, merge_steps  # noqa: E402
from textmetrics import cluster_npmi  # noqa: E402

FIXED_LIMITS = [
    "每篇文件只能屬於一群；同時涉及多個主題的文件會被歸入單一群。",
    "詞袋表示忽略詞序與語意，同義詞視為不同的詞。",
    "群數沒有唯一正確答案，指標僅供參考。",
    "群名稱由使用者根據關鍵詞決定（可採用 skill 依關鍵詞提出的建議名稱），並非分群模型產出。",
]

SHORT_DOC_MEDIAN = 10          # 有效詞數中位數低於此值視為文件過短
SMALL_CLUSTER_MIN = 10         # 文件數少於此值的群
SMALL_CLUSTER_RATIO = 0.3      # 或少於平均群大小的此比例
UNEVEN_RATIO = 5               # 最大群／最小群超過此倍數
NPMI_GAP = 0.1                 # 群的 NPMI 低於各群平均超過此值


def purity(clusters, labels):
    """每群取多數標籤，多數文件數加總後除以 n。"""
    ct = pd.crosstab(clusters, labels)
    return float(ct.max(axis=1).sum() / len(clusters))


def external_metrics(clusters, labels):
    return {
        "ARI": float(adjusted_rand_score(labels, clusters)),
        "NMI": float(normalized_mutual_info_score(labels, clusters)),
        "純度": purity(clusters, labels),
    }


def data_limits(*, n, k, sizes, npmi_per, silhouette, eff_median, n_dropped, k_range):
    """依本次資料產生的限制提示。"""
    notes = []
    if eff_median < SHORT_DOC_MEDIAN:
        notes.append(f"文件偏短：去除停用詞後的有效詞數中位數只有 {eff_median:g} 個，"
                     "詞袋向量稀疏，分群與關鍵詞較不穩定。")
    small = [c for c, s in enumerate(sizes)
             if s < SMALL_CLUSTER_MIN or s < SMALL_CLUSTER_RATIO * n / k]
    if small:
        desc = "、".join(f"群 {c}（{sizes[c]} 篇）" for c in small)
        notes.append(f"部分群文件數過少：{desc}。其關鍵詞容易受個別文件影響。")
    if min(sizes) > 0 and max(sizes) / min(sizes) > UNEVEN_RATIO:
        notes.append(f"各群大小差異大（最大 {max(sizes)} 篇、最小 {min(sizes)} 篇），"
                     "c-TF-IDF 對小群的關鍵詞較不穩定。")
    vals = np.asarray(npmi_per, dtype=float)
    if np.isfinite(vals).any():
        mean = float(np.nanmean(vals))
        low = [c for c, v in enumerate(vals) if np.isfinite(v) and (v < 0 or v < mean - NPMI_GAP)]
        if low:
            desc = "、".join(f"群 {c}（{vals[c]:.3f}）" for c in low)
            notes.append(f"以下群的關鍵詞一致性明顯偏低（各群平均 {mean:.3f}）：{desc}。"
                         "這些關鍵詞較少在同一文件一起出現，群的內容可能較混雜，命名需謹慎。")
    if silhouette < 0.1:
        notes.append(f"silhouette 為 {silhouette:.3f}，群與群之間界線不明顯。"
                     "這在高維文字資料很常見，但表示不少文件位於兩群交界。")
    if n_dropped:
        notes.append(f"有 {n_dropped} 篇文件在詞彙篩選後沒有任何詞，未納入分群。")
    if k == k_range[1]:
        notes.append(f"選定的群數 k = {k} 是候選範圍的上限，更大的群數未經評估。")
    return notes


def _md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def build_report(ctx):
    s1, s2 = ctx["cfg1"], ctx["cfg2"]
    filt = ("不篩選" if s1["min_df"] == 1 and s1["max_df"] == 1.0
            else f"min_df = {s1['min_df']}、max_df = {s1['max_df']}")
    ngram = "單詞" if s1["ngram_max"] == 1 else "單詞＋雙詞"
    L = ["# 文件分群報告", ""]

    L += ["## 設定紀錄", ""]
    L += [f"- 資料：{s1['input']}（文字欄 `{s1['text_col']}`"
          + (f"，標籤欄 `{s1['label_col']}`" if s1.get("label_col") else "，無標籤欄") + "）",
          f"- 讀入 {s1['load_stats']['原始列數']} 列，刪除空白列 {s1['load_stats']['刪除空白列']}、"
          f"重複文件 {s1['load_stats']['刪除重複文件']}、無有效詞文件 {len(s1['dropped_no_terms'])}；"
          f"分析 {ctx['n']} 篇",
          f"- 語言：{'英文' if s1['language'] == 'en' else '中文'}",
          f"- 前處理：{s1['preprocess']}；斷詞：{s1.get('tokenizer', '—')}；停用詞：{s1['stop_words']}",
          f"- 詞彙篩選：{filt}；詞組範圍：{ngram}；詞彙量 {ctx['n_vocab']}",
          f"- 向量化：TF-IDF → Truncated SVD {s1['svd_dim']} 維 → 長度正規化",
          f"- 分群：{ALGORITHMS.get(s2['algorithm'], s2['algorithm'])}，cosine 距離，{s2['n_init']} 次初始化",
          f"- 候選群數 k = {s2['k_min']} .. {s2['k_max']}；選定 k = {ctx['k']}",
          "- 關鍵詞：c-TF-IDF 前 10 名",
          f"- 隨機種子：{s1['seed']}（步驟 1 SVD、步驟 2 k-medoids）", ""]

    L += ["## 各群", ""]
    L += [_md_table(ctx["cluster_table"]), ""]
    if ctx.get("merge"):
        L += ["## 各群的上下層關係（階層式分群）", "",
              "由近到遠的合併順序；越早合併的群內容越接近：", ""]
        L += [f"{i}. {line}" for i, line in enumerate(ctx["merge"], 1)] + [""]

    L += ["## 內部指標", ""]
    samp = "；文件數超過 5,000 篇，以隨機抽樣 5,000 篇計算" if ctx["sil_sampled"] else ""
    L += [f"- silhouette 係數：{ctx['silhouette']:.3f}（範圍 −1 至 1；文字資料通常偏低，宜比較不同群數的相對高低{samp}）",
          f"- 關鍵詞一致性 NPMI：{ctx['npmi']:.3f}（範圍 −1 至 1；越高表示各群關鍵詞越常在同一文件共同出現）", ""]

    L += ["## 事後比對", ""]
    if ctx["external"] is None:
        L += ["未指定標籤欄位，略過事後比對。", ""]
    else:
        e = ctx["external"]
        L += [f"- ARI：{e['ARI']:.3f}（1 為完全一致，0 相當於隨機分群）",
              f"- NMI：{e['NMI']:.3f}（0 為毫無關聯，1 為完全對應）",
              f"- 純度：{e['純度']:.3f}（群數越多越高，須搭配 ARI、NMI 解讀）",
              f"- 群數 {ctx['k']}，標籤類別數 {ctx['n_classes']}",
              "", "群與標籤的交叉表：", "", _md_table(ctx["crosstab"].reset_index()), "",
              "既有分類只是一種分法，不吻合不代表分群錯誤。標籤只用於本節，未用於前處理、分群與選定群數。", ""]

    L += ["## 限制", ""]
    L += [f"- {s}" for s in FIXED_LIMITS]
    if ctx["data_notes"]:
        L += ["", "依本次資料："] + [f"- {s}" for s in ctx["data_notes"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    setup_stdout()
    ap = argparse.ArgumentParser(description="步驟 4　評估與限制")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    w1, w2, w3 = (out / "work" / s for s in ("step1", "step2", "step3"))
    if not (w3 / "names.json").exists():
        sys.exit("找不到群名稱，請先完成步驟 3 的命名。")
    names_info = load_json(w3 / "names.json")
    kw = load_json(w3 / "keywords.json")
    k = names_info["k"]
    if kw["k"] != k:
        sys.exit("步驟 3 的關鍵詞與群名稱的群數不一致，請重跑步驟 3。")
    names = names_info["names"]

    cfg1, cfg2 = load_json(w1 / "config.json"), load_json(w2 / "config.json")
    vectors = np.load(w1 / "vectors.npy")
    counts = sparse.load_npz(w1 / "counts.npz").tocsr()
    vocab = load_json(w1 / "vocab.json")
    docs = pd.read_csv(w1 / "docs.csv", dtype={"doc_id": str})
    clusters = np.load(w2 / "labels.npz")[f"k{k}"]
    n = len(docs)

    sil, sil_sampled = silhouette(vectors, clusters, seed=cfg1["seed"])
    merge = None
    if cfg2.get("algorithm") == "hierarchical" and (w2 / "linkage.npy").exists() and k >= 2:
        Z = cluster_linkage(np.load(w2 / "linkage.npy"), clusters, k)
        merge = merge_steps(Z, [f"群 {c} {names[c]}" for c in range(k)])
    vidx = {t: i for i, t in enumerate(vocab)}
    kw_idx = [[vidx[t] for t in c["keywords"]] for c in kw["clusters"]]
    npmi, npmi_per = cluster_npmi(counts, kw_idx)
    sizes = np.bincount(clusters, minlength=k).tolist()
    eff_median = float(np.median(np.asarray(counts.sum(axis=1)).ravel()))

    # 標籤只在此讀取
    external = crosstab = None
    n_classes = None
    labels_path = w1 / "labels.csv"
    if labels_path.exists():
        lab = pd.read_csv(labels_path, dtype={"doc_id": str, "label": str})
        if not (lab["doc_id"].to_numpy() == docs["doc_id"].to_numpy()).all():
            sys.exit("標籤檔與文件順序不一致，請重跑步驟 1。")
        y = lab["label"].to_numpy()
        external = external_metrics(clusters, y)
        n_classes = int(pd.Series(y).nunique())
        crosstab = pd.crosstab(pd.Series([f"{c} {names[c]}" for c in clusters], name="群"),
                               pd.Series(y, name="標籤"))

    sources = names_info.get("sources") or ["自訂"] * k
    cluster_table = pd.DataFrame({
        "群": range(k),
        "群名稱": names,
        "名稱來源": sources,
        "文件數": sizes,
        "NPMI": [float(v) for v in npmi_per],
        "關鍵詞": [", ".join(c["keywords"]) for c in kw["clusters"]],
    })
    notes = data_limits(n=n, k=k, sizes=sizes, npmi_per=npmi_per, silhouette=sil,
                        eff_median=eff_median, n_dropped=len(cfg1["dropped_no_terms"]),
                        k_range=(cfg2["k_min"], cfg2["k_max"]))

    rows = [("silhouette", "整體", sil), ("NPMI", "整體", npmi)]
    rows += [("NPMI", f"群 {c} {names[c]}", float(v)) for c, v in enumerate(npmi_per)]
    if external:
        rows += [(m, "整體", v) for m, v in external.items()]
    metrics = pd.DataFrame(rows, columns=["指標", "範圍", "數值"])
    metrics["數值"] = metrics["數值"].round(4)
    metrics.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")

    ctx = {"cfg1": cfg1, "cfg2": cfg2, "n": n, "n_vocab": len(vocab), "k": k,
           "merge": merge, "silhouette": sil, "sil_sampled": sil_sampled, "npmi": npmi, "external": external, "n_classes": n_classes,
           "crosstab": crosstab, "cluster_table": cluster_table, "data_notes": notes}
    report = build_report(ctx)
    (out / "report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"已寫出 {out / 'metrics.csv'}、{out / 'report.md'}")


if __name__ == "__main__":
    main()
