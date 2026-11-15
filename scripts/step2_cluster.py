"""步驟 2　文件分群。

讀入步驟 1 的向量，對每個候選群數 k 分群，計算 silhouette 與關鍵詞一致性（NPMI），
畫在同一張圖，並列出各 k 的群關鍵詞，供檢查點 2 選定群數。不讀取標籤。

用法：
  python scripts/step2_cluster.py [--algorithm kmedoids|spherical|hierarchical] [--k-min 2 --k-max 15] [--out output]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clustering import (  # noqa: E402
    ALGORITHMS, KMEDOIDS_MAX, LARGE_N_WARNING, N_INIT, needs_distance_matrix, run_clustering, silhouette,
    suggest_algorithm,
)
from common import DEFAULT_OUT, SEED, cjk_font, load_json, save_json, setup_stdout, work_dir  # noqa: E402
from hierarchical import build_linkage, cluster_linkage, plot_dendrogram  # noqa: E402
from kmedoids import cosine_distance_matrix  # noqa: E402
from textmetrics import cluster_npmi, top_terms  # noqa: E402

K_CAP = 15
DOCS_PER_CLUSTER = 30



def default_k_range(n):
    """k = 2 .. min(15, floor(n/30))；n < 60 時無有效範圍，回傳 None。"""
    k_max = min(K_CAP, n // DOCS_PER_CLUSTER)
    return (2, k_max) if k_max >= 2 else None


def run_k(vectors, D, counts, vocab, k, algorithm, seed=SEED, n_init=N_INIT, L=None):
    res = run_clustering(vectors, k, algorithm, D=D, seed=seed, n_init=n_init, L=L)
    labels = res["labels"]
    sil, _ = silhouette(vectors, labels, D=D, seed=seed)
    kw = top_terms(counts, labels, k, vocab)
    npmi, npmi_per = cluster_npmi(counts, [c["index"] for c in kw])
    return {
        "k": k,
        "labels": labels,
        "medoids": res["medoids"],
        "cost": res["cost"],
        "silhouette": sil,
        "npmi": npmi,
        "npmi_per_cluster": npmi_per,
        "sizes": np.bincount(labels, minlength=k),
        "keywords": [c["terms"] for c in kw],
    }


def plot_k_selection(metrics, path):
    """silhouette 與 NPMI 尺度不同，上下兩格共用 k 軸（不用雙 y 軸）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = cjk_font()
    if font:
        plt.rcParams["font.family"] = [font, "sans-serif"]
    zh = font is not None
    ink, muted, grid, color = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True, facecolor="#fcfcfb")
    rows = [("silhouette", "silhouette 係數" if zh else "Silhouette"),
            ("npmi", "關鍵詞一致性 NPMI" if zh else "Keyword coherence (NPMI)")]
    for ax, (col, title) in zip(axes, rows):
        x, y = metrics["k"], metrics[col]
        ax.set_facecolor("#fcfcfb")
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        best = int(np.nanargmax(y.values))
        ax.annotate(f"{y.iloc[best]:.3f}", (x.iloc[best], y.iloc[best]),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=9, color=ink)
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        lo, hi = np.nanmin(y), np.nanmax(y)
        pad = (hi - lo) * 0.25 or 0.01
        ax.set_ylim(lo - pad, hi + pad)
    axes[-1].set_xlabel("群數 k" if zh else "Number of clusters k", color=muted, fontsize=10)
    axes[-1].set_xticks(metrics["k"])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return zh


def print_checkpoint(n, k_range, metrics, results, png, zh_font, algorithm, sampled):
    print("=" * 60)
    print("檢查點 2　選定群數")
    print("=" * 60)
    print(f"文件數 {n}；分群演算法 {ALGORITHMS[algorithm]}，{N_INIT} 次初始化取成本最低，seed={SEED}")
    print(f"候選群數 k = {k_range[0]} .. {k_range[1]}")
    print()
    tbl = metrics[["k", "silhouette", "npmi", "min_size", "max_size"]].rename(columns={
        "npmi": "NPMI", "min_size": "最小群文件數", "max_size": "最大群文件數"})
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(f"指標圖：{png}" + ("" if zh_font else "（找不到中文字型，圖中文字改用英文）"))

    for r in results:
        print()
        print(f"--- k = {r['k']}（silhouette {r['silhouette']:.3f}，NPMI {r['npmi']:.3f}）")
        for c, (terms, size) in enumerate(zip(r["keywords"], r["sizes"])):
            print(f"  群 {c}（{size} 篇）：{', '.join(terms)}")

    notes = ["silhouette 在文字資料通常偏低，只宜比較不同 k 的相對高低。"]
    if sampled:
        notes.append("文件數超過 5,000 篇，silhouette 以隨機抽樣 5,000 篇計算（seed 固定）。")
    best_sil = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    best_npmi = int(metrics.loc[metrics["npmi"].idxmax(), "k"])
    notes.append(f"silhouette 最高在 k = {best_sil}；NPMI 最高在 k = {best_npmi}。")
    if 2 in (best_sil, best_npmi):
        notes.append("k = 2 的指標常偏高，但通常只是粗略切分，未必有用。")
    if k_range[1] in (best_sil, best_npmi):
        notes.append(f"有指標的最佳值落在範圍上限 k = {k_range[1]}，合適的群數可能更大，可考慮擴大範圍。")
    small = metrics[metrics["min_size"] < 10]
    if len(small):
        notes.append("以下 k 有文件數少於 10 篇的群，其關鍵詞較不穩定："
                     + "、".join(str(int(v)) for v in small["k"]))
    print()
    print("提示：")
    for s in notes:
        print(f"- {s}")


def main(argv=None):
    setup_stdout()
    ap = argparse.ArgumentParser(description="步驟 2　文件分群")
    ap.add_argument("--algorithm", choices=list(ALGORITHMS),
                    help=f"省略時依文件數建議（≤ {KMEDOIDS_MAX} 篇為 kmedoids，否則為 spherical）")
    ap.add_argument("--k-min", type=int)
    ap.add_argument("--k-max", type=int)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    s1 = Path(args.out) / "work" / "step1"
    if not (s1 / "vectors.npy").exists():
        sys.exit("找不到步驟 1 的結果，請先執行 step1_vectorize.py。")
    vectors = np.load(s1 / "vectors.npy")
    counts = sparse.load_npz(s1 / "counts.npz").tocsr()
    vocab = load_json(s1 / "vocab.json")
    n = vectors.shape[0]

    default = default_k_range(n)
    if args.k_min is None and args.k_max is None and default is None:
        sys.exit(f"文件數 {n} 少於 60 篇，無法形成有效的候選群數範圍（平均每群至少約 30 篇）。"
                 "如仍要分群，請以 --k-min、--k-max 自訂範圍。")
    k_min = args.k_min if args.k_min is not None else 2
    k_max = args.k_max if args.k_max is not None else (default[1] if default else k_min)
    if not 2 <= k_min <= k_max < n:
        sys.exit(f"候選群數範圍不合理：k = {k_min} .. {k_max}（需 2 ≤ k_min ≤ k_max < 文件數 {n}）。")
    algorithm = args.algorithm or suggest_algorithm(n)
    if algorithm in LARGE_N_WARNING and n > KMEDOIDS_MAX:
        print(f"注意：文件數 {n} 超過 {KMEDOIDS_MAX}，{ALGORITHMS[algorithm]} 需要大量記憶體與時間。")

    D = cosine_distance_matrix(vectors) if needs_distance_matrix(algorithm, n) else None
    L = build_linkage(vectors) if algorithm == "hierarchical" else None
    results = []
    for k in range(k_min, k_max + 1):
        print(f"分群中：k = {k}", file=sys.stderr, flush=True)
        results.append(run_k(vectors, D, counts, vocab, k, algorithm, L=L))

    metrics = pd.DataFrame([{
        "k": r["k"], "silhouette": r["silhouette"], "npmi": r["npmi"], "cost": r["cost"],
        "min_size": int(r["sizes"].min()), "max_size": int(r["sizes"].max()),
    } for r in results])

    wd = work_dir(args.out, "step2")
    metrics.to_csv(wd / "metrics_by_k.csv", index=False, encoding="utf-8")
    np.savez(wd / "labels.npz", **{f"k{r['k']}": r["labels"] for r in results})
    save_json({str(r["k"]): {
        "medoids": None if r["medoids"] is None else r["medoids"].tolist(),
        "sizes": r["sizes"].tolist(),
        "keywords": r["keywords"],
        "npmi_per_cluster": [None if np.isnan(v) else round(float(v), 4) for v in r["npmi_per_cluster"]],
    } for r in results}, wd / "clusters_by_k.json")
    save_json({
        "algorithm": algorithm,
        "algorithm_name": ALGORITHMS[algorithm],
        "distance": "cosine",
        "n_init": N_INIT,
        "seed": SEED,
        "k_min": k_min,
        "k_max": k_max,
        "n_docs": n,
    }, wd / "config.json")
    png = wd / "k_selection.png"
    zh_font = plot_k_selection(metrics, png)
    dendro = None
    if L is not None:
        np.save(wd / "linkage.npy", L)
        top = results[-1]
        Z = cluster_linkage(L, top["labels"], top["k"])
        dendro = wd / "dendrogram.png"
        font = cjk_font()
        leaf = [f"{', '.join(kw[:2])}（{s}）" if font else f"{c} ({s})"
                for c, (kw, s) in enumerate(zip(top["keywords"], top["sizes"]))]
        plot_dendrogram(Z, leaf, dendro,
                        f"切成 {top['k']} 群時各群的合併關係（前 2 個關鍵詞（文件數））" if font
                        else f"Merge tree of the {top['k']} clusters", font)
    elif (wd / "linkage.npy").exists():
        (wd / "linkage.npy").unlink()

    print_checkpoint(n, (k_min, k_max), metrics, results, png, zh_font, algorithm, D is None)
    if dendro is not None:
        print(f"樹狀圖（切成 k = {k_max} 群時，各群的上下層關係；較小的 k 即由這棵樹往上合併而成）：{dendro}")
    print()
    print(f"中間結果已存於 {wd}")


if __name__ == "__main__":
    main()
