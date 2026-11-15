"""步驟 3　群關鍵詞與命名。

依使用者在檢查點 2 選定的 k，取步驟 2 的分群結果，以 c-TF-IDF 列出每群前 10 個關鍵詞。
Claude 依關鍵詞為每群提出建議名稱，使用者逐群採用或自訂；確定後以 --names 再執行一次，
寫出結果檔。同時傳入 --suggestions 時，會記錄每個名稱是「採用建議」還是「自訂」。

用法：
  python scripts/step3_keywords.py --k 7
  python scripts/step3_keywords.py --k 7 --names 名稱0 名稱1 ... [--suggestions 建議0 建議1 ...]
  python scripts/step3_keywords.py --k 7 --names-file names.json [--suggestions-file sugg.json]
  （JSON 字串陣列，依群編號排列）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, cjk_font, load_json, save_json, setup_stdout, work_dir  # noqa: E402
from hierarchical import cluster_linkage, merge_steps, plot_dendrogram  # noqa: E402
from textmetrics import TOP_N, top_terms  # noqa: E402


def cluster_keywords(counts, labels, k, vocab):
    sizes = np.bincount(labels, minlength=k)
    return [{
        "cluster": c,
        "size": int(sizes[c]),
        "keywords": kw["terms"],
        "weights": [round(w, 5) for w in kw["weights"]],
    } for c, kw in enumerate(top_terms(counts, labels, k, vocab))]


def validate_names(names, k, what="群名稱"):
    names = [str(s).strip() for s in names]
    if len(names) != k:
        raise ValueError(f"需要 {k} 個{what}，收到 {len(names)} 個。")
    if any(not s for s in names):
        raise ValueError(f"{what}不可為空白。")
    dup = sorted({s for s in names if names.count(s) > 1})
    return names, dup


def name_sources(names, suggestions):
    """名稱與建議相同記為「採用建議」，否則為「自訂」；沒有建議時全部為「自訂」。"""
    if suggestions is None:
        return ["自訂"] * len(names)
    return ["採用建議" if n == s else "自訂" for n, s in zip(names, suggestions)]


def _read_list(inline, path):
    if inline is not None:
        return inline
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return None


def print_keywords(k, rows, n):
    print("=" * 60)
    print(f"步驟 3　群關鍵詞（k = {k}，c-TF-IDF 前 {TOP_N} 名）")
    print("=" * 60)
    for r in rows:
        print(f"群 {r['cluster']}（{r['size']} 篇，{r['size'] / n:.1%}）：{', '.join(r['keywords'])}")


def show_hierarchy(s2, wd, labels, k, leaf_names, title):
    """階層式分群：印出 k 群之間的合併順序，並畫樹狀圖。其他演算法不做任何事。"""
    lp = s2 / "linkage.npy"
    if not lp.exists() or k < 2:
        return
    Z = cluster_linkage(np.load(lp), labels, k)
    font = cjk_font()
    path = wd / "dendrogram.png"
    plot_dendrogram(Z, leaf_names if font else [f"{c}" for c in range(k)], path, title if font else "Merge tree", font)
    print()
    print("各群合併關係（由近到遠；越早合併的群內容越接近）：")
    for line in merge_steps(Z, leaf_names):
        print(f"  {line}")
    print(f"樹狀圖：{path}")


def main(argv=None):
    setup_stdout()
    ap = argparse.ArgumentParser(description="步驟 3　群關鍵詞與命名")
    ap.add_argument("--k", type=int, required=True)
    names_src = ap.add_mutually_exclusive_group()
    names_src.add_argument("--names", nargs="+", help="依群編號排列的群名稱")
    names_src.add_argument("--names-file", help="JSON 字串陣列，依群編號排列")
    sugg_src = ap.add_mutually_exclusive_group()
    sugg_src.add_argument("--suggestions", nargs="+", help="提供給使用者的建議名稱，用來記錄名稱來源")
    sugg_src.add_argument("--suggestions-file", help="建議名稱的 JSON 字串陣列")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    s1, s2 = out / "work" / "step1", out / "work" / "step2"
    if not (s2 / "labels.npz").exists():
        sys.exit("找不到步驟 2 的結果，請先執行 step2_cluster.py。")
    all_labels = np.load(s2 / "labels.npz")
    key = f"k{args.k}"
    if key not in all_labels.files:
        have = sorted(int(f[1:]) for f in all_labels.files)
        sys.exit(f"步驟 2 沒有 k = {args.k} 的結果（已有 k = {have[0]} .. {have[-1]}）。"
                 f"請以包含 {args.k} 的範圍重跑步驟 2。")
    labels = all_labels[key]
    counts = sparse.load_npz(s1 / "counts.npz").tocsr()
    vocab = load_json(s1 / "vocab.json")
    docs = pd.read_csv(s1 / "docs.csv", dtype={"doc_id": str})
    if len(docs) != len(labels):
        sys.exit("步驟 1 與步驟 2 的文件數不一致，步驟 1 重跑後請重跑步驟 2。")
    medoids = load_json(s2 / "clusters_by_k.json")[str(args.k)]["medoids"]

    rows = cluster_keywords(counts, labels, args.k, vocab)
    wd = work_dir(out, "step3")
    save_json({"k": args.k, "clusters": rows,
               "medoid_doc_ids": None if medoids is None else [docs["doc_id"][m] for m in medoids]},
              wd / "keywords.json")
    pd.DataFrame({"doc_id": docs["doc_id"], "cluster": labels}).to_csv(
        wd / "assignments.csv", index=False, encoding="utf-8")

    print_keywords(args.k, rows, len(docs))
    show_hierarchy(s2, wd, labels, args.k,
                   [f"群 {r['cluster']}：{', '.join(r['keywords'][:3])}" for r in rows],
                   f"{args.k} 群的合併關係（標示前 3 個關鍵詞）")

    if args.names is None and args.names_file is None:
        names_path = wd / "names.json"
        if names_path.exists():
            names_path.unlink()            # k 或分群可能已改變，舊名稱作廢
        print()
        print("請為各群提出建議名稱，由使用者逐群採用或自訂後，再以 --names（及 --suggestions）執行一次。")
        return

    raw = _read_list(args.names, args.names_file)
    raw_sugg = _read_list(args.suggestions, args.suggestions_file)
    try:
        names, dup = validate_names(raw, args.k)
        suggestions = None if raw_sugg is None else validate_names(raw_sugg, args.k, "建議名稱")[0]
    except ValueError as e:
        sys.exit(str(e))
    sources = name_sources(names, suggestions)
    save_json({"k": args.k, "names": names, "suggestions": suggestions, "sources": sources},
              wd / "names.json")

    clusters = docs.assign(cluster=labels, 群名稱=[names[c] for c in labels])
    clusters.to_csv(out / "clusters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "cluster": [r["cluster"] for r in rows],
        "群名稱": names,
        "文件數": [r["size"] for r in rows],
        "關鍵詞": [", ".join(r["keywords"]) for r in rows],
    }).to_csv(out / "cluster_summary.csv", index=False, encoding="utf-8-sig")

    print()
    print("群名稱：")
    for c, (s, src) in enumerate(zip(names, sources)):
        print(f"  群 {c}：{s}（{src}）")
    show_hierarchy(s2, wd, labels, args.k, [f"群 {c} {s}" for c, s in enumerate(names)],
                   f"{args.k} 群的合併關係")
    if dup:
        print(f"注意：有重複的群名稱（{'、'.join(dup)}），可考慮合併或改名。")
    print()
    print(f"已寫出 {out / 'clusters.csv'}、{out / 'cluster_summary.csv'}")


if __name__ == "__main__":
    main()
