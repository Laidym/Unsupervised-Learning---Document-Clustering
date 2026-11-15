import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, is_valid_linkage
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from clustering import run_clustering  # noqa: E402
from hierarchical import build_linkage, cluster_linkage, cut, merge_steps  # noqa: E402
from step1_vectorize import main as step1_main  # noqa: E402
from step2_cluster import main as step2_main  # noqa: E402
from step3_keywords import main as step3_main  # noqa: E402
from step4_evaluate import main as step4_main  # noqa: E402


def _data(seed=0, n=120, dim=6):
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[:4] * 4
    X = centers[rng.integers(4, size=n)] + rng.normal(scale=0.5, size=(n, dim))
    return normalize(X)


def test_cut_gives_k_clusters_and_nested_partitions():
    L = build_linkage(_data())
    for k in range(2, 8):
        a, b = cut(L, k), cut(L, k + 1)
        assert len(np.unique(a)) == k
        assert (pd.crosstab(b, a) > 0).sum(axis=1).max() == 1     # k+1 群是 k 群的細分
    firsts = [np.flatnonzero(cut(L, 5) == j).min() for j in range(5)]
    assert firsts == sorted(firsts)


def test_cluster_linkage_matches_full_tree():
    L = build_linkage(_data(seed=1))
    k = 6
    labels = cut(L, k)
    Z = cluster_linkage(L, labels, k)
    assert is_valid_linkage(Z)
    assert np.allclose(Z[:, 2], L[-(k - 1):, 2])
    for m in range(2, k):                    # 在群間樹切成 m 群 = 完整樹切成 m 群
        coarse = cut_tree(Z, n_clusters=m).ravel()[labels]
        assert (pd.crosstab(coarse, cut(L, m)) > 0).sum(axis=1).max() == 1


def test_merge_steps_text():
    Z = np.array([[0, 1, 1.0, 2], [2, 3, 2.0, 3]])
    assert merge_steps(Z, ["甲", "乙", "丙"]) == [
        "合併 1：甲 ＋ 乙（距離 1.00）", "合併 2：丙 ＋ 合併 1（距離 2.00）"]


def test_run_clustering_hierarchical():
    X = _data()
    res = run_clustering(X, 4, "hierarchical")
    assert res["medoids"] is None and len(np.unique(res["labels"])) == 4 and res["cost"] > 0


def test_pipeline_with_hierarchical(tmp_path, capsys):
    step1_main(["--dataset", "20ng", "--out", str(tmp_path)])
    step2_main(["--algorithm", "hierarchical", "--k-min", "3", "--k-max", "5", "--out", str(tmp_path)])
    wd2 = tmp_path / "work" / "step2"
    assert (wd2 / "linkage.npy").exists() and (wd2 / "dendrogram.png").exists()
    step3_main(["--k", "4", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "各群合併關係" in out and "合併 3：" in out
    assert (tmp_path / "work" / "step3" / "dendrogram.png").exists()
    step3_main(["--k", "4", "--names", "甲", "乙", "丙", "丁", "--out", str(tmp_path)])
    step4_main(["--out", str(tmp_path)])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "階層式分群（Ward 連結）" in report and "上下層關係" in report

    # 改用其他演算法重跑步驟 2：舊的樹要刪除，報告不再有上下層關係
    step2_main(["--algorithm", "spherical", "--k-min", "4", "--k-max", "4", "--out", str(tmp_path)])
    assert not (wd2 / "linkage.npy").exists()
    step3_main(["--k", "4", "--names", "甲", "乙", "丙", "丁", "--out", str(tmp_path)])
    step4_main(["--out", str(tmp_path)])
    assert "上下層關係" not in (tmp_path / "report.md").read_text(encoding="utf-8")
    cfg = json.loads((wd2 / "config.json").read_text(encoding="utf-8"))
    assert cfg["algorithm"] == "spherical"
