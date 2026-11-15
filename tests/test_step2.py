import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from kmedoids import cosine_distance_matrix, kmedoids  # noqa: E402
from step1_vectorize import main as step1_main  # noqa: E402
from step2_cluster import default_k_range, main as step2_main  # noqa: E402
from textmetrics import cluster_npmi, ctfidf, npmi_matrix, top_terms  # noqa: E402


def _cost(D, medoids):
    return D[:, medoids].min(axis=1).sum()


# ---------- k-medoids ----------

def test_kmedoids_reaches_global_optimum_small():
    rng = np.random.default_rng(1)
    D = cosine_distance_matrix(normalize(rng.normal(size=(14, 4))), np.float64)
    opt = min(_cost(D, list(c)) for c in itertools.combinations(range(14), 3))
    assert kmedoids(D, 3)["cost"] == pytest.approx(opt)


@pytest.mark.parametrize("seed", range(10))
def test_kmedoids_is_swap_optimal(seed):
    rng = np.random.default_rng(seed)
    D = cosine_distance_matrix(normalize(rng.normal(size=(40, 5))), np.float64)
    k = 4
    res = kmedoids(D, k, seed=seed, n_init=1)
    m = list(res["medoids"])
    c = _cost(D, m)
    assert res["cost"] == pytest.approx(c)
    for i in range(k):
        for x in set(range(40)) - set(m):
            mm = m.copy()
            mm[i] = x
            assert _cost(D, mm) >= c - 1e-9


def test_kmedoids_labels_and_reproducibility():
    rng = np.random.default_rng(0)
    X = normalize(np.vstack([rng.normal(loc=c, size=(20, 3)) for c in np.eye(3) * 5]))
    D = cosine_distance_matrix(X)
    a, b = kmedoids(D, 3, seed=7), kmedoids(D, 3, seed=7)
    assert np.array_equal(a["labels"], b["labels"])
    assert sorted(np.bincount(a["labels"])) == [20, 20, 20]
    assert list(a["medoids"]) == sorted(a["medoids"])
    assert all(a["labels"][m] == i for i, m in enumerate(a["medoids"]))


# ---------- c-TF-IDF ----------

def test_ctfidf_matches_formula():
    counts = sparse.csr_matrix(np.array([[2, 0, 1], [1, 1, 0], [0, 3, 1]]))
    labels = np.array([0, 0, 1])
    W = ctfidf(counts, labels, 2)
    C = np.array([[3, 1, 1], [0, 3, 1]], dtype=float)
    A = C.sum() / 2
    f = C.sum(axis=0)
    expected = C / C.sum(axis=1, keepdims=True) * np.log(1 + A / f)
    assert np.allclose(W, expected)


def test_top_terms_skip_zero_weight():
    counts = sparse.csr_matrix(np.array([[2, 0], [0, 3]]))
    kw = top_terms(counts, np.array([0, 1]), 2, ["a", "b"])
    assert kw[0]["terms"] == ["a"] and kw[1]["terms"] == ["b"]


# ---------- NPMI ----------

def test_npmi_values():
    # 4 篇文件；詞 0、1 永遠一起出現；詞 2 只與詞 0 出現在同一篇一次；詞 3 從不與詞 0 共現
    B = sparse.csc_matrix(np.array([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
    ], dtype=float))
    M = npmi_matrix(B, [0, 1, 2, 3])
    assert M[0, 1] == pytest.approx(1.0)
    assert M[0, 3] == -1.0
    p02, p0, p2 = 1 / 4, 2 / 4, 2 / 4
    assert M[0, 2] == pytest.approx(math.log(p02 / (p0 * p2)) / -math.log(p02))


def test_cluster_npmi_averages_pairs_then_clusters():
    counts = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]]))
    overall, per = cluster_npmi(counts, [[0, 1], [0, 2]])
    assert per[0] == pytest.approx(1.0) and per[1] == -1.0
    assert overall == pytest.approx(0.0)


# ---------- 步驟 2 ----------

def test_default_k_range():
    assert default_k_range(800) == (2, 15)
    assert default_k_range(200) == (2, 6)
    assert default_k_range(59) is None


def test_step2_end_to_end(tmp_path, capsys):
    step1_main(["--dataset", "20ng", "--out", str(tmp_path)])
    step2_main(["--k-min", "3", "--k-max", "5", "--out", str(tmp_path)])
    wd = tmp_path / "work" / "step2"
    m = pd.read_csv(wd / "metrics_by_k.csv")
    assert list(m["k"]) == [3, 4, 5]
    assert m["silhouette"].between(-1, 1).all() and m["npmi"].between(-1, 1).all()
    labels = np.load(wd / "labels.npz")
    assert set(labels.files) == {"k3", "k4", "k5"}
    assert (wd / "k_selection.png").stat().st_size > 0
    assert "檢查點 2" in capsys.readouterr().out
