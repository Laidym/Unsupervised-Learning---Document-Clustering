import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import clustering  # noqa: E402
from clustering import run_clustering, silhouette, suggest_algorithm  # noqa: E402
from spherical_kmeans import spherical_kmeans  # noqa: E402
from step1_vectorize import main as step1_main  # noqa: E402
from step2_cluster import main as step2_main  # noqa: E402
from step3_keywords import main as step3_main  # noqa: E402
from step4_evaluate import main as step4_main  # noqa: E402


def _blobs(seed=0, per=30, dim=5, k=3):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(loc=c, scale=0.3, size=(per, dim)) for c in np.eye(dim)[:k] * 4])
    return normalize(X), np.repeat(np.arange(k), per)


def test_recovers_separated_clusters():
    X, y = _blobs()
    res = spherical_kmeans(X, 3, seed=1)
    assert pd.crosstab(res["labels"], y).max(axis=1).sum() == len(y)


def test_centers_are_unit_normalized_means():
    X, _ = _blobs(seed=2)
    res = spherical_kmeans(X, 3)
    assert np.allclose(np.linalg.norm(res["centers"], axis=1), 1.0)
    for j in range(3):
        mean = X[res["labels"] == j].sum(axis=0)
        assert np.allclose(res["centers"][j], mean / np.linalg.norm(mean))
    sims = X @ res["centers"].T                                  # 收斂：每篇都在最相似的中心
    assert np.array_equal(np.argmax(sims, axis=1), res["labels"])
    assert res["cost"] == pytest.approx(np.sum(1 - sims.max(axis=1)))


def test_reproducible_and_relabelled():
    X, _ = _blobs(seed=3)
    a, b = spherical_kmeans(X, 3, seed=5), spherical_kmeans(X, 3, seed=5)
    assert np.array_equal(a["labels"], b["labels"])
    firsts = [np.flatnonzero(a["labels"] == j).min() for j in range(3)]
    assert firsts == sorted(firsts)


def test_suggest_algorithm():
    assert suggest_algorithm(5000) == "kmedoids"
    assert suggest_algorithm(5001) == "spherical"


def test_run_clustering_interface():
    X, _ = _blobs()
    assert run_clustering(X, 3, "spherical")["medoids"] is None
    assert len(run_clustering(X, 3, "kmedoids")["medoids"]) == 3
    with pytest.raises(ValueError):
        run_clustering(X, 3, "nope")


def test_silhouette_sampling(monkeypatch):
    X, y = _blobs(per=40)
    full, sampled = silhouette(X, y)
    assert not sampled
    monkeypatch.setattr(clustering, "SILHOUETTE_FULL_MAX", 50)
    approx, sampled = silhouette(X, y, seed=0)
    assert sampled and approx == pytest.approx(full, abs=0.1)


def test_pipeline_with_spherical(tmp_path, capsys):
    step1_main(["--dataset", "20ng", "--out", str(tmp_path)])
    step2_main(["--algorithm", "spherical", "--k-min", "3", "--k-max", "4", "--out", str(tmp_path)])
    assert "球面 k-means" in capsys.readouterr().out
    wd2 = tmp_path / "work" / "step2"
    assert json.loads((wd2 / "config.json").read_text(encoding="utf-8"))["algorithm"] == "spherical"
    assert json.loads((wd2 / "clusters_by_k.json").read_text(encoding="utf-8"))["4"]["medoids"] is None
    step3_main(["--k", "4", "--names", "a", "b", "c", "d", "--out", str(tmp_path)])
    kw = json.loads((tmp_path / "work" / "step3" / "keywords.json").read_text(encoding="utf-8"))
    assert kw["medoid_doc_ids"] is None
    step4_main(["--out", str(tmp_path)])
    assert "球面 k-means（k-means++ 初始化）" in (tmp_path / "report.html").read_text(encoding="utf-8")


def test_step2_default_algorithm_is_kmedoids_for_small_data(tmp_path):
    step1_main(["--dataset", "20ng", "--out", str(tmp_path)])
    step2_main(["--k-min", "3", "--k-max", "3", "--out", str(tmp_path)])
    cfg = json.loads((tmp_path / "work" / "step2" / "config.json").read_text(encoding="utf-8"))
    assert cfg["algorithm"] == "kmedoids"
