import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from step1_vectorize import main as step1_main  # noqa: E402
from step2_cluster import main as step2_main  # noqa: E402
from step3_keywords import main as step3_main, name_sources, validate_names  # noqa: E402


def test_name_sources():
    assert name_sources(["a", "x"], ["a", "b"]) == ["採用建議", "自訂"]
    assert name_sources(["a", "b"], None) == ["自訂", "自訂"]


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    step1_main(["--dataset", "20ng", "--out", str(d)])
    step2_main(["--k-min", "3", "--k-max", "4", "--out", str(d)])
    return d


def test_validate_names():
    assert validate_names([" a ", "b"], 2) == (["a", "b"], [])
    assert validate_names(["a", "a"], 2)[1] == ["a"]
    with pytest.raises(ValueError, match="需要 3 個"):
        validate_names(["a", "b"], 3)
    with pytest.raises(ValueError, match="空白"):
        validate_names(["a", " "], 2)


def test_keywords_without_names(out):
    step3_main(["--k", "4", "--out", str(out)])
    kw = json.loads((out / "work" / "step3" / "keywords.json").read_text(encoding="utf-8"))
    assert kw["k"] == 4 and len(kw["clusters"]) == 4
    assert all(len(c["keywords"]) == 10 for c in kw["clusters"])
    assert sum(c["size"] for c in kw["clusters"]) == 800
    assert not (out / "work" / "step3" / "names.json").exists()


def test_names_write_outputs(out):
    names = ["甲", "乙", "丙", "丁"]
    step3_main(["--k", "4", "--names", *names, "--out", str(out)])
    clusters = pd.read_csv(out / "clusters.csv", encoding="utf-8-sig", dtype={"doc_id": str})
    assert list(clusters.columns) == ["doc_id", "text", "cluster", "群名稱"]
    assert (clusters["群名稱"] == clusters["cluster"].map(dict(enumerate(names)))).all()
    assert not (out / "cluster_summary.csv").exists() and not (out / "clusters_files.zip").exists()
    # 依群排序、群內維持原始順序
    assert clusters["cluster"].is_monotonic_increasing
    docs = pd.read_csv(out / "work" / "step1" / "docs.csv", dtype={"doc_id": str})
    labels = np.load(out / "work" / "step2" / "labels.npz")["k4"]
    expected = docs.assign(cluster=labels).sort_values("cluster", kind="stable")
    assert clusters["doc_id"].tolist() == expected["doc_id"].tolist()


def test_names_file(out, tmp_path):
    f = tmp_path / "names.json"
    f.write_text(json.dumps(["一", "二", "三"], ensure_ascii=False), encoding="utf-8")
    step3_main(["--k", "3", "--names-file", str(f), "--out", str(out)])
    saved = json.loads((out / "work" / "step3" / "names.json").read_text(encoding="utf-8"))
    assert saved["names"] == ["一", "二", "三"] and saved["sources"] == ["自訂"] * 3


def test_suggestions_record_sources(out):
    step3_main(["--k", "3", "--names", "甲", "自填", "丙",
                "--suggestions", "甲", "乙", "丙", "--out", str(out)])
    saved = json.loads((out / "work" / "step3" / "names.json").read_text(encoding="utf-8"))
    assert saved["suggestions"] == ["甲", "乙", "丙"]
    assert saved["sources"] == ["採用建議", "自訂", "採用建議"]
    with pytest.raises(SystemExit, match="需要 3 個建議名稱"):
        step3_main(["--k", "3", "--names", "a", "b", "c", "--suggestions", "a", "--out", str(out)])


def test_missing_k_and_wrong_name_count(out):
    with pytest.raises(SystemExit, match="沒有 k = 9"):
        step3_main(["--k", "9", "--out", str(out)])
    with pytest.raises(SystemExit, match="需要 4 個"):
        step3_main(["--k", "4", "--names", "a", "--out", str(out)])
