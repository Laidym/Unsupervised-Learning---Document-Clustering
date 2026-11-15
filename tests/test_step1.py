import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import load_docs  # noqa: E402
from step1_vectorize import (  # noqa: E402
    clean_en, default_min_df, detect_language, main, summarize, svd_dim, vectorize,
)


def test_default_min_df():
    assert default_min_df(800) == 2
    assert default_min_df(1000) == 5


def test_summary_lists_bigrams_only_when_present():
    docs = pd.DataFrame({"text": TEXTS})
    uni = vectorize(TEXTS, "en", min_df=1, max_df=1.0)
    assert "最高頻雙詞" not in summarize(docs, uni["counts"], uni["vocab"])
    bi = vectorize(TEXTS, "en", min_df=1, max_df=1.0, ngram_max=2)
    s = summarize(docs, bi["counts"], bi["vocab"])
    assert s["雙詞數"] > 0 and all(" " in r["詞"] for r in s["最高頻雙詞"])


def test_rerun_prints_comparison_and_auto_min_df(tmp_path, capsys):
    main(["--dataset", "20ng", "--out", str(tmp_path)])
    first = capsys.readouterr().out
    assert "min_df=2" in first and "自動設定" in first and "與上一次執行比較" not in first
    main(["--dataset", "20ng", "--min-df", "5", "--out", str(tmp_path)])
    second = capsys.readouterr().out
    assert "與上一次執行比較" in second and "自動設定" not in second


def test_clean_en_removes_addresses_keeps_filenames():
    s = clean_en("Mail me@cs.cmu.edu or see ftp.cica.indiana.edu/pub, "
                 "http://x.org/a and www.nasa.gov. Open image.GIF now")
    assert not {"cmu", "edu", "indiana", "pub", "org", "nasa", "gov"} & set(s.split())
    assert "image.gif" in s and "mail" in s and "now" in s

TEXTS = [
    "The rocket launch to orbit was delayed. NASA don't know why.",
    "Space shuttle orbit and rocket engines, launch 1993",
    "Baseball pitcher threw a fast ball, the team won the game",
    "The team lost the baseball game after the pitcher left",
    "God and religion: belief, faith and the bible",
    "Faith in God, the bible says, is belief",
]


def test_detect_language():
    assert detect_language(["Hello world, this is English."])[0] == "en"
    assert detect_language(["這是一段中文文字，用來測試語言判斷。"])[0] == "zh"


def test_svd_dim():
    assert svd_dim(5000, 20000) == 100
    assert svd_dim(80, 20000) == 79
    assert svd_dim(80, 10) == 9


def test_vectorize_tokens_and_norm():
    info = vectorize(TEXTS, "en", min_df=1, max_df=1.0)
    vocab = set(info["vocab"])
    assert {"rocket", "baseball", "bible"} <= vocab
    assert not vocab & {"the", "don", "1993", "a"}          # 停用詞、縮寫殘片、數字、單字母
    assert info["svd_dim"] == len(TEXTS) - 1
    assert np.allclose(np.linalg.norm(info["vectors"], axis=1), 1.0)
    assert info["counts"].shape == (len(TEXTS), len(info["vocab"]))


def test_vectorize_bigrams():
    info = vectorize(TEXTS, "en", min_df=1, max_df=1.0, ngram_max=2)
    assert "rocket launch" in info["vocab"]


def test_docs_without_terms_are_dropped():
    texts = TEXTS + ["the and of 123"]
    info = vectorize(texts, "en", min_df=1, max_df=1.0)
    assert list(info["keep"]) == list(range(len(TEXTS)))


def test_load_docs_cleans(tmp_path):
    f = tmp_path / "in.csv"
    pd.DataFrame({"t": ["a b", "", "a b", "c d"], "y": ["x", "x", "x", "z"]}).to_csv(f, index=False)
    docs, stats = load_docs(f, "t", "y")
    assert len(docs) == 2
    assert stats == {"原始列數": 4, "刪除空白列": 1, "刪除重複文件": 1}


def test_load_docs_single_column_csv(tmp_path):
    f = tmp_path / "in.csv"
    pd.DataFrame({"t": ["hello world", "foo bar"]}).to_csv(f, index=False)
    docs, _ = load_docs(f, "t")
    assert list(docs["text"]) == ["hello world", "foo bar"]


def test_load_docs_missing_column(tmp_path):
    f = tmp_path / "in.csv"
    pd.DataFrame({"t": ["a"]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="找不到欄位"):
        load_docs(f, "text")


def test_main_writes_outputs_and_separates_labels(tmp_path):
    main(["--dataset", "20ng", "--out", str(tmp_path)])
    wd = tmp_path / "work" / "step1"
    docs = pd.read_csv(wd / "docs.csv")
    assert list(docs.columns) == ["doc_id", "text"]          # 標籤不放在步驟 2–3 會讀的檔案
    labels = pd.read_csv(wd / "labels.csv")
    assert (labels["doc_id"] == docs["doc_id"]).all()
    assert np.load(wd / "vectors.npy").shape == (len(docs), 100)


def test_main_is_reproducible(tmp_path):
    main(["--dataset", "20ng", "--out", str(tmp_path / "a")])
    main(["--dataset", "20ng", "--out", str(tmp_path / "b")])
    a = np.load(tmp_path / "a" / "work" / "step1" / "vectors.npy")
    b = np.load(tmp_path / "b" / "work" / "step1" / "vectors.npy")
    assert np.array_equal(a, b)
