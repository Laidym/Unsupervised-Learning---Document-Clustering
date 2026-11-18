"""第 3 階段：檢查點 2 的群數變化摘要、Usenet 群組名稱清理、加強的限制提示。"""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from step1_vectorize import clean_en, main as step1_main, repeated_sentences  # noqa: E402
from step2_cluster import describe_transition  # noqa: E402
from step4_evaluate import data_limits, keyword_dominance, shared_keyword_pairs  # noqa: E402


def _res(labels, keywords):
    labels = np.asarray(labels)
    return {"labels": labels, "keywords": keywords,
            "sizes": np.bincount(labels, minlength=len(keywords))}


# ---------- 檢查點 2：群數增加時的變化 ----------

def test_transition_split():
    prev = _res([0, 0, 0, 0, 1, 1], [list("abcde"), list("fghij")])
    cur = _res([0, 0, 2, 2, 1, 1], [list("abcxy"), list("fghij"), list("pqrst")])
    lines = describe_transition(prev, cur)
    assert lines[0].startswith("拆開：[a, b, c]（4 篇）→ [a, b, c]（2 篇）、[p, q, r]（2 篇）")
    assert lines[-1] == "其餘 1 群主題不變"


def test_transition_vanish_and_topic_change():
    prev = _res([0, 0, 1, 1, 2, 2], [list("abcde"), list("fghij"), list("klmno")])
    # 舊群 1 的文件被拆給新群 0、2；新群 1 主題完全不同
    cur = _res([0, 0, 0, 2, 1, 1], [list("abcde"), list("uvwxy"), list("fghij")])
    lines = describe_transition(prev, cur)
    assert any(l.startswith("主題改變：[k, l, m]") for l in lines)
    assert not any(l.startswith("拆開") for l in lines)


def test_transition_unchanged_when_keywords_match():
    prev = _res([0, 0, 1, 1], [list("abcde"), list("fghij")])
    cur = _res([0, 1, 1, 1], [list("abcdz"), list("fghiz")])
    assert describe_transition(prev, cur) == ["其餘 2 群主題不變"]


# ---------- Usenet 群組名稱 ----------

def test_clean_en_removes_usenet_names_keeps_files():
    s = clean_en("See news.answers, sci.space.shuttle and talk.religion.misc; open misc.txt or news.gz")
    for w in ("answers", "shuttle", "religion"):
        assert w not in s
    assert "misc.txt" in s and "news.gz" in s


# ---------- 限制提示 ----------

def _limits(**kw):
    base = dict(n=800, k=4, sizes=[200] * 4, npmi_per=[0.3] * 4, silhouette=0.2,
                eff_median=30, n_dropped=0, k_range=(2, 15))
    return data_limits(**{**base, **kw})


def test_keyword_dominance():
    counts = sparse.csr_matrix(np.array([[9, 0], [1, 0], [0, 5], [0, 5]]))
    dom = keyword_dominance(counts, np.array([0, 0, 1, 1]), [[0], [1]])
    assert dom[0] == (0.9, 2) and dom[1] == (0.5, 2)


def test_dominance_hint():
    notes = _limits(dominance=[(0.1, 200), (0.45, 200), (0.9, 2), (0.2, 200)])
    hit = [s for s in notes if "單一文件" in s]
    assert len(hit) == 1 and "群 1（45%）" in hit[0] and "群 2" not in hit[0]   # 太小的群不提示


def test_shared_keyword_hint():
    kws = [list("abcdefghij"), list("abcdeklmno"), list("pqrstuvwxy"), list("0123456789")]
    assert [(i, j) for i, j, _ in shared_keyword_pairs(kws)] == [(0, 1)]
    assert any("群 0 與群 1" in s for s in _limits(keywords=kws))


def test_mixed_language_hints():
    assert any("中英混雜：中文字元占中文字元與英文詞合計的 12%" in s for s in _limits(language="en", zh_ratio=0.12))
    assert any("中英混雜：中文字元占中文字元與英文詞合計的 60%" in s for s in _limits(language="zh", zh_ratio=0.6))
    assert _limits(language="en", zh_ratio=0.01) == [] and _limits(language="zh", zh_ratio=0.95) == []


def test_removed_rows_hint():
    ls = {"原始列數": 100, "刪除空白列": 5, "刪除重複文件": 10}
    assert any("15 列為空白或重複" in s for s in _limits(load_stats=ls))
    assert _limits(load_stats={"原始列數": 100, "刪除空白列": 1, "刪除重複文件": 2}) == []


# ---------- 疑似版面文字（重複出現的句子） ----------

BOILER = "本文版權所有，未經同意不得轉載。"


def test_repeated_sentences_found():
    docs = [f"第{i}則新聞，談論主題{i}的內容與細節。\n{BOILER}\nRead more news at our site today" for i in range(10)]
    docs += [f"完全不同的第{i}篇文章內容，沒有任何版面文字。" for i in range(3)]
    hits = dict(repeated_sentences(docs))
    assert hits == {BOILER: 10, "Read more news at our site today": 10}


def test_repeated_sentences_ignores_short_and_rare():
    docs = [f"謝謝大家。這是第{i}篇獨立的文章內容，主題各不相同。" for i in range(20)]
    docs += ["這一句只出現在兩篇文件裡面而已。"] * 2
    assert repeated_sentences(docs) == []                       # 「謝謝大家。」太短；第二句太少


def test_checkpoint1_shows_repeated_sentences(tmp_path, capsys):
    import pandas as pd
    src = Path(__file__).resolve().parent / "data" / "zh_sample.csv"
    d = pd.read_csv(src)
    d["內容"] = d["內容"] + "\n" + BOILER
    f = tmp_path / "in.csv"
    d.to_csv(f, index=False)
    step1_main(["--input", str(f), "--text-col", "內容", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert "疑似網站版面文字" in out and f"「{BOILER}」（240 篇，100%）" in out
    step1_main(["--input", str(src), "--text-col", "內容", "--out", str(tmp_path / "o2")])
    assert "疑似網站版面文字" not in capsys.readouterr().out
