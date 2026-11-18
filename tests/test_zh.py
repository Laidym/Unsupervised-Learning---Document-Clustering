import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from step1_vectorize import EN_STOP_WORDS, clean_en, detect_language, main as step1_main  # noqa: E402
from step2_cluster import main as step2_main  # noqa: E402
from step3_keywords import main as step3_main  # noqa: E402
from step4_evaluate import main as step4_main  # noqa: E402
from tokenize_zh import TextAnalyzer, get_jieba, load_zh_stopwords  # noqa: E402

ZH_SAMPLE = ROOT / "tests" / "data" / "zh_sample.csv"


def test_vendored_jieba_is_used():
    jieba = get_jieba()
    assert Path(jieba.__file__).resolve().is_relative_to(ROOT / "vendor")
    assert Path(jieba.dt.dictionary).name == "dict.txt.big"


def test_traditional_segmentation():
    words = list(get_jieba().cut("這款筆記型電腦的螢幕解析度不錯"))
    assert "筆記型電腦" in words and "螢幕" in words


def test_stopwords_file():
    sw = load_zh_stopwords()
    assert {"我們", "我们", "因為", "因为", "但是"} <= sw
    assert not any(w.startswith("#") for w in sw)


def test_analyzer_filters_tokens():
    an = TextAnalyzer(1, clean_en, EN_STOP_WORDS)
    toks = an("我們覺得 iPhone 的相機很好，2023 年售價 30000 元。Email: a@b.com")
    assert "相機" in toks and "售價" in toks and "iphone" in toks
    assert not {"我們", "覺得", "的", "很", "2023", "30000", "年", "元", "com"} & set(toks)


def test_analyzer_bigrams_after_stopword_removal():
    an = TextAnalyzer(2, clean_en, EN_STOP_WORDS)
    toks = an("手機的電池續航力很強")
    assert "手機 電池" in toks and "電池 續航力" in toks


def test_detect_language_mixed():
    assert detect_language(["這是一款 iPhone 手機，相機很好。"])[0] == "zh"
    assert detect_language(["The 手機 is a phone with a great camera and battery."])[0] == "en"
    # 夾雜英文術語與文獻引用的中文論文：英文以詞計數，不因字母多而誤判為英文
    paper = "本研究採用 Transformer 模型（Vaswani et al., 2017）進行迴歸分析 (regression analysis)。"
    assert detect_language([paper])[0] == "zh"


def test_citation_fragments_are_stop_words():
    an = TextAnalyzer(1, clean_en, EN_STOP_WORDS)
    toks = an("如 Smith et al. (2020) 所示，見 Fig. 3 與 pp. 12–15，Vol. 5，doi:10.1000/xyz。")
    assert "smith" in toks and not {"et", "al", "fig", "pp", "vol", "doi"} & set(toks)


def test_chinese_pipeline(tmp_path, capsys):
    out = tmp_path / "out"
    step1_main(["--input", str(ZH_SAMPLE), "--text-col", "內容", "--label-col", "主題",
                "--id-col", "編號", "--out", str(out)])
    s1 = capsys.readouterr().out
    assert "主要語言：中文" in s1 and "字（中位數" in s1
    cfg = json.loads((out / "work" / "step1" / "config.json").read_text(encoding="utf-8"))
    assert cfg["language"] == "zh" and "jieba" in cfg["tokenizer"]

    step2_main(["--k-min", "3", "--k-max", "5", "--out", str(out)])
    m = pd.read_csv(out / "work" / "step2" / "metrics_by_k.csv")
    assert m.loc[m["silhouette"].idxmax(), "k"] == 4      # 合成資料有 4 個主題

    step3_main(["--k", "4", "--names", "甲", "乙", "丙", "丁", "--out", str(out)])
    step4_main(["--out", str(out)])
    metrics = pd.read_csv(out / "work" / "step4" / "metrics.csv", encoding="utf-8-sig")
    ari = metrics.loc[metrics["指標"] == "ARI", "數值"].item()
    assert ari > 0.9
    assert "<dt>斷詞</dt><dd>中文字段以 jieba" in (out / "kmedoids_k4" / "report.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("ngram", ["1", "2"])
def test_chinese_vocab_has_no_single_chars(tmp_path, ngram):
    step1_main(["--input", str(ZH_SAMPLE), "--text-col", "內容", "--ngram", ngram, "--out", str(tmp_path)])
    vocab = json.loads((tmp_path / "work" / "step1" / "vocab.json").read_text(encoding="utf-8"))
    assert all(len(part) >= 2 for t in vocab for part in t.split(" "))
    assert any(" " in t for t in vocab) == (ngram == "2")


def test_english_text_matches_plain_english_tokenizer():
    """中英文共用流程：純英文資料的詞彙與只用英文規則切詞（CountVectorizer）完全相同。"""
    from sklearn.feature_extraction.text import CountVectorizer
    from common import DATA_20NG
    from step1_vectorize import EN_TOKEN_PATTERN, make_vectorizer
    texts = pd.read_csv(DATA_20NG)["text"].astype(str).tolist()[:300]
    plain = CountVectorizer(preprocessor=clean_en, token_pattern=EN_TOKEN_PATTERN,
                            stop_words=sorted(EN_STOP_WORDS), min_df=2)
    ours = make_vectorizer(2, 1.0, 1)
    assert plain.fit(texts).get_feature_names_out().tolist() == ours.fit(texts).get_feature_names_out().tolist()


def test_accented_and_adjacent_words():
    an = TextAnalyzer(1, clean_en, EN_STOP_WORDS)
    toks = an("Müller 的 naïve 方法用在 iPhone手機與 café 調查")
    assert {"müller", "naïve", "café", "iphone", "手機", "方法", "調查"} <= set(toks)
