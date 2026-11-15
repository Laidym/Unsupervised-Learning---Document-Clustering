"""中文斷詞：內附 jieba（精確模式）＋內附大詞典 dict.txt.big＋內建中文停用詞。

規則（固定）：
- 先移除 email、網址與主機名稱（與英文相同），英文字母轉小寫。
- jieba 精確模式斷詞（含 HMM 新詞識別）。
- 保留：兩字以上的中文詞，以及兩個字母以上的英文詞；
  單字詞、數字、標點與其他符號一律捨棄。
- 去除內建中文停用詞與英文停用詞。
- 詞組範圍為單詞＋雙詞時，以去除停用詞後相鄰的兩詞組成雙詞（中間以空白分隔）。
"""
import logging
import re
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "vendor"
JIEBA_DICT = VENDOR / "jieba" / "dict.txt.big"
STOPWORDS_ZH_FILE = HERE / "stopwords_zh.txt"

_CJK_WORD = re.compile(r"^[㐀-䶿一-鿿豈-﫿]{2,}$")
_LATIN_WORD = re.compile(r"^[a-z]{2,}$")

_jieba = None


def get_jieba():
    """匯入內附的 jieba 並載入大詞典（只做一次）。"""
    global _jieba
    if _jieba is None:
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        with warnings.catch_warnings():
            # jieba 0.42.1 的正規表示式字串在 Python 3.12+ 會觸發 SyntaxWarning，不影響結果
            warnings.simplefilter("ignore", SyntaxWarning)
            import jieba
        jieba.setLogLevel(logging.WARNING)
        jieba.set_dictionary(str(JIEBA_DICT))
        jieba.initialize()
        _jieba = jieba
    return _jieba


def load_zh_stopwords():
    words = set()
    for line in STOPWORDS_ZH_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line)
    return frozenset(words)


def keep_token(tok, stop_words):
    return (bool(_CJK_WORD.match(tok)) or bool(_LATIN_WORD.match(tok))) and tok not in stop_words


class ZhAnalyzer:
    """供 CountVectorizer(analyzer=...) 使用的斷詞函式。"""

    def __init__(self, ngram_max, clean, en_stop_words):
        self.ngram_max = ngram_max
        self.clean = clean
        self.stop_words = load_zh_stopwords() | en_stop_words

    def __call__(self, text):
        jieba = get_jieba()
        toks = [t.strip() for t in jieba.cut(self.clean(text))]
        toks = [t for t in toks if keep_token(t, self.stop_words)]
        if self.ngram_max == 2:
            toks = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        return toks
