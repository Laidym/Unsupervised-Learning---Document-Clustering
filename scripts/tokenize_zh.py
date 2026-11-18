"""斷詞：中英文一律使用同一套流程（不依語言分流）。

規則（固定）：
- 先移除 email、網址、主機名稱與 Usenet 群組名稱，英文字母轉小寫（由呼叫端傳入的 clean 處理）。
- 連續的中文字段交給內附 jieba（精確模式＋大詞典 dict.txt.big）斷詞，只保留兩字以上的詞。
- 其他文字依空白、標點與數字切分，只保留兩個字母以上的詞（含 é、ü 等帶重音的字母）。
  純英文資料的結果與只用英文規則切詞相同。
- 單字詞、數字、標點與其他符號一律捨棄；去除內建中文停用詞與英文停用詞。
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

_CJK_CHARS = "㐀-䶿一-鿿豈-﫿"
_CJK_RUN = re.compile(f"[{_CJK_CHARS}]+")
_CJK_WORD = re.compile(f"^[{_CJK_CHARS}]{{2,}}$")
# 中文以外的文字：由兩個以上字母組成的詞，數字與標點都是分隔符
WORD_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"
_WORD = re.compile(WORD_PATTERN)

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


class TextAnalyzer:
    """供 CountVectorizer(analyzer=...) 使用的斷詞函式，中英文共用。"""

    def __init__(self, ngram_max, clean, en_stop_words):
        self.ngram_max = ngram_max
        self.clean = clean
        self.stop_words = load_zh_stopwords() | en_stop_words

    def tokens(self, text):
        """依原文順序回傳詞（尚未去除停用詞）。只有中文字段才呼叫 jieba。"""
        text = self.clean(text)
        toks, pos = [], 0
        for m in _CJK_RUN.finditer(text):
            toks += _WORD.findall(text[pos:m.start()])
            toks += [t for t in get_jieba().cut(m.group()) if _CJK_WORD.match(t)]
            pos = m.end()
        toks += _WORD.findall(text[pos:])
        return toks

    def __call__(self, text):
        toks = [t for t in self.tokens(text) if t not in self.stop_words]
        if self.ngram_max == 2:
            toks = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        return toks
