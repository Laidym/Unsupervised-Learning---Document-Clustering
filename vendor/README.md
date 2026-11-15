# vendor/

claude.ai 沙盒沒有網路、不能 pip install，因此把需要的第三方套件放在這裡，由 `scripts/tokenize_zh.py` 以相對路徑匯入。

## jieba 0.42.1（MIT 授權，見 `jieba/LICENSE`）

- 來源：PyPI `jieba-0.42.1.tar.gz`（sha256 `055ca12f62674fafed09427f176506079bc135638a14e23e25be909131928db2`）
- 只保留精確模式斷詞需要的部分：`__init__.py`、`_compat.py`、`finalseg/`（HMM 新詞識別，只留 `.py` 機率表）
- 移除：預設詞典 `dict.txt`（改用大詞典）、`analyse/`、`posseg/`、`lac_small/`（paddle 模式）、`finalseg/*.p`
- 詞典：`dict.txt.big` 取自 jieba GitHub repo 的 `extra_dict/`（同為 MIT），對繁體中文的斷詞較預設詞典好；繁簡體一律使用此詞典
- 程式碼未修改。jieba 的部分正規表示式字串在 Python 3.12+ 會觸發 `SyntaxWarning`，由 `tokenize_zh.py` 在匯入時忽略，不影響結果
