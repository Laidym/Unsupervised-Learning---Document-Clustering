# CLAUDE.md

本 repo 開發一個 Claude skill：對一批沒有標籤的英文或中文文件做非監督分群，萃取各群關鍵詞供使用者命名，並評估結果。這是課堂專案，需與另一組的「多變量分析」skill 及組員的通用分群 skill 做出區隔，重點在**文字資料特有的處理與可解釋性**。

使用說明的唯一原始檔是 `README.qmd`（Quarto）。本檔記錄實作所需的規格與已做的設計決定；兩者衝突時，先與使用者確認。

## 設計原則

1. **可解釋優先**：分群預設 k-medoids（群中心是真實文件），每群以關鍵詞描述。
2. **關鍵決策由使用者做**：skill 提供選項、優缺點與建議值，在檢查點停下等待確認，不全自動完成。沒有判斷依據的技術細節（如 SVD 維度）一律固定，不交給使用者。
3. **評估與限制並陳**：輸出指標，並依資料主動提示限制。

## 流程（四步驟）

讀取資料時，請使用者指定**文字欄位**與**標籤欄位**（無標籤可略過）。也接受 .txt 資料夾或 zip（每個檔案一篇文件，`common.read_table`）：文件編號為相對路徑，有子資料夾時另產生 `folder` 欄，由使用者決定是否當標籤欄。

### 步驟 1　前處理與向量化

固定設定：

- 語言：依中文字元比例自動判斷。
- 中文：內附 jieba 0.42.1 精確模式＋大詞典 `dict.txt.big`（繁簡體一律使用），只留兩字以上中文詞與 ≥ 2 字母英文詞（捨棄單字詞、數字、標點）；內建中文停用詞 `scripts/stopwords_zh.txt`（自行整理，繁簡並列）＋英文停用詞。實作在 `scripts/tokenize_zh.py`，vendor 來源與刪減記錄於 `vendor/README.md`。英文：先移除 email、網址、主機名稱（只認 edu/com/org/net/gov/mil 結尾，避免誤刪檔名）、Usenet 群組名稱（alt/comp/misc/news/rec/sci/soc/talk 開頭，排除常見副檔名；20NG 中也等於洩漏類別），依空白、標點與數字切分（只留 ≥ 2 個字母的詞），內建英文停用詞（scikit-learn 清單＋縮寫殘片 don、ve 等）。
- TF-IDF 加權 → Truncated SVD 降至 100 維（文件數 ≤ 100 時改為 n − 1）→ 向量長度正規化為 1。
- 20 Newsgroups 一律移除 headers、footers、quotes。

可選設定（預設值標 ★）：

- 詞彙篩選：★ `min_df=5, max_df=0.95`（文件數 < 1,000 時 ★ 改為 `min_df=2`，省略參數即自動採用）；不篩選。
- 重跑步驟 1 時，腳本自動列出與上一次執行的對照；選雙詞時另列最高頻雙詞。
- 疑似版面文字：以換行與句末標點切句，≥ 10 字的句子出現在 ≥ 20% 且 ≥ 3 篇文件中即在檢查點 1 列出（`repeated_sentences`）。只提示，由使用者自行從原始資料移除；不自動刪除，也不視為停用詞（維持「不自訂停用詞」）。
- 詞組範圍：★ 單詞；單詞＋雙詞。

**檢查點 1**：回報文件數、詞彙量、最高頻詞、平均文件長度，等使用者確認或調整。

### 步驟 2　文件分群

- 距離：cosine（向量已正規化）。
- 演算法：★ 文件數 ≤ 5,000 用 k-medoids；> 5,000 用球面 k-means；階層式分群為選用。演算法需使用者確認。
- 候選群數：預設 `k = 2 .. min(15, floor(n/30))`，使用者可自訂。n < 60 時無有效範圍，提示資料量不足。
- 對每個候選 k 分群，計算 silhouette 與 NPMI，畫在同一張圖，並列出各 k 的群關鍵詞。檢查點 2 以「群數增加時的變化」摘要呈現（`describe_transition`：每個新群歸給重疊最多的舊群，列出拆開、消失、主題改變〔前 5 個關鍵詞相同 < 3〕），完整關鍵詞另列。

**檢查點 2**：使用者選定群數。提醒：k = 2 指標常偏高但未必有用；最佳值落在上限時應考慮擴大範圍。

### 步驟 3　群關鍵詞

- 固定用 c-TF-IDF 取每群前 10 個關鍵詞，無可選設定。
- skill（Claude）只依每群 10 個關鍵詞提出一個建議名稱（繁中、不重複、不參考標籤或資料集類別名、不讀文件內容）；使用者逐群採用或自訂。名稱來源（採用建議／自訂）記錄於 `names.json` 與報告。

### 步驟 4　評估與限制

- 內部指標（固定）：silhouette、關鍵詞一致性 NPMI。
- 事後比對（自動）：有指定標籤欄才計算 ARI、NMI、純度；否則略過。
- 限制：固定說明（硬歸屬、詞袋忽略語序、群數無唯一解、群名由人決定〔可採用建議名稱〕），並依資料提示：有效詞數中位數 < 10、群過小、群大小差 > 5 倍、某群 NPMI 低於各群平均 0.1 以上、silhouette < 0.1、篩選後無詞文件、k 落在上限、單一文件占某群關鍵詞出現次數 > 30%、兩群前 10 關鍵詞相同 ≥ 5 個、中英混雜（判英文但中文字元 > 5%，或判中文但中文字元 < 70%）、空白與重複列 > 10%。

## 硬性規則

- **標籤不得用於步驟 1–3**，包括選群數。只用於步驟 4 事後比對。
- 所有隨機性固定 seed，並記錄於報告，確保可重現。
- 使用者看到的文字一律繁體中文。

## 演算法規格

- **c-TF-IDF**（Grootendorst 2022）：每群文件合併為一份；`W[t,c] = tf[t,c] * log(1 + A / f[t])`。`tf[t,c]` 為詞 t 在群 c 的次數（依群總詞數正規化），`A` 為每群平均詞數，`f[t]` 為詞 t 在所有群的總次數。用原始詞頻計數（CountVectorizer），詞彙與 TF-IDF 相同。
- **NPMI**（Bouma 2009；Newman et al. 2010）：以「文件內是否共同出現」估計機率，資料為本次語料。每群前 10 個關鍵詞兩兩計算 `NPMI = log(p(i,j) / (p(i) p(j))) / (−log p(i,j))`；`p(i,j) = 0` 時記為 −1。先對群內所有詞對取平均，再對所有群取平均。
- **silhouette**：`sklearn.metrics.silhouette_score(D, labels, metric="precomputed")`，D 為 cosine 距離矩陣。文件數 > 5,000 時不建距離矩陣，改用 `silhouette_score(X, labels, metric="cosine", sample_size=5000, random_state=seed)` 抽樣計算，並在檢查點與報告註明。文字資料數值通常偏低，報告須說明只宜比較相對高低。
- **k-medoids**：自行實作於 `scripts/kmedoids.py`，不依賴 `kmedoids` 或 `scikit-learn-extra`（沙盒沒有）。已實作 k-medoids++ 初始化＋FasterPAM 交換（Schubert & Rousseeuw 2021），5 次初始化取總成本最低者；群編號依 medoid 文件索引排序。
- **球面 k-means**：自行實作於 `scripts/spherical_kmeans.py`（Dhillon & Modha 2001）：依 cosine 相似度分配、中心為群內向量和再正規化為 1；k-means++ 初始化，5 次初始化取總 cosine 距離最小者；群編號依群內最小文件索引排序。不用 sklearn KMeans（它不會重新正規化中心，只是近似）。演算法統一介面在 `scripts/clustering.py`。
- **階層式分群**：`scripts/hierarchical.py`，scipy `linkage(X, "ward")` 作用於單位向量（歐氏距離² = 2 − 2cos），連結法固定為 Ward（開發時比較：average 在 20NG 小群數時幾乎全併成一群、complete 分不開，Ward 群大小較均衡且 silhouette 最高）。樹建一次，以 `cut_tree` 依 k 切割；步驟 2 存 `linkage.npy` 並畫上限 k 的樹狀圖，以虛線標出各候選 k 的切割位置（`cut_positions`），檢查點 2 與指標一併呈現；步驟 3 印出 k 群的合併順序並畫樹狀圖；報告列出合併順序，並在「群數的選擇」放入以實線標出選定 k 的樹狀圖。不依合併距離自動推薦群數（等同手肘法）。
- **事後比對**：`adjusted_rand_score`、`normalized_mutual_info_score`；純度自行計算（每群取多數標籤，加總後除以 n）。

## 執行環境限制（重要）

skill 最終在 claude.ai 的沙盒執行，**通常沒有網路**，不能 pip install，也不能下載資料。因此：

- 只能依賴沙盒內建套件。實測過的版本：Python 3.13、numpy 2.5.3、scipy 1.18.1、pandas 3.0.5、scikit-learn 1.9.1、matplotlib 3.11.2、openpyxl 3.1.5。**jieba、kmedoids、scikit-learn-extra 都沒有。**
- jieba（MIT 授權，純 Python）連同詞典放進 `vendor/jieba/`，用相對路徑匯入。
- 20 Newsgroups 事先整理成小型子集 CSV 放進 `data/`（已移除 headers／footers／quotes，附原始類別欄作為標籤）。
- 完成後必須在 claude.ai 沙盒實測一次，確認沒有依賴網路。

## 建議的 repo 結構

```
SKILL.md            給 Claude 的執行指令（含檢查點互動方式）
text-clustering.zip 安裝用的 skill 套件（tools/build_skill.py 產生）
README.qmd          使用說明原始檔；references.bib、references.csl 為其文獻與樣式
README.md           由 README.qmd 產生（gfm），GitHub 首頁顯示用，勿直接編輯
CLAUDE.md           本檔
scripts/            分析程式（各步驟可獨立執行，中間結果存檔以支援回到前一步重做）
vendor/jieba/       內附中文斷詞
data/               20 Newsgroups 範例子集（打包進 skill）；bbc-fulltext.zip、NPA_165.csv 為開發測試資料（不打包，見 data/README.md）
tests/              測試
```

輸出（`output/`）：`report.html`、`clusters.csv`（doc_id, text, cluster, 群名稱；依群穩定排序，同群文件相鄰）；輸入為 .txt 資料夾或 zip 時另有 `clusters_files.zip`（每群一個資料夾「群編號_群名稱」，檔名為原始相對路徑，子資料夾以 `__` 連接）。群摘要與指標只放在報告中，不另輸出 CSV（指標的中間結果在 `work/step4/metrics.csv`）。

## 開發順序

1. **最小可用版本**：只支援英文與 20 Newsgroups 子集、只用 k-medoids、報告先輸出文字與 CSV。目標是驗證四個步驟與兩個檢查點的互動可行。
2. 加入中文（內附 jieba）、球面 k-means、階層式分群。
3. HTML 報告、限制自動提示、沙盒實測。

## 已否決的方案（不要再提出，除非使用者主動問）

- **NMF／LDA／LSA 主題模型**：分群即主題（BERTopic、Zhang et al. 2022 的做法），另做主題模型與分群重複；軟歸屬的損失列為限制。
- **TextRank 關鍵詞**：沒有區辨力，不利命名；短文效果差。
- **代表句、代表文件**：使用者決定只用關鍵詞命名。
- **分群穩定性、關鍵詞多樣性、手肘法**：精簡指標；多樣性在 c-TF-IDF 下必然偏高，手肘在高維文字資料常不明顯。
- **讓使用者選 SVD 維度、不降維**：使用者在檢查點 1 沒有判斷依據。
- **自訂停用詞、自訂 jieba 詞典、手動指定語言**：使用者決定用固定設定。
- **預訓練語言模型嵌入（如 sentence-transformers）**：沙盒無法下載模型。
- **與組員分群 skill 對接**：改為自己做分群。

## 文件維護

- 修改流程或預設值時，`README.qmd` 與 `SKILL.md` 必須同步更新。
- `README.qmd` 的參考文獻用 `[@key]` 引用 `references.bib`；新增或刪除方法時同步增刪文獻，不保留未引用的條目。
- 表格第一欄有連續相同內容時要合併儲存格：`README.qmd` 用 grid table 的跨列儲存格（Pandoc 3 以上支援）；`SKILL.md` 及與使用者互動時的表格無法合併，重複的內容留空，只寫在該組第一列。
- 產生使用說明：`quarto render README.qmd`，同時輸出 README.html（不納入版本控制）與 README.md（GitHub 格式，納入版本控制，作為 repo 首頁）。修改 README.qmd 後必須重新產生並一起 commit。
- 安裝用的 `text-clustering.zip` 放在 repo 根目錄並納入版本控制，由 `python tools/build_skill.py` 產生（只含 SKILL.md、scripts/、vendor/、data/，換行一律 LF）。修改這些內容後，commit 前必須重新打包，讓 zip 與原始碼一致。
