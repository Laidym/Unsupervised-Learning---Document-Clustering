# 文本非監督學習 Skill 使用說明


<!-- 本文件為使用說明的唯一原始檔。修改後執行 `quarto render README.qmd`，同時產生 README.html 與 GitHub 首頁顯示的 README.md（勿直接編輯 README.md）。 -->

# 第一章　動機

新聞、評論、問卷開放題、公文等文字資料數量龐大，但大多沒有類別標籤，人工標註又耗時昂貴。面對一批未標註的文件，分析者最常問的問題是：

- 這批文件可以分成哪幾類？
- 每一類在談什麼，該如何命名？
- 這樣的分類可信嗎？

非監督學習不需要標籤，直接從資料本身找出結構。本 skill 採用以分群為基礎的主題分析 (Grootendorst 2022; Zhang et al. 2022)：先將內容相似的文件分為同一群，再從每一群萃取關鍵詞來描述它，一群即對應一個主題。設計上依循三項原則：

1.  **可解釋優先。** 分群以 k-medoids 為預設，群中心是一篇真實文件；每一群以關鍵詞描述，結果可直接閱讀與檢驗。
2.  **關鍵決策由使用者做。** 群數、詞彙篩選等選擇沒有客觀的唯一最佳解，需要領域知識判斷。完全自動化容易產生「看似合理但不可靠」的結果，因此 skill 只提供選項、優缺點與建議，在關鍵處停下來由使用者確認。
3.  **評估與限制並陳。** 非監督結果沒有標準答案，skill 以多種指標交叉檢查，並主動說明結果的限制。

> [!NOTE]
>
> ### 適用範圍
>
> 英文與中文（繁體、簡體）文件，建議至少數百篇。本 skill 採用傳統方法（詞袋表示、矩陣降維、距離式分群），不使用需下載的預訓練語言模型。

# 第二章　分析流程與使用者設定

流程共四個步驟。下圖中，黃色節點為使用者決定或確認的環節，菱形為依資料自動分流之處。

``` mermaid
flowchart TD
  A(["輸入：文件集合"]) --> B["指定文字欄位與標籤欄位<br/>（無標籤可略過）"]:::user

  subgraph S1["步驟 1　前處理與向量化"]
    C{"語言判斷"}
    C -->|中文| D1["jieba 斷詞<br/>中文停用詞"]
    C -->|英文| D2["移除 email 與網址<br/>依空白、標點與數字切分<br/>英文停用詞"]
    D1 --> E["TF-IDF → SVD 降維 → 長度正規化"]
    D2 --> E
    E --> CP1["檢查點 1：確認詞彙摘要<br/>可調整詞彙篩選、詞組範圍"]:::user
  end

  subgraph S2["步驟 2　文件分群"]
    N{"文件數 ≤ 5,000？"}
    N -->|是| M1["建議 k-medoids"]
    N -->|否| M2["建議球面 k-means"]
    M1 --> U["確認分群演算法"]:::user
    M2 --> U
    U --> F["對候選群數分群<br/>計算 silhouette 與 NPMI"]
    F --> CP2["檢查點 2：選定群數"]:::user
  end

  subgraph S3["步驟 3　群關鍵詞"]
    G["c-TF-IDF 萃取各群關鍵詞"] --> NM["為各群命名"]:::user
  end

  subgraph S4["步驟 4　評估與限制"]
    I["內部指標：silhouette、NPMI"] --> J{"是否指定標籤欄？"}
    J -->|是| K["事後比對：ARI、NMI、純度"]
    J -->|否| L["略過事後比對"]
  end

  B --> C
  CP1 --> N
  CP2 --> G
  NM --> I
  K --> R(["輸出：分析報告與結果檔"])
  L --> R

  classDef user fill:#fdf3e1,stroke:#c58a1f,color:#6b4508
```

以下逐步列出可選設定。標示 **（建議）** 者為預設值；使用者在檢查點回覆「使用建議設定」即可沿用。

## 步驟 1　前處理與向量化

將每篇文件轉為一個數值向量，作為分群的依據。

**讀取資料時**，使用者指定要分析的文字欄位，以及標籤欄位（若有）。標籤欄位只用於步驟 4 的事後比對。

**固定設定**：

| 項目 | 中文 | 英文 |
|----|----|----|
| 語言判斷 | 依中文字元比例自動判斷 | 同左 |
| 清理 | 移除 email、網址、主機名稱與 Usenet 群組名稱，英文字母轉小寫 | 移除 email、網址、主機名稱（如 `ftp.cs.cmu.edu`）與 Usenet 群組名稱（如 `news.answers`），轉為小寫 |
| 斷詞 | jieba 精確模式 (Sun, n.d.)，使用對繁體較佳的大詞典；只保留兩字以上的中文詞與兩個字母以上的英文詞 | 依空白、標點與數字切分，只保留兩個字母以上的詞 |
| 停用詞 | 內建中文停用詞清單（本專案整理，繁簡並列） | 內建英文停用詞清單 (<span class="nocase">Pedregosa et al.</span> 2011)，另含縮寫切開後的殘片（如 don’t 的 `don`） |
| 加權 | TF-IDF (Salton and Buckley 1988) | 同左 |
| 降維與正規化 | Truncated SVD 降至 100 維 (Deerwester et al. 1990)，再將向量長度正規化為 1 | 同左 |

**可選設定**：

<table style="width:99%;">
<colgroup>
<col style="width: 7%" />
<col style="width: 37%" />
<col style="width: 26%" />
<col style="width: 28%" />
</colgroup>
<thead>
<tr>
<th>設定</th>
<th>選項</th>
<th>優點</th>
<th>缺點</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">詞彙篩選</td>
<td>min_df = 5、max_df = 0.95 <strong>（文件數 ≥ 1,000 時建議）</strong></td>
<td>去除罕見詞與幾乎每篇都有的詞，降低雜訊</td>
<td>小型語料會刪除過多詞</td>
</tr>
<tr>
<td>min_df = 2、max_df = 0.95 <strong>（文件數 &lt; 1,000 時建議）</strong></td>
<td>小型語料保留較多詞</td>
<td>雜訊較多</td>
</tr>
<tr>
<td>不篩選</td>
<td>保留全部資訊</td>
<td>維度大、雜訊多，分群品質下降</td>
</tr>
<tr>
<td rowspan="2">詞組範圍</td>
<td>單詞 <strong>（建議）</strong></td>
<td>維度較小、結果穩定</td>
<td>無法表示片語（如 <code>new york</code>）</td>
</tr>
<tr>
<td>單詞＋雙詞</td>
<td>能捕捉片語</td>
<td>維度大幅增加；中文經 jieba 斷詞後效益較小</td>
</tr>
</tbody>
</table>

中文捨棄單字詞，是因為單字多為虛詞或量詞（如「的」「很」「個」），保留下來雜訊大於資訊。英文的清理與切詞規則是為了排除不帶主題意義的片段：email 與主機名稱切開後會留下 `edu`、`com` 等詞，Usenet 群組名稱（20 Newsgroups 文中常見）切開後會留下 `news`、`answers` 等詞，且等於洩漏類別名稱；年份與版本號等數字也很少有助於區分主題。

min_df 為整數時代表最少出現篇數，max_df 為小數時代表最多出現比例；兩者皆以「出現在幾篇文件中」計算，而非出現次數。SVD 降至 100 維為經驗值，與 scikit-learn 官方文件分群範例相同 (<span class="nocase">scikit-learn developers</span>, n.d.)；LSA 文獻中常用數百維 (Landauer and Dumais 1997)。文件數不足 101 篇時，自動調為文件數減 1。

> [!NOTE]
>
> ### 20 Newsgroups
>
> 使用 20 Newsgroups (Lang 1995) 時，固定移除標頭、簽名檔與引用段落。這些內容會洩漏類別資訊，使評估結果過於樂觀。

> [!TIP]
>
> ### 檢查點 1
>
> skill 呈現文件數、詞彙量、最高頻詞與平均文件長度。使用者確認詞彙合理（例如沒有大量無意義的高頻詞）後再繼續；必要時調整上述可選設定。

## 步驟 2　文件分群

依文件向量的 cosine 距離，將每篇文件分到一個群。

<table style="width:99%;">
<colgroup>
<col style="width: 6%" />
<col style="width: 22%" />
<col style="width: 33%" />
<col style="width: 36%" />
</colgroup>
<thead>
<tr>
<th>設定</th>
<th>選項</th>
<th>優點</th>
<th>缺點</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">分群演算法</td>
<td>k-medoids <strong>（文件數 ≤ 5,000 時建議）</strong></td>
<td>群中心為真實文件，可直接閱讀；對離群文件穩健</td>
<td>需計算全部文件兩兩距離，文件過多時記憶體不足</td>
</tr>
<tr>
<td>球面 k-means <strong>（文件數 &gt; 5,000 時建議）</strong></td>
<td>速度快，適用大量文件</td>
<td>群中心為平均方向，不對應真實文件；對離群文件較敏感</td>
</tr>
<tr>
<td>階層式分群</td>
<td>可用樹狀圖呈現群的上下層關係；不同群數的結果彼此巢狀，容易比較</td>
<td>需計算全部文件兩兩距離，文件過多時記憶體不足；合併一旦發生就無法修正</td>
</tr>
<tr>
<td rowspan="2">群數</td>
<td>候選範圍評估後由使用者選定 <strong>（建議）</strong></td>
<td>兼顧指標與人工判斷</td>
<td>需執行多次，耗時較長</td>
</tr>
<tr>
<td>使用者直接指定</td>
<td>快速</td>
<td>缺乏依據，可能過多或過少</td>
</tr>
</tbody>
</table>

**候選群數範圍**預設為 $k = 2$ 至 $\min(15,\ \lfloor n/30 \rfloor)$，$n$ 為文件數，使用者可自訂：

- 下限 2：群數 1 不構成分群，silhouette 係數亦無法計算。
- 上限 15：超過此數，逐一檢視與命名的負擔過重（經驗值）。
- $\lfloor n/30 \rfloor$：使平均每群至少約 30 篇，關鍵詞不致受個別文件左右（經驗值）。文件數少於 60 篇時無法形成有效範圍，skill 會提示資料量不足。

> [!TIP]
>
> ### 檢查點 2
>
> skill 對每個候選群數分群，並將 silhouette 係數與關鍵詞一致性（定義見步驟 4）繪於同一張圖，同時列出各候選群數的群關鍵詞。使用者綜合判斷後選定群數。解讀時注意：
>
> - $k = 2$ 時指標常偏高，但通常只是粗略切分，未必有用。
> - 若最佳值落在範圍上限，合適的群數可能更大，應考慮擴大範圍。
>
> 使用階層式分群時，另外呈現樹狀圖，標出各候選群數的切割位置，使用者可對照每增加一群是拆開了哪兩群。樹狀圖只用於呈現，不依合併距離自動推薦群數。

## 步驟 3　群關鍵詞

以 c-TF-IDF (Grootendorst 2022) 為每一群萃取最具代表性的 10 個關鍵詞，作為使用者為各群命名的依據。c-TF-IDF 突顯「此群有、其他群少見」的詞，適合用於區分與命名各群；其限制是群的文件數差異很大時，小群的關鍵詞較不穩定。

> [!TIP]
>
> ### 命名
>
> skill 只依各群的 10 個關鍵詞為每群提出一個建議名稱（不參考標籤、不閱讀文件內容），使用者逐群決定採用建議或自行填寫，最終名稱由使用者決定。報告會記錄每個名稱是採用建議還是自訂。

## 步驟 4　評估與限制

**內部指標**（固定輸出）：

| 指標 | 計算方式 | 解讀 |
|----|----|----|
| silhouette 係數 (Rousseeuw 1987) | 每篇文件比較「與同群文件的平均距離」及「與最近他群文件的平均距離」，再取平均；文件數超過 5,000 篇時，以固定 seed 隨機抽樣 5,000 篇計算 | 範圍 −1 至 1。文字資料數值通常偏低，宜比較不同群數的相對高低 |
| 關鍵詞一致性（NPMI）(Newman et al. 2010; Bouma 2009) | 各群關鍵詞兩兩計算「實際共同出現於同一文件的頻率」相對於「彼此獨立時的預期頻率」，正規化後取平均 | 範圍 −1 至 1。越高表示關鍵詞語意越連貫、越容易命名 |

**事後比對**（自動執行）：若步驟 1 指定了標籤欄位，skill 以 ARI (Hubert and Arabie 1985)、NMI (Strehl and Ghosh 2002) 與純度 (Manning, Raghavan, and Schütze 2008) 量化分群與既有分類的吻合程度；未指定則略過。既有分類只是一種分法，不吻合不代表分群錯誤。

> [!IMPORTANT]
>
> ### 標籤的使用範圍
>
> 即使指定了標籤欄位，步驟 1 至 3（包括檢查點 2 選定群數）都不使用標籤。以標籤挑選群數等同參考答案，會使事後比對失去意義。

**限制**：報告固定說明以下限制，並依本次資料主動提示，例如平均文件過短、某群文件數過少、某群關鍵詞一致性明顯偏低、某群關鍵詞主要來自單一長文件、兩群關鍵詞大量重疊、中英文混雜、空白或重複列過多。

- 每篇文件只能屬於一群；同時涉及多個主題的文件會被歸入單一群。
- 詞袋表示忽略詞序與語意，同義詞視為不同的詞。
- 群數沒有唯一正確答案，指標僅供參考。
- 群名稱由使用者根據關鍵詞決定（可採用 skill 依關鍵詞提出的建議名稱），並非分群模型產出。

# 第三章　使用的非監督學習方法

本章說明各方法與評估指標的核心想法及其在流程中的角色，並附上數學定義（不含推導）。以下以 $n$ 表示文件數、$k$ 表示群數。整體方法可參考文字探勘與資訊檢索的標準教材 (Manning, Raghavan, and Schütze 2008; Aggarwal 2018)。

## 向量化與降維

**TF-IDF**　一個詞在某篇文件中出現越多次越重要，但若它在所有文件中都很常見（如 “the”、「的」），重要性就應降低。TF-IDF 以這兩個因素為每個詞加權，讓具區別力的詞突顯出來 (Salton and Buckley 1988)。本 skill 使用 scikit-learn 的預設公式：

$$
\text{tfidf}(t,d) = \text{tf}(t,d) \times \left( \ln\frac{1+n}{1+\text{df}(t)} + 1 \right),
$$

其中 $\text{tf}(t,d)$ 為詞 $t$ 在文件 $d$ 的出現次數，$\text{df}(t)$ 為含有詞 $t$ 的文件數；每篇文件的向量再除以其長度，使長度為 1。*流程角色：步驟 1，將文件轉為數值向量。*

**Truncated SVD（潛在語意分析）**　完整的 SVD 將矩陣拆解為所有成分，可完全還原原矩陣；Truncated SVD 只計算並保留最重要的前 $k$ 個成分，是以 $k$ 個方向對原矩陣的最佳近似。經常一起出現的詞會被合併到相同方向，因此能部分處理同義用詞並濾除雜訊。它不需先扣除平均值，可直接處理稀疏的文字矩陣，可視為適用於文字資料的 PCA (Deerwester et al. 1990)。*流程角色：步驟 1，產生用於分群的低維向量。*

## 分群

**k-medoids**　選出 $k$ 篇文件作為群中心（medoid），將每篇文件分給最近的中心，再反覆調整中心，使群內距離總和最小 (Kaufman and Rousseeuw 1990; Hastie, Tibshirani, and Friedman 2009; Schubert and Rousseeuw 2021)。因群中心必須是真實文件，可使用任意距離（如 cosine），中心也可直接閱讀。*流程角色：步驟 2，文件數 5,000 篇以內的預設演算法。*

**球面 k-means**　先將每篇文件的向量長度正規化為 1，所有文件即落在單位球面上，分群只依向量方向（cosine 相似度），不受文章長短影響。演算法反覆執行「將文件分給最近的群中心」與「以群內平均更新中心，並將中心長度縮放回 1」(Dhillon and Modha 2001)。*流程角色：步驟 2，文件數超過 5,000 篇的預設演算法。*

**階層式分群**　從每篇文件自成一群開始，每次合併最相似的兩群，直到全部合併，形成樹狀結構 (Steinbach, Karypis, and Kumar 2000)。「最相似的兩群」以 Ward 準則判斷：合併後群內差異增加最少者優先 (Ward 1963)；向量長度已正規化為 1，此準則仍以 cosine 相似度為基礎。樹只建一次，依候選群數切割即得到粗細不同、彼此巢狀的分群，並可畫出各群的合併關係，協助命名。*流程角色：步驟 2 的選用方法。*

## 關鍵詞萃取

**c-TF-IDF**　將同一群的所有文件合併成一份長文件，再以「群」為單位計算 TF-IDF：在此群頻繁出現、在其他群少見的詞得分最高 (Grootendorst 2022)：

$$
W_{t,c} = \text{tf}_{t,c} \times \ln\!\left(1 + \frac{A}{f_t}\right),
$$

其中 $\text{tf}_{t,c}$ 為詞 $t$ 在群 $c$ 的出現次數除以群 $c$ 的總詞數，$A$ 為每群平均詞數，$f_t$ 為詞 $t$ 在所有群的總出現次數。每群取 $W_{t,c}$ 最高的 10 個詞為關鍵詞。*流程角色：步驟 3。*

## 評估指標

**silhouette 係數**　對每篇文件提問：「它離同群的文件比較近，還是離最近的他群文件比較近？」接近 1 表示文件明確屬於自己的群，接近 0 表示位於兩群交界，小於 0 表示可能分錯群；全部文件取平均即為整體指標 (Rousseeuw 1987)。對文件 $i$，令 $a(i)$ 為它與同群其他文件的平均距離，$b(i)$ 為它與「平均距離最小的他群」中文件的平均距離（距離為 cosine 距離 $1-\cos$），則

$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}, \qquad S = \frac{1}{n}\sum_{i=1}^{n} s(i).
$$

只有一篇文件的群，其 $s(i)$ 記為 0。*流程角色：步驟 2 比較候選群數；步驟 4 內部指標。*

**PMI 與 NPMI**　點互資訊（PMI）衡量兩個詞「實際共同出現的機率」比「彼此無關時預期共同出現的機率」高出多少：大於 0 表示兩詞常結伴出現，等於 0 表示彼此無關。PMI 沒有固定範圍，正規化點互資訊（NPMI）將其調整到 −1 至 1：1 表示兩詞總是一起出現，0 表示無關，−1 表示從不一起出現 (Bouma 2009)。一群關鍵詞兩兩 NPMI 的平均越高，代表這組詞描述的是同一件事，也越容易被人理解 (Newman et al. 2010)。機率以本次語料估計：$p(w_i)$ 為含有詞 $w_i$ 的文件比例，$p(w_i, w_j)$ 為同時含有兩詞的文件比例。

$$
\text{PMI}(w_i, w_j) = \ln \frac{p(w_i, w_j)}{p(w_i)\, p(w_j)}, \qquad
\text{NPMI}(w_i, w_j) = \frac{\text{PMI}(w_i, w_j)}{-\ln p(w_i, w_j)}.
$$

$p(w_i, w_j) = 0$ 時 NPMI 記為 $-1$。一群的 NPMI 為其前 10 個關鍵詞 $\binom{10}{2} = 45$ 個詞對的平均，整體 NPMI 再對各群取平均：

$$
\text{NPMI}_c = \frac{1}{45} \sum_{i<j} \text{NPMI}(w_i, w_j), \qquad
\text{NPMI} = \frac{1}{k} \sum_{c=1}^{k} \text{NPMI}_c.
$$

*流程角色：步驟 2 比較候選群數；步驟 4 內部指標。*

**ARI（調整蘭德指數）**　比較兩種分法（分群結果與既有標籤）是否一致：逐一檢查每一對文件，看兩種分法是否都認為它們「同群」或都認為「不同群」，並扣除隨機分群也會碰巧一致的部分。1 表示完全一致，0 相當於隨機分群，也可能小於 0 (Hubert and Arabie 1985)。

以交叉表表示兩種分法：$n_{ij}$ 為群 $i$ 中標籤為 $j$ 的文件數，$a_i = \sum_j n_{ij}$ 為群 $i$ 的文件數，$b_j = \sum_i n_{ij}$ 為標籤 $j$ 的文件數。$\binom{n_{ij}}{2}$ 即「兩種分法都放在一起」的文件對數。

$$
\text{ARI} = \frac{\displaystyle\sum_{ij}\binom{n_{ij}}{2} - E}
{\displaystyle\frac{1}{2}\left[\sum_i\binom{a_i}{2} + \sum_j\binom{b_j}{2}\right] - E},
\qquad
E = \frac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom{n}{2}},
$$

$E$ 為隨機分群（各群、各類別大小不變）時的期望值。*流程角色：步驟 4 事後比對。*

**NMI（正規化互資訊）**　從資訊的角度比較兩種分法：得知一篇文件的群別後，能減少多少對其標籤的不確定性。0 表示兩者毫無關聯，1 表示完全對應 (Strehl and Ghosh 2002)。令 $U$ 為群別、$V$ 為標籤，沿用上式符號：

$$
I(U;V) = \sum_{ij} \frac{n_{ij}}{n} \ln \frac{n\, n_{ij}}{a_i\, b_j}, \qquad
H(U) = -\sum_i \frac{a_i}{n} \ln \frac{a_i}{n}, \qquad
H(V) = -\sum_j \frac{b_j}{n} \ln \frac{b_j}{n},
$$

$$
\text{NMI} = \frac{I(U;V)}{\tfrac{1}{2}\left[H(U) + H(V)\right]}.
$$

本 skill 以算術平均正規化（scikit-learn 預設）；Strehl and Ghosh (2002) 原文使用幾何平均 $\sqrt{H(U)H(V)}$，兩者數值相近。*流程角色：步驟 4 事後比對。*

**純度**　每一群取其中最多數的標籤，將各群的多數文件數加總後除以總文件數。直觀易懂，但群數越多純度越高（每篇文件自成一群時為 1），因此須搭配 ARI 或 NMI 解讀 (Manning, Raghavan, and Schütze 2008)。

$$
\text{純度} = \frac{1}{n} \sum_{i=1}^{k} \max_j n_{ij}.
$$

*流程角色：步驟 4 事後比對。*

# 第四章　Skill 使用方法

## 4.1　執行環境

所有分析在 Claude 的沙盒中執行，使用者的電腦不需安裝任何軟體。沙盒通常無法連網下載套件或資料，因此 skill 內附所需元件：

- 中文斷詞套件 jieba 及其詞典 (Sun, n.d.)。
- 分群演算法（k-medoids、球面 k-means）的實作程式，不依賴額外套件。
- 20 Newsgroups 的範例子集 (Lang 1995)。

其餘計算使用沙盒內建的 NumPy、SciPy、pandas、scikit-learn (<span class="nocase">Pedregosa et al.</span> 2011) 與 matplotlib。

**安裝**：

1.  下載本 repo 根目錄的 `text-clustering.zip`（不需解壓縮）。
2.  在 claude.ai 的「設定 → 功能」開啟程式碼執行與檔案建立。
3.  在同一頁的 Skills 區塊上傳 `text-clustering.zip`。

## 4.2　支援的輸入

一篇文件對應一列或一個檔案即可上傳。

| 格式 | 說明 |
|----|----|
| CSV 或 Excel | 每一列為一篇文件 |
| 純文字檔 | 每個 `.txt` 檔為一篇文件，可上傳整個資料夾或壓縮成 zip；文件編號為檔案的相對路徑（含子資料夾，避免不同資料夾的同名檔案重複） |
| 內建範例 | 20 Newsgroups（英文），可指定要使用的類別 |

HTML 網頁、PDF、Word 等格式需先轉為純文字。skill 會自動處理編碼判斷、空白列與完全相同的重複文件，並移除 email、網址與數字。

> [!WARNING]
>
> ### 從網路下載的文章
>
> 以下內容 skill 不會自動處理，建議上傳前自行清理：
>
> - **網站固定版面文字**：導覽列、「相關新聞」、「延伸閱讀」、版權聲明、記者署名、訂閱提示等。同一網站的文章都有這些文字，可能讓文件依來源網站而非主題分群。步驟 1 會列出出現在 20% 以上文件中的相同句子，提示疑似版面文字，但不會自動刪除。
> - **讀者留言**：常離題，使關鍵詞混雜。
> - **轉載造成的近似重複**：skill 只刪除完全相同的文件，內容幾乎相同的轉載會保留，並使所屬群的關鍵詞偏向該篇。
> - **長度極端的文件**：只有標題或一兩句的短訊分群依據太少；極長的文件（如 FAQ）可能主導所屬群的關鍵詞。報告會提示這兩種情況。
> - **語言**：一次分析一種語言；中英混雜時可能依語言分群。
>
> 若文章原有分類（如新聞版別），可另放一欄作為標籤，只用於最後的事後比對。

**依資料夾分類的 `.txt` 檔**：若文章已依類別放在不同子資料夾（例如 `business/001.txt`、`sport/001.txt`），skill 會把子資料夾名稱整理成 `folder` 欄，並詢問是否當作標籤欄位。與其他標籤一樣，它只用於步驟 4 的事後比對，不影響分群。

## 4.3　啟動

上傳資料後，以自然語言說明需求即可，例如：

``` text
幫我把這份評論資料分群，看看大家在談哪些事情。
用 20 Newsgroups 的 sci.space、rec.sport.baseball、comp.graphics、talk.religion.misc 示範文件分群。
```

## 4.4　互動流程

1.  skill 讀取資料，請使用者指定文字欄位與標籤欄位（無標籤可略過），並回報文件數與判斷的語言。
2.  在檢查點 1、分群演算法確認、檢查點 2 與命名階段，skill 呈現中間結果並等待回覆。可回覆具體設定（例如「群數選 6」），或回覆「使用建議設定」；命名時可回覆「全部採用建議」或「群 2 改成〇〇，其他採用建議」。
3.  使用者可隨時要求回到前面的步驟重做，例如「把 min_df 改成 2 後重新分群」。
4.  完成後輸出報告與結果檔。

## 4.5　輸出

``` text
output/
├── report.html          分析報告：各群名稱、文件數與關鍵詞，評估指標、圖表、限制與設定紀錄
├── clusters.csv         doc_id, text, cluster, 群名稱（依群排序，同群文件相鄰；群內維持原始順序）
└── clusters_files.zip   僅限 .txt 輸入：每群一個資料夾（群編號_群名稱），放入該群的文件
```

各群摘要與評估指標都在 `report.html` 中，不另外輸出 CSV。

報告完整記錄每一步驟採用的設定與隨機種子，以確保結果可以重現。

## 4.6　開發與測試使用的資料

| 資料 | 內容 | 用途 |
|----|----|----|
| 20 Newsgroups (Lang 1995) | 英文新聞群組文章，取 4 類各 200 篇 | 內附於 skill 的範例 |
| BBC News (Greene and Cunningham 2006) | 英文新聞 2,225 篇（2004–2005），依 5 個類別分放於資料夾，每個 `.txt` 檔一篇；取自 <http://mlg.ucd.ie/datasets/bbc.html> | 測試 `.txt` 資料夾／zip 輸入與依群匯出；資料夾名稱作為事後比對的標籤 |
| 165 反詐騙諮詢專線－詐騙闢謠專區 (內政部警政署 2016) | 內政部警政署的繁體中文闢謠公告 89 篇，無類別標籤；依政府資料開放授權條款第 1 版使用 | 測試中文、小樣本且無標籤的資料 |

後兩者只用於開發測試，不包含在安裝用的 skill 中；BBC 文章的著作權屬於 BBC。

# 參考資料

<div id="refs" class="references csl-bib-body hanging-indent" data-entry-spacing="0">

<div id="ref-aggarwal2018" class="csl-entry">

Aggarwal, Charu C. 2018. *Machine Learning for Text*. Springer.

</div>

<div id="ref-bouma2009" class="csl-entry">

Bouma, Gerlof. 2009. “Normalized (Pointwise) Mutual Information in Collocation Extraction”. In *Proceedings of the Biennial GSCL Conference*, 31～40.

</div>

<div id="ref-deerwester1990" class="csl-entry">

Deerwester, Scott, Susan T. Dumais, George W. Furnas, Thomas K. Landauer, and Richard Harshman. 1990. “Indexing by Latent Semantic Analysis”. *Journal of the American Society for Information Science* 41 (6): 391～407.

</div>

<div id="ref-dhillon2001" class="csl-entry">

Dhillon, Inderjit S., and Dharmendra S. Modha. 2001. “Concept Decompositions for Large Sparse Text Data Using Clustering”. *Machine Learning* 42: 143～75.

</div>

<div id="ref-greene2006" class="csl-entry">

Greene, Derek, and Pádraig Cunningham. 2006. “Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering”. In *Proceedings of the 23rd International Conference on Machine Learning*, 377～84. <https://doi.org/10.1145/1143844.1143892>.

</div>

<div id="ref-grootendorst2022" class="csl-entry">

Grootendorst, Maarten. 2022. “BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure”. arXiv:2203.05794.

</div>

<div id="ref-hastie2009" class="csl-entry">

Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009. *The Elements of Statistical Learning*. 2nd ed. Springer.

</div>

<div id="ref-hubert1985" class="csl-entry">

Hubert, Lawrence, and Phipps Arabie. 1985. “Comparing Partitions”. *Journal of Classification* 2: 193～218.

</div>

<div id="ref-kaufman1990" class="csl-entry">

Kaufman, Leonard, and Peter J. Rousseeuw. 1990. *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley.

</div>

<div id="ref-landauer1997" class="csl-entry">

Landauer, Thomas K., and Susan T. Dumais. 1997. “A Solution to Plato’s Problem: The Latent Semantic Analysis Theory of Acquisition, Induction, and Representation of Knowledge”. *Psychological Review* 104 (2): 211～40.

</div>

<div id="ref-lang1995" class="csl-entry">

Lang, Ken. 1995. “NewsWeeder: Learning to Filter Netnews”. In *Proceedings of the Twelfth International Conference on Machine Learning*, 331～39.

</div>

<div id="ref-manning2008" class="csl-entry">

Manning, Christopher D., Prabhakar Raghavan, and Hinrich Schütze. 2008. *Introduction to Information Retrieval*. Cambridge University Press.

</div>

<div id="ref-newman2010" class="csl-entry">

Newman, David, Jey Han Lau, Karl Grieser, and Timothy Baldwin. 2010. “Automatic Evaluation of Topic Coherence”. In *Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics*, 100～108.

</div>

<div id="ref-pedregosa2011" class="csl-entry">

<span class="nocase">Pedregosa, Fabian et al.</span> 2011. “Scikit-learn: Machine Learning in Python”. *Journal of Machine Learning Research* 12: 2825～30.

</div>

<div id="ref-rousseeuw1987" class="csl-entry">

Rousseeuw, Peter J. 1987. “Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis”. *Journal of Computational and Applied Mathematics* 20: 53～65.

</div>

<div id="ref-salton1988" class="csl-entry">

Salton, Gerard, and Christopher Buckley. 1988. “Term-Weighting Approaches in Automatic Text Retrieval”. *Information Processing & Management* 24 (5): 513～23.

</div>

<div id="ref-schubert2021" class="csl-entry">

Schubert, Erich, and Peter J. Rousseeuw. 2021. “Fast and Eager k-Medoids Clustering: O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms”. *Information Systems* 101: 101804.

</div>

<div id="ref-sklearn_textclust" class="csl-entry">

<span class="nocase">scikit-learn developers</span>. n.d. “Clustering Text Documents Using k-Means”. scikit-learn example gallery. <https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html>.

</div>

<div id="ref-steinbach2000" class="csl-entry">

Steinbach, Michael, George Karypis, and Vipin Kumar. 2000. “A Comparison of Document Clustering Techniques”. In *KDD Workshop on Text Mining*.

</div>

<div id="ref-strehl2002" class="csl-entry">

Strehl, Alexander, and Joydeep Ghosh. 2002. “Cluster Ensembles: A Knowledge Reuse Framework for Combining Multiple Partitions”. *Journal of Machine Learning Research* 3: 583～617.

</div>

<div id="ref-jieba" class="csl-entry">

Sun, Junyi. n.d. “jieba: Chinese Text Segmentation”. GitHub repository. <https://github.com/fxsjy/jieba>.

</div>

<div id="ref-ward1963" class="csl-entry">

Ward, Jr., Joe H. 1963. “Hierarchical Grouping to Optimize an Objective Function”. *Journal of the American Statistical Association* 58 (301): 236～44. <https://doi.org/10.1080/01621459.1963.10500845>.

</div>

<div id="ref-zhang2022" class="csl-entry">

Zhang, Zihan, Meng Fang, Ling Chen, and Mohammad-Reza Namazi-Rad. 2022. “Is Neural Topic Modelling Better than Clustering? An Empirical Study on Clustering with Contextual Embeddings for Topics”. In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 3886～93. <https://doi.org/10.18653/v1/2022.naacl-main.285>.

</div>

<div id="ref-npa165" class="csl-entry">

內政部警政署. 2016. “165反詐騙諮詢專線－詐騙闢謠專區”. 政府資料開放平臺. <https://data.gov.tw/dataset/38262>.

</div>

</div>
