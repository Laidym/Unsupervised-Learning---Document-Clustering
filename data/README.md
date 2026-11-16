# data/

| 檔案 | 內容 | 打包進 skill |
|---|---|---|
| `20ng_subset.csv` | 20 Newsgroups 4 類各 200 篇，已移除 headers、footers、quotes（`tools/prepare_20ng.py` 產生）；內建範例 | 是 |
| `bbc-fulltext.zip` | BBC News 2,225 篇（2004–2005），5 類各一個資料夾，每個 `.txt` 一篇；含說明檔 `README.TXT`。來源：Greene & Cunningham (2006)，<http://mlg.ucd.ie/datasets/bbc.html>。文章著作權屬 BBC | 否，開發測試用（.txt 資料夾／zip 輸入） |
| `NPA_165.csv` | 內政部警政署 165 反詐騙諮詢專線公告 89 篇（欄位：編號、標題、發佈時間、發佈內容），無類別標籤 | 否，開發測試用（中文、小樣本） |

注意：`NPA_165.csv` 只有 89 篇，預設候選群數上限為 floor(89/30) = 2，需自訂 k 範圍。
