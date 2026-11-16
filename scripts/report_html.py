"""步驟 4 的 HTML 報告：單一檔案、不依賴網路（圖片以 base64 內嵌、樣式內嵌）。"""
import base64
import html
from pathlib import Path

CSS = """
:root{--bg:#fcfcfb;--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;
--line:#e4e3df;--accent:#2a78d6;--accent-soft:#cde2fb;--warn-bg:#fdf3e1;--warn-ink:#6b4508;
--shade:42,120,214}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--surface:#232322;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#9a998f;--line:#3a3a37;--accent:#3987e5;--accent-soft:#1c3a5e;--warn-bg:#3a2e17;--warn-ink:#f3d9a4;
--shade:57,135,229}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI",
"Noto Sans TC","Microsoft JhengHei","PingFang TC",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:19px;margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--ink2);margin:0 0 24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.card .l{color:var(--ink2);font-size:13px}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--surface)}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{display:inline-block;height:8px;border-radius:4px;background:var(--accent);vertical-align:middle;margin-right:6px}
.kw{color:var(--ink2)}
.tag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:9px;border:1px solid var(--line);color:var(--ink2);white-space:nowrap}
.tag.sugg{border-color:var(--accent);color:var(--accent)}
figure{margin:12px 0;background:#fcfcfb;border:1px solid var(--line);border-radius:10px;padding:8px}
figure img{max-width:100%;height:auto;display:block;margin:0 auto}
figcaption{color:var(--muted);font-size:13px;padding:4px 6px 0}
.note{background:var(--warn-bg);color:var(--warn-ink);border-radius:10px;padding:10px 14px;margin:12px 0}
.note ul{margin:4px 0 0;padding-left:20px}
ul.plain{padding-left:20px}
ol li,ul li{margin:2px 0}
details{margin-top:8px}
summary{cursor:pointer;color:var(--ink2)}
dl.cfg{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:8px 0}
dl.cfg dt{color:var(--ink2)}
dl.cfg dd{margin:0}
.muted{color:var(--muted);font-size:13px}
"""


def _e(x):
    return html.escape(str(x))


def _img(path):
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _table(headers, rows, num_cols=()):
    h = "".join(f"<th>{_e(c)}</th>" for c in headers)
    body = []
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            cls = ' class="num"' if i in num_cols else ""
            cells.append(f"<td{cls}>{v}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _fmt(v, nd=3):
    return "—" if v is None or v != v else f"{v:.{nd}f}"


def render(ctx):
    s1, s2, k, n = ctx["cfg1"], ctx["cfg2"], ctx["k"], ctx["n"]
    P = []
    title = "文件分群報告"
    P.append(f"<h1>{title}</h1>")
    P.append(f'<p class="sub">{_e(Path(s1["input"]).name)}｜{n:,} 篇文件｜'
             f'{_e(ctx["algorithm_name"])}｜{k} 群</p>')

    # 摘要
    cards = [("文件數", f"{n:,}"), ("群數", str(k)),
             ("silhouette", _fmt(ctx["silhouette"])), ("關鍵詞一致性 NPMI", _fmt(ctx["npmi"]))]
    if ctx["external"]:
        cards += [("ARI（事後比對）", _fmt(ctx["external"]["ARI"])),
                  ("NMI（事後比對）", _fmt(ctx["external"]["NMI"]))]
    P.append('<div class="cards">' + "".join(
        f'<div class="card"><div class="v">{_e(v)}</div><div class="l">{_e(l)}</div></div>'
        for l, v in cards) + "</div>")

    # 各群
    P.append("<h2>各群</h2>")
    ct = ctx["cluster_table"]
    max_size = max(ct["文件數"]) or 1
    rows = []
    for r in ct.itertuples(index=False):
        tag = ('<span class="tag sugg">採用建議</span>' if r.名稱來源 == "採用建議"
               else '<span class="tag">自訂</span>')
        bar = f'<span class="bar" style="width:{max(4, round(60 * r.文件數 / max_size))}px"></span>'
        rows.append([str(r.群), f"<strong>{_e(r.群名稱)}</strong> {tag}",
                     f"{bar}{r.文件數:,}（{r.文件數 / n:.1%}）", _fmt(r.NPMI),
                     f'<span class="kw">{_e(r.關鍵詞)}</span>'])
    P.append(_table(["群", "群名稱", "文件數", "NPMI", "關鍵詞（c-TF-IDF 前 10 名）"], rows, num_cols=(3,)))
    P.append('<p class="muted">名稱來源：「採用建議」為採用 skill 依關鍵詞提出的建議名稱；「自訂」為使用者自行填寫。'
             "NPMI 為該群前 10 個關鍵詞兩兩共同出現程度的平均。</p>")

    if ctx.get("merge"):
        P.append("<h2>各群的上下層關係</h2>")
        P.append("<p>由近到遠的合併順序（Ward 連結）；越早合併的群內容越接近。</p>")
        P.append('<ol>' + "".join(f"<li>{_e(m)}</li>" for m in ctx["merge"]) + "</ol>")
        src = _img(ctx["images"].get("dendrogram", ""))
        if src:
            P.append(f'<figure><img src="{src}" alt="各群合併關係樹狀圖"><figcaption>各群的合併關係</figcaption></figure>')

    # 群數選擇
    P.append("<h2>群數的選擇</h2>")
    P.append(f'<p>候選群數 k = {s2["k_min"]} 到 {s2["k_max"]}，使用者選定 <strong>k = {k}</strong>。'
             "silhouette 衡量文件是否明確屬於自己的群；NPMI 衡量各群關鍵詞是否常在同一文件出現。"
             "兩者都只宜比較相對高低。</p>")
    src = _img(ctx["images"].get("k_selection", ""))
    if src:
        P.append(f'<figure><img src="{src}" alt="各候選群數的 silhouette 與 NPMI">'
                 "<figcaption>各候選群數的 silhouette（上）與 NPMI（下），淡藍色帶為選定的群數</figcaption></figure>")
    src = _img(ctx["images"].get("dendrogram_overview", ""))
    if src:
        P.append(f'<figure><img src="{src}" alt="候選群數在樹狀圖上的切割位置">'
                 f"<figcaption>切成候選上限 {s2['k_max']} 群時的合併關係；虛線為各候選群數的切割位置，"
                 f"橘色實線為選定的 k = {k}（切割線右側的分叉數即為群數）</figcaption></figure>")
    mk = ctx["metrics_by_k"]
    rows = [[(f"<strong>{int(r.k)}</strong>" if int(r.k) == k else str(int(r.k))),
             _fmt(r.silhouette), _fmt(r.npmi), f"{int(r.min_size):,}", f"{int(r.max_size):,}"]
            for r in mk.itertuples(index=False)]
    P.append("<details><summary>各候選群數的指標</summary>"
             + _table(["k", "silhouette", "NPMI", "最小群文件數", "最大群文件數"], rows, num_cols=(1, 2, 3, 4))
             + "</details>")

    # 內部指標
    P.append("<h2>內部指標</h2>")
    samp = "文件數超過 5,000 篇，以固定 seed 隨機抽樣 5,000 篇計算。" if ctx["sil_sampled"] else ""
    P.append('<ul class="plain">'
             f'<li><strong>silhouette 係數 {_fmt(ctx["silhouette"])}</strong>：範圍 −1 至 1。'
             f"每篇文件比較與同群、與最近他群的平均距離；文字資料數值通常偏低，宜比較不同群數的相對高低。{samp}</li>"
             f'<li><strong>關鍵詞一致性 NPMI {_fmt(ctx["npmi"])}</strong>：範圍 −1 至 1。'
             "越高表示各群關鍵詞越常在同一篇文件共同出現，主題越連貫、越容易命名。</li></ul>")

    # 事後比對
    P.append("<h2>事後比對</h2>")
    if ctx["external"] is None:
        P.append("<p>未指定標籤欄位，略過事後比對。</p>")
    else:
        e = ctx["external"]
        P.append('<ul class="plain">'
                 f'<li><strong>ARI {_fmt(e["ARI"])}</strong>：1 為完全一致，0 相當於隨機分群。</li>'
                 f'<li><strong>NMI {_fmt(e["NMI"])}</strong>：0 為毫無關聯，1 為完全對應。</li>'
                 f'<li><strong>純度 {_fmt(e["純度"])}</strong>：群數越多越高，須搭配 ARI、NMI 解讀。</li>'
                 f'<li>群數 {k}，標籤類別數 {ctx["n_classes"]}'
                 + ("；群數與類別數不同時，ARI、NMI 本來就不會接近 1。" if k != ctx["n_classes"] else "。")
                 + "</li></ul>")
        cross = ctx["crosstab"]
        cols = list(cross.columns)
        rows = []
        for idx, r in cross.iterrows():
            tot = r.sum() or 1
            cells = [f"<strong>{_e(idx)}</strong>"]
            for c in cols:
                a = r[c] / tot
                cells.append(f'<span style="display:block;margin:-7px -10px;padding:7px 10px;'
                             f'background:rgba(var(--shade),{0.08 + 0.6 * a:.2f})">{int(r[c]):,}</span>'
                             if r[c] else "0")
            rows.append(cells)
        P.append("<p>群與標籤的交叉表（底色深淺為該群內各標籤的比例）：</p>")
        P.append(_table(["群"] + [str(c) for c in cols], rows, num_cols=tuple(range(1, len(cols) + 1))))
        P.append('<p class="muted">既有分類只是一種分法，不吻合不代表分群錯誤。標籤只用於本節，'
                 "未用於前處理、分群與選定群數。</p>")

    # 限制
    P.append("<h2>限制</h2>")
    P.append('<ul class="plain">' + "".join(f"<li>{_e(s)}</li>" for s in ctx["fixed_limits"]) + "</ul>")
    if ctx["data_notes"]:
        P.append('<div class="note"><strong>依本次資料</strong><ul>'
                 + "".join(f"<li>{_e(s)}</li>" for s in ctx["data_notes"]) + "</ul></div>")

    # 詞彙摘要
    summ = ctx.get("summary1")
    if summ:
        P.append("<h2>詞彙摘要（步驟 1）</h2>")
        unit = summ.get("原始長度單位", "詞")
        P.append(f'<p>詞彙量 {summ["詞彙量"]:,}；平均文件長度：原始 {summ.get("平均原始長度", "—")} {unit}，'
                 f'去除停用詞與篩選後 {summ["平均有效詞數"]} 詞（中位數 {summ["有效詞數中位數"]:g}）。</p>')
        rows = [[_e(t["詞"]), f'{t["總次數"]:,}', f'{t["出現篇數"]:,}', f'{t["出現比例"]:.1%}']
                for t in summ["最高頻詞"]]
        P.append("<details><summary>最高頻詞</summary>"
                 + _table(["詞", "總次數", "出現篇數", "出現比例"], rows, num_cols=(1, 2, 3)) + "</details>")

    # 設定紀錄
    P.append("<h2>設定紀錄</h2>")
    P.append('<dl class="cfg">' + "".join(f"<dt>{_e(a)}</dt><dd>{_e(b)}</dd>" for a, b in ctx["settings"]) + "</dl>")
    P.append('<p class="muted">所有隨機步驟皆固定隨機種子，以相同資料與設定重跑可得到相同結果。</p>')

    return ("<!doctype html><html lang=\"zh-Hant\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{title}</title><style>{CSS}</style></head><body><main>"
            + "\n".join(P) + "</main></body></html>\n")
