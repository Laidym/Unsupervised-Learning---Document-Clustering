"""步驟 4 的 HTML 報告段落：單一檔案、不依賴網路（圖片以 base64 內嵌、樣式內嵌）。

report.qmd 以 SECTIONS 的順序，逐段呼叫這裡的函式並以原始 HTML 輸出；同一份 qmd 由 Quarto 或
skill 內附的執行器（qmd_render.py）render。CSS 用於兩者；深色模式只在執行器的頁面（body.tc-page）啟用，
避免在 Quarto 的淺色版面中出現深色元件。PAGE_CSS 只用於執行器的頁面外框。
"""
import base64
import html
from pathlib import Path

from wordcloud import N_COLORS, TOP_WORDS, render_svg as word_cloud_svg

CSS = """
:root{--bg:#fcfcfb;--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;
--line:#e4e3df;--accent:#2a78d6;--accent-soft:#cde2fb;--warn-bg:#fdf3e1;--warn-ink:#6b4508;
--shade:42,120,214}
@media (prefers-color-scheme:dark){body.tc-page{--bg:#1a1a19;--surface:#232322;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#9a998f;--line:#3a3a37;--accent:#3987e5;--accent-soft:#1c3a5e;--warn-bg:#3a2e17;--warn-ink:#f3d9a4;
--shade:57,135,229}}
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
.wc-box{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px}
svg.wcloud{display:block;width:100%;height:auto;font-family:inherit}
svg.wcloud text{text-anchor:middle;dominant-baseline:central}
svg.wcloud text.plain{fill:var(--ink)}
svg.wcloud text.shared{fill:var(--muted)}
svg.wcloud .lbl rect{fill:var(--surface);stroke-width:2}
svg.wcloud .lbl rect.plain{stroke:var(--ink2)}
svg.wcloud .lbl text{fill:var(--ink)}
svg.wcloud text:hover{text-decoration:underline}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;margin:8px 2px 0;font-size:13px;color:var(--ink2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
:root{--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948}
@media (prefers-color-scheme:dark){body.tc-page{--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;
--s6:#0ca30c;--s7:#9085e9;--s8:#e66767}}
svg.wcloud text.s1{fill:var(--s1)}svg.wcloud .lbl rect.s1{stroke:var(--s1)}.legend i.s1{background:var(--s1)}
svg.wcloud text.s2{fill:var(--s2)}svg.wcloud .lbl rect.s2{stroke:var(--s2)}.legend i.s2{background:var(--s2)}
svg.wcloud text.s3{fill:var(--s3)}svg.wcloud .lbl rect.s3{stroke:var(--s3)}.legend i.s3{background:var(--s3)}
svg.wcloud text.s4{fill:var(--s4)}svg.wcloud .lbl rect.s4{stroke:var(--s4)}.legend i.s4{background:var(--s4)}
svg.wcloud text.s5{fill:var(--s5)}svg.wcloud .lbl rect.s5{stroke:var(--s5)}.legend i.s5{background:var(--s5)}
svg.wcloud text.s6{fill:var(--s6)}svg.wcloud .lbl rect.s6{stroke:var(--s6)}.legend i.s6{background:var(--s6)}
svg.wcloud text.s7{fill:var(--s7)}svg.wcloud .lbl rect.s7{stroke:var(--s7)}.legend i.s7{background:var(--s7)}
svg.wcloud text.s8{fill:var(--s8)}svg.wcloud .lbl rect.s8{stroke:var(--s8)}.legend i.s8{background:var(--s8)}
"""

# 只用於 skill 內附的 qmd 執行器產生的頁面（Quarto render 時由 Quarto 的版面處理）
PAGE_CSS = """
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI",
"Noto Sans TC","Microsoft JhengHei","PingFang TC",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:19px;margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:16px;margin:24px 0 8px}
.sub{color:var(--ink2);margin:0 0 24px}
code{font:13px ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--surface);border:1px solid var(--line);
border-radius:4px;padding:0 4px}
pre{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 12px;overflow-x:auto;margin:6px 0}
pre code{border:0;padding:0;background:none;font-size:12.5px;line-height:1.5}
details.code-fold{margin:6px 0 12px}
details.code-fold>summary{font-size:13px}
a{color:var(--accent)}
nav.toc{border:1px solid var(--line);border-radius:10px;padding:8px 14px;margin:0 0 16px;background:var(--surface);
font-size:14px}
nav.toc ul{display:flex;flex-wrap:wrap;gap:2px 16px;margin:4px 0 0;padding:0;list-style:none}
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


def subtitle(ctx):
    return f'{Path(ctx["cfg1"]["input"]).name}｜{ctx["n"]:,} 篇文件｜{ctx["algorithm_name"]}｜{ctx["k"]} 群'


def summary(ctx):
    """摘要卡片（放在標題下方）。"""
    cards = [("文件數", f"{ctx['n']:,}"), ("群數", str(ctx["k"])),
             ("silhouette", _fmt(ctx["silhouette"])), ("關鍵詞一致性 NPMI", _fmt(ctx["npmi"]))]
    if ctx["external"]:
        cards += [("ARI（事後比對）", _fmt(ctx["external"]["ARI"])),
                  ("NMI（事後比對）", _fmt(ctx["external"]["NMI"]))]
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="v">{_e(v)}</div><div class="l">{_e(l)}</div></div>'
        for l, v in cards) + "</div>"


def clusters(ctx):
    n = ctx["n"]
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
    return (_table(["群", "群名稱", "文件數", "NPMI", "關鍵詞（c-TF-IDF 前 10 名）"], rows, num_cols=(3,))
            + '<p class="muted">名稱來源：「採用建議」為採用 skill 依關鍵詞提出的建議名稱；「自訂」為使用者自行填寫。'
            "NPMI 為該群前 10 個關鍵詞兩兩共同出現程度的平均。</p>")


def merge(ctx):
    """階層式分群的合併關係；非階層式分群回傳空字串。"""
    if not ctx.get("merge"):
        return ""
    P = ["<p>由近到遠的合併順序（Ward 連結）；越早合併的群內容越接近。</p>",
         "<ol>" + "".join(f"<li>{_e(m)}</li>" for m in ctx["merge"]) + "</ol>"]
    src = _img(ctx["images"].get("dendrogram", ""))
    if src:
        P.append(f'<figure><img src="{src}" alt="各群合併關係樹狀圖"><figcaption>各群的合併關係</figcaption></figure>')
    return "\n".join(P)


def k_selection(ctx):
    s2, k = ctx["cfg2"], ctx["k"]
    P = [f'<p>候選群數 k = {s2["k_min"]} 到 {s2["k_max"]}，使用者選定 <strong>k = {k}</strong>。'
         "silhouette 衡量文件是否明確屬於自己的群；NPMI 衡量各群關鍵詞是否常在同一文件出現。"
         "兩者都只宜比較相對高低。</p>"]
    src = _img(ctx["images"].get("k_selection", ""))
    if src:
        P.append(f'<figure><img src="{src}" alt="各候選群數的 silhouette 與 NPMI">'
                 "<figcaption>各候選群數的 silhouette（上）與 NPMI（下），淡藍色帶為選定的群數</figcaption></figure>")
    src = _img(ctx["images"].get("dendrogram_overview", ""))
    if src:
        P.append(f'<figure><img src="{src}" alt="候選群數在樹狀圖上的切割位置">'
                 f"<figcaption>切成候選上限 {s2['k_max']} 群時的合併關係；虛線為各候選群數的切割位置，"
                 f"橘色實線為選定的 k = {k}（切割線右側的分叉數即為群數）</figcaption></figure>")
    rows = [[(f"<strong>{int(r.k)}</strong>" if int(r.k) == k else str(int(r.k))),
             _fmt(r.silhouette), _fmt(r.npmi), f"{int(r.min_size):,}", f"{int(r.max_size):,}"]
            for r in ctx["metrics_by_k"].itertuples(index=False)]
    P.append("<details><summary>各候選群數的指標</summary>"
             + _table(["k", "silhouette", "NPMI", "最小群文件數", "最大群文件數"], rows, num_cols=(1, 2, 3, 4))
             + "</details>")
    return "\n".join(P)


def internal(ctx):
    samp = "文件數超過 5,000 篇，以固定 seed 隨機抽樣 5,000 篇計算。" if ctx["sil_sampled"] else ""
    return ('<ul class="plain">'
            f'<li><strong>silhouette 係數 {_fmt(ctx["silhouette"])}</strong>：範圍 −1 至 1。'
            f"每篇文件比較與同群、與最近他群的平均距離；文字資料數值通常偏低，宜比較不同群數的相對高低。{samp}</li>"
            f'<li><strong>關鍵詞一致性 NPMI {_fmt(ctx["npmi"])}</strong>：範圍 −1 至 1。'
            "越高表示各群關鍵詞越常在同一篇文件共同出現，主題越連貫、越容易命名。</li></ul>")


def external(ctx):
    if ctx["external"] is None:
        return "<p>未指定標籤欄位，略過事後比對。</p>"
    e, k = ctx["external"], ctx["k"]
    P = ['<ul class="plain">'
         f'<li><strong>ARI {_fmt(e["ARI"])}</strong>：1 為完全一致，0 相當於隨機分群。</li>'
         f'<li><strong>NMI {_fmt(e["NMI"])}</strong>：0 為毫無關聯，1 為完全對應。</li>'
         f'<li><strong>純度 {_fmt(e["純度"])}</strong>：群數越多越高，須搭配 ARI、NMI 解讀。</li>'
         f'<li>群數 {k}，標籤類別數 {ctx["n_classes"]}'
         + ("；群數與類別數不同時，ARI、NMI 本來就不會接近 1。" if k != ctx["n_classes"] else "。")
         + "</li></ul>"]
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
    return "\n".join(P)


def limits(ctx):
    P = ['<ul class="plain">' + "".join(f"<li>{_e(s)}</li>" for s in ctx["fixed_limits"]) + "</ul>"]
    if ctx["data_notes"]:
        P.append('<div class="note"><strong>依本次資料</strong><ul>'
                 + "".join(f"<li>{_e(s)}</li>" for s in ctx["data_notes"]) + "</ul></div>")
    return "\n".join(P)


def vocabulary(ctx):
    summ = ctx.get("summary1")
    if not summ:
        return ""
    unit = summ.get("原始長度單位", "詞")
    rows = [[_e(t["詞"]), f'{t["總次數"]:,}', f'{t["出現篇數"]:,}', f'{t["出現比例"]:.1%}']
            for t in summ["最高頻詞"]]
    return (f'<p>詞彙量 {summ["詞彙量"]:,}；平均文件長度：原始 {summ.get("平均原始長度", "—")} {unit}，'
            f'去除停用詞與篩選後 {summ["平均有效詞數"]} 詞（中位數 {summ["有效詞數中位數"]:g}）。</p>'
            "<details><summary>最高頻詞</summary>"
            + _table(["詞", "總次數", "出現篇數", "出現比例"], rows, num_cols=(1, 2, 3)) + "</details>")


def word_cloud(ctx):
    clouds = ctx.get("word_clouds")
    if not clouds:
        return ""
    colored = len(clouds) <= N_COLORS
    legend = ""
    if colored:
        legend = ('<div class="legend">' + "".join(
            f'<span><i class="s{c["cluster"] + 1}"></i>群 {c["cluster"]} {_e(c["name"])}</span>' for c in clouds)
            + '<span><i style="background:var(--muted)"></i>多群共有的詞</span></div>')
    return (f"<p>所有群的詞畫在同一張圖上：每群取 c-TF-IDF 權重前 {TOP_WORDS} 個詞，字越大表示越能代表該群，"
            "粗體為前 10 個群關鍵詞（與「各群」表相同，用於命名與計算 NPMI）。"
            "各群的位置依群中心之間的 cosine 距離排列（古典 MDS），越近表示兩群的用詞越相似；"
            "同時出現在多群的詞只畫一次，以灰色標示，放在相關各群之間。滑鼠移到詞上可看所屬群與權重。</p>"
            f'<figure class="wc-box">{word_cloud_svg(clouds, ctx["cloud_anchors"])}{legend}</figure>'
            '<p class="muted">位置只是把高維距離壓到平面上的近似，群與群之間的遠近僅供參考；'
            + ("" if colored else f"群數超過 {N_COLORS}，不以顏色區分群，請依群名稱標籤的位置與滑鼠提示辨識；")
            + "多群共有的詞依它在各群的相對權重決定位置。</p>")


def settings(ctx):
    return ('<dl class="cfg">' + "".join(f"<dt>{_e(a)}</dt><dd>{_e(b)}</dd>" for a, b in ctx["settings"]) + "</dl>"
            '<p class="muted">所有隨機步驟皆固定隨機種子，以相同資料與設定重跑可得到相同結果。</p>')


# 報告段落：(標題, 函式名稱)。report.qmd 依此順序產生。
SECTIONS = [("摘要", "summary"), ("各群", "clusters"), ("各群的上下層關係", "merge"),
            ("群數的選擇", "k_selection"), ("內部指標", "internal"), ("事後比對", "external"),
            ("限制", "limits"), ("詞彙摘要（步驟 1）", "vocabulary"), ("文字雲", "word_cloud"),
            ("設定紀錄", "settings")]


def page(title, sub, body, toc=()):
    """單一 HTML 檔的頁面外框（skill 內附的 qmd 執行器使用）。"""
    nav = ""
    if toc:
        nav = ('<nav class="toc"><strong>目錄</strong><ul>'
               + "".join(f'<li><a href="#{_e(a)}">{_e(t)}</a></li>' for a, t in toc) + "</ul></nav>")
    return ("<!doctype html><html lang=\"zh-Hant\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_e(title)}</title><style>{CSS}{PAGE_CSS}</style></head><body class=\"tc-page\"><main>"
            f"<h1>{_e(title)}</h1>" + (f'<p class="sub">{_e(sub)}</p>' if sub else "")
            + nav + body + "</main></body></html>\n")


def render(ctx):
    """不經 qmd，直接由 ctx 組成整份報告（qmd 執行失敗時的備援）。"""
    P = []
    for title, name in SECTIONS:
        part = globals()[name](ctx)
        if part:
            P.append((f"<h2>{_e(title)}</h2>" if title else "") + part)
    return page("文件分群報告", subtitle(ctx), "\n".join(P))
