"""報告中的整體文字雲：所有群的詞放在同一張圖上，以 SVG 文字排版（不依賴 wordcloud 套件，中文不需另備字型）。

- 群的位置：各群中心（群內文件向量平均後正規化）兩兩的 cosine 距離，以古典 MDS 降到二維；
  越相似的群放得越近。
- 詞：每群 c-TF-IDF 前 TOP_WORDS 個詞，字級與「權重 / 該群最高權重」成正比；前 10 個（群關鍵詞）為粗體。
- 多群共有的詞只畫一次，以灰色標示，位置在相關各群之間（依各群相對權重加權平均）。
- 排版：先放群名稱標籤，再依字級由大到小，從目標位置沿橢圓螺旋向外找第一個不重疊的位置。
  完全沒有隨機性，相同輸入得到相同結果。
"""
import html

import numpy as np

TOP_WORDS = 30
N_KEYWORDS = 10
MIN_PX, MAX_PX = 11, 34
LABEL_PX = 14
N_COLORS = 8              # 報告 CSS 定義的類別色數；群數超過時不以顏色區分群
PAD = 1.5


def cluster_anchors(vectors, clusters, k):
    """各群中心的二維位置（古典 MDS，cosine 距離），回傳 (k, 2)，尺度約為單位圓。"""
    C = np.zeros((k, vectors.shape[1]))
    for c in range(k):
        m = vectors[clusters == c].mean(axis=0)
        C[c] = m / (np.linalg.norm(m) or 1.0)
    if k == 1:
        return np.zeros((1, 2))
    D = np.clip(1.0 - C @ C.T, 0, None)
    J = np.eye(k) - 1.0 / k
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(-vals)[:2]
    xy = np.zeros((k, 2))
    for j, i in enumerate(order):
        if vals[i] > 1e-12:
            v = vecs[:, i] * np.sqrt(vals[i])
            if v[np.argmax(np.abs(v))] < 0:          # 固定正負號，結果可重現
                v = -v
            xy[:, j] = v
    r = np.abs(xy).max()
    return xy / r if r > 0 else xy


def _text_width(s, px, bold=False):
    w = 0.0
    for ch in s:
        if ord(ch) > 0x2e80:
            w += 1.0
        elif ch == " ":
            w += 0.3
        elif ch.isupper():
            w += 0.66
        else:
            w += 0.56
    return w * px * (1.07 if bold else 1.0)


def _place(boxes, cx, cy, w, h, step=0.22, a=1.1, max_steps=20000):
    """從 (cx, cy) 沿橢圓螺旋找第一個與已放置方框不重疊的位置，回傳方框 (x0, y0, x1, y1)。"""
    arr = np.array(boxes) if boxes else np.zeros((0, 4))
    for t in range(max_steps):
        th = t * step
        r = a * th
        x, y = cx + 1.5 * r * np.cos(th), cy + r * np.sin(th)
        b = (x - w / 2 - PAD, y - h / 2 - PAD, x + w / 2 + PAD, y + h / 2 + PAD)
        if not len(arr) or not np.any((arr[:, 0] < b[2]) & (arr[:, 2] > b[0])
                                      & (arr[:, 1] < b[3]) & (arr[:, 3] > b[1])):
            return b
    return b


def build_items(clouds):
    """clouds：每群 {"cluster", "name", "terms", "weights"}。合併多群共有的詞，回傳待排版的詞。"""
    words = {}
    for c in clouds:
        wmax = max(c["weights"], default=0) or 1.0
        for rank, (t, w) in enumerate(zip(c["terms"], c["weights"])):
            d = words.setdefault(t, {"term": t, "clusters": [], "rel": [], "weights": [], "keyword": False})
            d["clusters"].append(c["cluster"])
            d["rel"].append(max(w, 0) / wmax)
            d["weights"].append(w)
            d["keyword"] |= rank < N_KEYWORDS
    items = list(words.values())
    for d in items:
        d["size"] = MIN_PX + (MAX_PX - MIN_PX) * max(d["rel"])
    # 字級由大到小；同字級依詞排序，確保可重現
    items.sort(key=lambda d: (-d["size"], d["term"]))
    return items


def layout(clouds, anchors):
    """回傳 (labels, words, viewBox)。座標單位為 px。"""
    k = len(clouds)
    items = build_items(clouds)
    area = sum(_text_width(d["term"], d["size"], d["keyword"]) * d["size"] * 1.1 for d in items)
    radius = 0.55 * np.sqrt(area) if k > 1 else 0.0
    pos = np.asarray(anchors, dtype=float) * radius * np.array([1.4, 1.0])

    boxes, labels, words = [], [], []
    for c in clouds:
        text = f"群 {c['cluster']}　{c['name']}"
        w, h = _text_width(text, LABEL_PX, True) + 16, LABEL_PX + 10
        b = _place(boxes, *pos[c["cluster"]], w, h)
        boxes.append(b)
        labels.append({"cluster": c["cluster"], "text": text, "box": b})
    for d in items:
        rel = np.array(d["rel"])
        target = (pos[d["clusters"]] * rel[:, None]).sum(axis=0) / rel.sum() if rel.sum() > 0 \
            else pos[d["clusters"]].mean(axis=0)
        w, h = _text_width(d["term"], d["size"], d["keyword"]), d["size"] * 1.05
        b = _place(boxes, *target, w, h)
        boxes.append(b)
        words.append({**d, "box": b})
    arr = np.array(boxes)
    m = 8
    view = (arr[:, 0].min() - m, arr[:, 1].min() - m,
            arr[:, 2].max() - arr[:, 0].min() + 2 * m, arr[:, 3].max() - arr[:, 1].min() + 2 * m)
    return labels, words, view


def render_svg(clouds, anchors):
    k = len(clouds)
    colored = k <= N_COLORS
    names = {c["cluster"]: c["name"] for c in clouds}
    labels, words, (vx, vy, vw, vh) = layout(clouds, anchors)
    e = html.escape
    out = [f'<svg class="wcloud" viewBox="{vx:.0f} {vy:.0f} {vw:.0f} {vh:.0f}" '
           f'role="img" aria-label="所有群的文字雲">']
    for d in words:
        x0, y0, x1, y1 = d["box"]
        shared = len(d["clusters"]) > 1
        cls = "shared" if shared else (f"s{d['clusters'][0] + 1}" if colored else "plain")
        tip = "；".join(f"群 {c} {names[c]}：c-TF-IDF {w:.4f}" for c, w in zip(d["clusters"], d["weights"]))
        if shared:
            tip = f"{len(d['clusters'])} 群共有｜{tip}"
        weight = ' font-weight="700"' if d["keyword"] else ""
        out.append(f'<text class="{cls}" x="{(x0 + x1) / 2:.1f}" y="{(y0 + y1) / 2:.1f}" '
                   f'font-size="{d["size"]:.1f}"{weight}><title>{e(d["term"])}｜{e(tip)}</title>{e(d["term"])}</text>')
    for lb in labels:
        x0, y0, x1, y1 = lb["box"]
        cls = f"s{lb['cluster'] + 1}" if colored else "plain"
        out.append(f'<g class="lbl"><rect class="{cls}" x="{x0 + PAD:.1f}" y="{y0 + PAD:.1f}" '
                   f'width="{x1 - x0 - 2 * PAD:.1f}" height="{y1 - y0 - 2 * PAD:.1f}" rx="6"/>'
                   f'<text x="{(x0 + x1) / 2:.1f}" y="{(y0 + y1) / 2:.1f}" font-size="{LABEL_PX}" '
                   f'font-weight="700">{e(lb["text"])}</text></g>')
    out.append("</svg>")
    return "".join(out)
