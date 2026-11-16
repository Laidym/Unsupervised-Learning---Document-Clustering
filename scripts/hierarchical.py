"""階層式分群：Ward 連結（Ward 1963），作用於長度已正規化為 1 的向量。

單位向量之間的歐氏距離平方 = 2 − 2·cos，因此 Ward 合併準則仍以 cosine 相似度為基礎。
連結法為固定設定：開發時比較 average、complete、Ward 三種，average 在 20 Newsgroups
上會在小群數時把幾乎所有文件併成一群，complete 幾乎分不開，Ward 的群大小較均衡、
silhouette 也最高。

樹只建一次，再依群數 k 切割（scipy cut_tree 保證剛好 k 群）。
"""
import numpy as np
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage


def build_linkage(vectors):
    return linkage(np.asarray(vectors, dtype=np.float64), method="ward")


def cut(L, k):
    """依群數 k 切割，群編號依群內最小文件索引排序。"""
    raw = cut_tree(L, n_clusters=k).ravel()
    first = {}
    for i, c in enumerate(raw):
        first.setdefault(int(c), i)
    order = sorted(first, key=first.get)
    remap = {c: j for j, c in enumerate(order)}
    return np.array([remap[int(c)] for c in raw])


def _leaf_member(L, node, n):
    """找出節點底下任一篇文件的索引。"""
    while node >= n:
        node = int(L[node - n, 0])
    return node


def cluster_linkage(L, labels, k):
    """把完整的樹縮成「k 群之間」的樹，回傳以群編號 0..k−1 為葉的 linkage 矩陣。

    完整樹的最後 k − 1 次合併，正是切成 k 群之後各群彼此合併的過程。
    """
    n = L.shape[0] + 1
    rows = L[n - k:]                                  # 最後 k − 1 次合併
    new_id = {}
    for j in range(k - 1):
        new_id[n + (n - k) + j] = k + j               # 最後 k − 1 個合併節點的新編號
    Z = np.zeros((k - 1, 4))
    for j, (a, b, h, _) in enumerate(rows):
        ids = []
        for node in (int(a), int(b)):
            if node in new_id:
                ids.append(new_id[node])
            else:                                     # 這是切割後的一整群
                ids.append(int(labels[_leaf_member(L, node, n)]))
        # 第 4 欄依 scipy 規定為節點底下的葉（群）數
        leaves = sum(1 if i < k else Z[i - k, 3] for i in ids)
        Z[j] = [ids[0], ids[1], h, leaves]
    return Z


def cut_positions(L, ks):
    """各群數 k 在完整樹上的切割位置（合併距離），回傳 {k: x}。

    切成 k 群 = 保留最後 k − 1 次合併以外的所有合併；切割線落在
    「剩 k 群前的最後一次合併」與「由 k 群併成 k − 1 群」兩個高度的中點。
    """
    n = L.shape[0] + 1
    out = {}
    for k in ks:
        if 2 <= k < n:
            out[k] = (float(L[n - k - 1, 2]) + float(L[n - k, 2])) / 2
    return out


def plot_dendrogram(Z, leaf_labels, path, title, font=None, cuts=None, highlight_k=None):
    """畫出 k 群之間的樹狀圖（水平方向，群名在左側）。

    cuts：{k: 合併距離}，以虛線標出各候選群數的切割位置；
    highlight_k：以實線強調其中一個群數（報告標出選定的 k）。
    只標示切割位置，不依合併距離推薦群數。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if font:
        plt.rcParams["font.family"] = [font, "sans-serif"]
    ink, muted, grid, color = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    cut_color, hi_color = "#9a998f", "#eb6834"
    k = Z.shape[0] + 1
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * k + 1.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    dendrogram(Z, orientation="right", labels=leaf_labels, ax=ax,
               color_threshold=0, above_threshold_color=color,
               link_color_func=lambda _: color)
    for coll in ax.collections:
        coll.set_linewidth(2)
    # 群內部的合併已省略，橫軸從第一次群間合併之前開始，避免樹被擠在右側
    lo, hi = float(Z[:, 2].min()), float(Z[:, 2].max())
    left = lo - 0.25 * (hi - lo) - 0.05
    if cuts:
        left = min(left, min(cuts.values()) - 0.05 * (hi - lo))
    ax.set_xlim(max(0.0, left), hi * 1.03)

    if cuts:
        span = hi * 1.03 - max(0.0, left)
        last_x = None
        for kk, x in sorted(cuts.items(), key=lambda t: t[1]):
            is_hi = kk == highlight_k
            ax.axvline(x, color=hi_color if is_hi else cut_color, linewidth=1.6 if is_hi else 0.9,
                       linestyle="-" if is_hi else (0, (3, 3)), zorder=0)
            # 標籤太擠時略過（選定的 k 一定標示）
            if is_hi or last_x is None or (x - last_x) > 0.075 * span:
                ax.annotate(f"k={kk}", (x, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=8.5, color=hi_color if is_hi else muted,
                            fontweight="bold" if is_hi else "normal")
                last_x = x
    ax.set_title(title, loc="left", fontsize=11, color=ink, pad=18 if cuts else 6)
    ax.set_xlabel("合併距離（Ward）" if font else "Merge distance (Ward)", color=muted, fontsize=10)
    ax.tick_params(colors=muted, labelsize=9)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def merge_steps(Z, names):
    """以文字描述 k 群之間由近到遠的合併順序。"""
    k = Z.shape[0] + 1
    label = {i: names[i] for i in range(k)}
    lines = []
    for j, (a, b, h, _) in enumerate(Z):
        lines.append(f"合併 {j + 1}：{label[int(a)]} ＋ {label[int(b)]}（距離 {h:.2f}）")
        label[k + j] = f"合併 {j + 1}"
    return lines
