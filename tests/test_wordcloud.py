import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from wordcloud import MAX_PX, MIN_PX, build_items, cluster_anchors, layout, render_svg  # noqa: E402

CLOUDS = [
    {"cluster": 0, "name": "太空", "terms": ["space", "orbit", "earth", "moon"], "weights": [4.0, 2.0, 1.0, 0.0]},
    {"cluster": 1, "name": "宗教", "terms": ["god", "earth", "faith"], "weights": [3.0, 3.0, 1.5]},
]


def _overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def test_shared_words_merged_and_sized_by_relative_weight():
    items = {d["term"]: d for d in build_items(CLOUDS)}
    assert len(items) == 6                                       # earth 只出現一次
    assert items["earth"]["clusters"] == [0, 1]
    assert items["earth"]["size"] == MAX_PX                      # 在群 1 的相對權重為 1
    assert items["space"]["size"] == MAX_PX and items["god"]["size"] == MAX_PX
    assert items["orbit"]["size"] == MIN_PX + (MAX_PX - MIN_PX) * 0.5
    assert items["moon"]["size"] == MIN_PX


def test_layout_has_no_overlaps_and_is_reproducible():
    anchors = np.array([[-1.0, 0.0], [1.0, 0.0]])
    labels, words, _ = layout(CLOUDS, anchors)
    boxes = [lb["box"] for lb in labels] + [w["box"] for w in words]
    assert not any(_overlap(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    assert render_svg(CLOUDS, anchors) == render_svg(CLOUDS, anchors)


def test_svg_marks_shared_and_cluster_words():
    svg = render_svg(CLOUDS, np.array([[-1.0, 0.0], [1.0, 0.0]]))
    assert re.search(r'<text class="shared"[^>]*><title>earth｜2 群共有', svg)
    assert re.search(r'<text class="s1"[^>]*font-weight="700"><title>space', svg)
    assert re.search(r'<text class="s2"[^>]*><title>faith', svg)
    many = [{"cluster": c, "name": f"g{c}", "terms": [f"t{c}"], "weights": [1.0]} for c in range(9)]
    assert 'class="s' not in render_svg(many, np.zeros((9, 2)))   # 超過 8 群不以顏色區分


def test_anchors_place_similar_clusters_closer():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3, 50))
    base[1] = base[0] + 0.1 * rng.normal(size=50)                # 群 0、1 相似，群 2 不同
    X = np.repeat(base, 10, axis=0) + 0.01 * rng.normal(size=(30, 50))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    xy = cluster_anchors(X, np.repeat(np.arange(3), 10), 3)
    d = lambda i, j: np.linalg.norm(xy[i] - xy[j])               # noqa: E731
    assert d(0, 1) < d(0, 2) and d(0, 1) < d(1, 2)
    assert np.allclose(xy, cluster_anchors(X, np.repeat(np.arange(3), 10), 3))
