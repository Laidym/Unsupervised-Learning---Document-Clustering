"""k-medoids：k-medoids++ 初始化＋FasterPAM 交換（Schubert & Rousseeuw 2021）。

只依賴 numpy，輸入為預先算好的距離矩陣。

FasterPAM 的要點：
- 對每個物件記錄最近、次近 medoid 的距離，因此評估「以候選 xc 取代某個 medoid」
  時，可在 O(n) 內同時算出 k 種取代方式的成本變化，不必逐一重算。
- 只要找到能降低總成本的交換就立即執行（eager swapping），
  候選依隨機順序循環檢查，直到完整一輪都找不到改善為止。
"""
import numpy as np


def cosine_distance_matrix(vectors, dtype=np.float32):
    """向量已正規化為長度 1，cosine 距離 = 1 − 內積。"""
    v = np.asarray(vectors, dtype=dtype)
    d = 1.0 - v @ v.T
    np.clip(d, 0.0, 2.0, out=d)
    np.fill_diagonal(d, 0.0)
    return d


def _kmedoids_pp(D, k, rng):
    """k-medoids++：第一個中心隨機，其後依「到最近中心距離的平方」比例抽樣。"""
    n = D.shape[0]
    medoids = [int(rng.integers(n))]
    dmin = D[medoids[0]].astype(np.float64)
    for _ in range(1, k):
        w = dmin ** 2
        total = w.sum()
        if total <= 0:  # 剩下的點都與現有中心重合
            cand = np.setdiff1d(np.arange(n), medoids)
            nxt = int(rng.choice(cand))
        else:
            nxt = int(rng.choice(n, p=w / total))
        medoids.append(nxt)
        np.minimum(dmin, D[nxt], out=dmin)
    return np.array(medoids)


def _assign(D, medoids):
    """回傳 nearest（第幾個 medoid）、dn（最近距離）、ds（次近距離）。"""
    dm = D[:, medoids].astype(np.float64)
    order = np.argsort(dm, axis=1)
    rows = np.arange(D.shape[0])
    nearest = order[:, 0]
    dn = dm[rows, nearest]
    ds = dm[rows, order[:, 1]] if len(medoids) > 1 else np.full_like(dn, np.inf)
    return nearest, dn, ds


def _removal_loss(nearest, dn, ds, k):
    """移除各 medoid 時，原屬該群的物件改歸次近 medoid 所增加的成本。"""
    return np.bincount(nearest, weights=ds - dn, minlength=k)


def fasterpam(D, k, rng, max_passes=100):
    n = D.shape[0]
    medoids = _kmedoids_pp(D, k, rng)
    is_medoid = np.zeros(n, dtype=bool)
    is_medoid[medoids] = True
    nearest, dn, ds = _assign(D, medoids)
    rl = _removal_loss(nearest, dn, ds, k)

    order = rng.permutation(n)
    n_swaps = 0
    since_swap = 0          # 連續檢查而未交換的候選數；達 n 表示完整一輪無改善，收斂
    for step in range(max_passes * n):
        if since_swap >= n:
            break
        xc = order[step % n]
        since_swap += 1
        if not is_medoid[xc]:
            d = D[:, xc].astype(np.float64)
            closer = d < dn
            # 改歸 xc 的物件：無論移除哪個 medoid 都會省下 dn − d
            gain_xc = np.sum(d[closer] - dn[closer])
            delta = rl.copy()
            # 這些物件不再依賴次近 medoid，移除其最近 medoid 不再有額外成本
            delta += np.bincount(nearest[closer], weights=dn[closer] - ds[closer], minlength=k)
            # 次近距離被 xc 取代的物件：移除其最近 medoid 時改歸 xc 而非原次近
            mid = ~closer & (d < ds)
            delta += np.bincount(nearest[mid], weights=d[mid] - ds[mid], minlength=k)
            i = int(np.argmin(delta))
            if delta[i] + gain_xc < -1e-12:
                is_medoid[medoids[i]] = False
                medoids[i] = xc
                is_medoid[xc] = True
                nearest, dn, ds = _assign(D, medoids)
                rl = _removal_loss(nearest, dn, ds, k)
                n_swaps += 1
                since_swap = 0
    labels, dn, _ = _assign(D, medoids)
    return medoids, labels, float(dn.sum()), n_swaps


def kmedoids(D, k, seed=42, n_init=5):
    """多次初始化取總成本最低者。回傳 dict：medoids、labels、cost、n_init。

    labels 依 medoid 的文件索引排序後重新編號為 0..k−1，結果與初始化順序無關。
    """
    n = D.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"群數需介於 1 與文件數之間（k={k}，n={n}）")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        medoids, labels, cost, _ = fasterpam(D, k, rng)
        if best is None or cost < best[2] - 1e-9:
            best = (medoids.copy(), labels, cost)
    medoids, labels, cost = best
    perm = np.argsort(medoids)
    remap = np.empty(k, dtype=int)
    remap[perm] = np.arange(k)
    return {"medoids": medoids[perm], "labels": remap[labels], "cost": cost, "n_init": n_init}
