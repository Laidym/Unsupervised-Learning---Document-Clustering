"""球面 k-means（Dhillon & Modha 2001）：只依賴 numpy。

輸入為長度已正規化為 1 的向量。反覆執行：
1. 將每篇文件分給 cosine 相似度最高的中心；
2. 以群內向量的總和（等同平均的方向）更新中心，並將中心長度縮放回 1。
直到分群不再改變。初始化採 k-means++（依「1 − 與最近中心的相似度」的平方比例抽樣），
多次初始化取目標函數（總 cosine 距離）最小者。
"""
import numpy as np


def _init_pp(X, k, rng):
    n = X.shape[0]
    centers = [int(rng.integers(n))]
    dmin = 1.0 - X @ X[centers[0]]
    np.clip(dmin, 0.0, 2.0, out=dmin)
    for _ in range(1, k):
        w = dmin ** 2
        total = w.sum()
        if total <= 0:
            cand = np.setdiff1d(np.arange(n), centers)
            nxt = int(rng.choice(cand))
        else:
            nxt = int(rng.choice(n, p=w / total))
        centers.append(nxt)
        d = np.clip(1.0 - X @ X[nxt], 0.0, 2.0)
        np.minimum(dmin, d, out=dmin)
    return X[centers].copy()


def _normalize_rows(C):
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    return np.divide(C, norms, out=np.zeros_like(C), where=norms > 0)


def _run_once(X, k, rng, max_iter):
    C = _init_pp(X, k, rng)
    labels = None
    for _ in range(max_iter):
        S = X @ C.T
        new = np.argmax(S, axis=1)
        sizes = np.bincount(new, minlength=k)
        # 空群：改以目前最不像自己中心的文件作為新中心
        for j in np.flatnonzero(sizes == 0):
            far = int(np.argmin(S[np.arange(len(new)), new]))
            new[far] = j
            S[far, :] = -np.inf
            S[far, j] = 1.0
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        C = np.zeros_like(C)
        np.add.at(C, labels, X)
        C = _normalize_rows(C)
    sims = np.einsum("ij,ij->i", X, C[labels])
    return labels, C, float(np.sum(1.0 - sims))


def spherical_kmeans(X, k, seed=42, n_init=5, max_iter=100):
    """回傳 dict：labels、centers、cost（總 cosine 距離）、n_init。

    群編號依「群內最小的文件索引」排序後重新編號，結果與初始化順序無關。
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"群數需介於 1 與文件數之間（k={k}，n={n}）")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        labels, C, cost = _run_once(X, k, rng, max_iter)
        if best is None or cost < best[2] - 1e-9:
            best = (labels, C, cost)
    labels, C, cost = best
    first = np.array([np.flatnonzero(labels == j).min() for j in range(k)])
    perm = np.argsort(first)
    remap = np.empty(k, dtype=int)
    remap[perm] = np.arange(k)
    return {"labels": remap[labels], "centers": C[perm], "cost": cost, "n_init": n_init}
