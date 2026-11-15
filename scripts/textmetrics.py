"""群關鍵詞（c-TF-IDF）與關鍵詞一致性（NPMI）。步驟 2、3、4 共用。"""
import numpy as np
from scipy import sparse

TOP_N = 10


def _cluster_term_matrix(counts, labels, k):
    n = counts.shape[0]
    onehot = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(k, n))
    return np.asarray((onehot @ counts).todense(), dtype=np.float64)


def ctfidf(counts, labels, k):
    """c-TF-IDF（Grootendorst 2022）：W[t,c] = tf[t,c] · log(1 + A / f[t])。

    tf[t,c]：詞 t 在群 c 的次數除以群 c 總詞數；A：每群平均詞數；f[t]：詞 t 在所有群的總次數。
    """
    C = _cluster_term_matrix(counts, labels, k)
    size = C.sum(axis=1, keepdims=True)
    tf = np.divide(C, size, out=np.zeros_like(C), where=size > 0)
    f = C.sum(axis=0)
    A = C.sum() / k
    idf = np.log1p(np.divide(A, f, out=np.zeros_like(f), where=f > 0))
    return tf * idf


def top_terms(counts, labels, k, vocab, top_n=TOP_N):
    """回傳每群前 top_n 個關鍵詞：詞彙索引、詞、c-TF-IDF 權重。權重為 0 的詞不列入。"""
    W = ctfidf(counts, labels, k)
    out = []
    for c in range(k):
        idx = [int(i) for i in np.argsort(-W[c], kind="stable")[:top_n] if W[c, i] > 0]
        out.append({"index": idx, "terms": [vocab[i] for i in idx],
                    "weights": [float(W[c, i]) for i in idx]})
    return out


def npmi_matrix(binary, idx):
    """以文件內是否共同出現估計機率，回傳 idx 中各詞兩兩的 NPMI 矩陣。"""
    n = binary.shape[0]
    B = binary[:, idx]
    co = np.asarray((B.T @ B).todense(), dtype=np.float64) / n
    p = np.diag(co).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(co / np.outer(p, p))
        npmi = pmi / -np.log(co)
    npmi[co == 0] = -1.0
    npmi[co >= 1.0] = 1.0           # 兩詞出現在每一篇文件：視為完全共現
    return npmi


def cluster_npmi(counts, keyword_idx):
    """每群前 10 個關鍵詞兩兩 NPMI 的平均；回傳 (整體平均, 各群平均)。

    先對群內所有詞對取平均，再對所有群取平均。關鍵詞少於 2 個的群記為 NaN 並略過。
    """
    binary = (counts > 0).astype(np.float64).tocsc()
    per = []
    for idx in keyword_idx:
        if len(idx) < 2:
            per.append(float("nan"))
            continue
        M = npmi_matrix(binary, idx)
        iu = np.triu_indices(len(idx), k=1)
        per.append(float(M[iu].mean()))
    per = np.array(per)
    overall = float(np.nanmean(per)) if np.isfinite(per).any() else float("nan")
    return overall, per
