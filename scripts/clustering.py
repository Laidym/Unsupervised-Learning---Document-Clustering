"""分群演算法的統一介面，以及大資料時的 silhouette 計算。步驟 2、4 共用。"""
import numpy as np
from sklearn.metrics import silhouette_score

from hierarchical import build_linkage, cut
from kmedoids import cosine_distance_matrix, kmedoids
from spherical_kmeans import spherical_kmeans

N_INIT = 5
KMEDOIDS_MAX = 5000          # 文件數 ≤ 此值建議 k-medoids，否則建議球面 k-means
SILHOUETTE_FULL_MAX = 5000   # 文件數 > 此值時，silhouette 以隨機抽樣計算

ALGORITHMS = {
    "kmedoids": "k-medoids（k-medoids++ 初始化＋FasterPAM）",
    "spherical": "球面 k-means（k-means++ 初始化）",
    "hierarchical": "階層式分群（Ward 連結）",
}
LARGE_N_WARNING = {"kmedoids", "hierarchical"}   # 需要 O(n²) 記憶體的演算法


def suggest_algorithm(n):
    return "kmedoids" if n <= KMEDOIDS_MAX else "spherical"


def needs_distance_matrix(algorithm, n):
    return algorithm == "kmedoids" or n <= SILHOUETTE_FULL_MAX


def _cosine_cost(vectors, labels, k):
    """各文件到所屬群平均方向的 cosine 距離總和。"""
    C = np.zeros((k, vectors.shape[1]))
    np.add.at(C, labels, vectors)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return float(np.sum(1.0 - np.einsum("ij,ij->i", vectors, C[labels])))


def run_clustering(vectors, k, algorithm, D=None, seed=42, n_init=N_INIT, L=None):
    """回傳 dict：labels、medoids（k-medoids 以外為 None）、cost。

    階層式分群可傳入預先建好的樹 L，避免每個 k 重建。
    """
    if algorithm == "kmedoids":
        if D is None:
            D = cosine_distance_matrix(vectors)
        res = kmedoids(D, k, seed=seed, n_init=n_init)
        return {"labels": res["labels"], "medoids": res["medoids"], "cost": res["cost"]}
    if algorithm == "spherical":
        res = spherical_kmeans(vectors, k, seed=seed, n_init=n_init)
        return {"labels": res["labels"], "medoids": None, "cost": res["cost"]}
    if algorithm == "hierarchical":
        if L is None:
            L = build_linkage(vectors)
        labels = cut(L, k)
        return {"labels": labels, "medoids": None, "cost": _cosine_cost(vectors, labels, k)}
    raise ValueError(f"未知的演算法：{algorithm}")


def silhouette(vectors, labels, D=None, seed=42):
    """文件數 ≤ 5,000 用完整的 cosine 距離矩陣；否則隨機抽 5,000 篇計算（固定 seed）。

    回傳 (silhouette, 是否抽樣)。
    """
    n = len(labels)
    if n <= SILHOUETTE_FULL_MAX:
        if D is None:
            D = cosine_distance_matrix(vectors)
        return float(silhouette_score(D, labels, metric="precomputed")), False
    return float(silhouette_score(vectors, labels, metric="cosine",
                                  sample_size=SILHOUETTE_FULL_MAX, random_state=seed)), True
