"""產生 data/20ng_subset.csv（開發用，需網路；skill 執行時不會用到本檔）。

做法：
1. 以 scikit-learn 下載 20 Newsgroups（train + test），移除 headers、footers、quotes。
2. 只取 4 個類別；去除前後空白後刪除少於 MIN_WORDS 個詞的文件與重複文件。
3. 每類以固定 seed 隨機抽 N_PER_CLASS 篇，輸出 doc_id, text, label 三欄（UTF-8）。

用法：python tools/prepare_20ng.py [--data-home 快取資料夾]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

CATEGORIES = ["comp.graphics", "rec.sport.baseball", "sci.space", "talk.religion.misc"]
N_PER_CLASS = 200
MIN_WORDS = 20
SEED = 42

OUT = Path(__file__).resolve().parent.parent / "data" / "20ng_subset.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-home", default=None)
    args = ap.parse_args()

    bunch = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=False,
        data_home=args.data_home,
    )
    df = pd.DataFrame({
        "text": [t.strip() for t in bunch.data],
        "label": [bunch.target_names[i] for i in bunch.target],
    })
    n_raw = df["label"].value_counts().sort_index()

    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    df = df.drop_duplicates(subset="text")
    n_kept = df["label"].value_counts().sort_index()

    rng = np.random.default_rng(SEED)
    parts = []
    for cat in CATEGORIES:
        sub = df[df["label"] == cat]
        idx = rng.choice(len(sub), size=N_PER_CLASS, replace=False)
        parts.append(sub.iloc[np.sort(idx)])
    out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out.insert(0, "doc_id", [f"ng{i:04d}" for i in range(len(out))])

    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")

    print(pd.DataFrame({"原始": n_raw, "篩選後": n_kept, "抽樣": out["label"].value_counts()}))
    print(f"已寫入 {OUT}（{len(out)} 篇）")


if __name__ == "__main__":
    main()
