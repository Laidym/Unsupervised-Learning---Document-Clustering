"""每個 .txt 檔一篇文件：資料夾與 zip 輸入、子資料夾作為可選標籤、分群後依群匯出 zip。"""
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import read_table  # noqa: E402
from step1_vectorize import main as step1_main  # noqa: E402
from step2_cluster import main as step2_main  # noqa: E402
from step3_keywords import main as step3_main  # noqa: E402
from step4_evaluate import main as step4_main  # noqa: E402

ZH = Path(__file__).resolve().parent / "data" / "zh_sample.csv"


def _make_folder(root):
    d = pd.read_csv(ZH)
    seen = {}
    for i, (t, lab) in enumerate(zip(d["內容"], d["主題"])):
        seen[lab] = seen.get(lab, -1) + 1
        f = root / "corpus" / lab / f"{seen[lab]:03d}.txt"     # 不同資料夾有同名檔案
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(t, encoding="utf-8" if i % 2 else "cp950", errors="ignore")
    (root / "corpus" / ".DS_Store").write_text("x")
    return root / "corpus", d["主題"].nunique()


def test_read_folder_and_zip(tmp_path):
    folder, n_lab = _make_folder(tmp_path)
    df = read_table(folder)
    assert list(df.columns) == ["doc_id", "text", "folder"] and len(df) == 240
    assert df["doc_id"].is_unique and df["folder"].nunique() == n_lab
    assert "/" in df["doc_id"][0]

    zpath = tmp_path / "corpus.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in folder.rglob("*.txt"):
            z.write(f, Path("wrapper") / f.relative_to(folder))   # 外層多包一層資料夾
        z.writestr("__MACOSX/._x.txt", b"\x00\x01")
    dz = read_table(zpath)
    assert dz.sort_values("doc_id").reset_index(drop=True).equals(
        df.sort_values("doc_id").reset_index(drop=True))


def test_flat_folder_has_no_folder_column(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}.txt").write_text(f"文件 {i}", encoding="utf-8")
    assert list(read_table(tmp_path).columns) == ["doc_id", "text"]


@pytest.mark.parametrize("as_zip", [False, True])
def test_pipeline_exports_cluster_folders(tmp_path, as_zip):
    folder, _ = _make_folder(tmp_path)
    src = folder
    if as_zip:
        src = tmp_path / "in.zip"
        with zipfile.ZipFile(src, "w") as z:
            for f in folder.rglob("*.txt"):
                z.write(f, f.relative_to(folder))
    out = tmp_path / "out"
    step1_main(["--input", str(src), "--label-col", "folder", "--out", str(out)])
    step2_main(["--k-min", "3", "--k-max", "4", "--out", str(out)])
    step3_main(["--k", "4", "--names", "甲 乙", "乙/丙", "丙", "丁", "--out", str(out)])
    step4_main(["--out", str(out)])

    clusters = pd.read_csv(out / "clusters.csv", encoding="utf-8-sig", dtype={"doc_id": str})
    with zipfile.ZipFile(out / "clusters_files.zip") as z:
        names = z.namelist()
        assert len(names) == len(clusters)
        assert {n.split("/")[0] for n in names} == {"0_甲_乙", "1_乙_丙", "2_丙", "3_丁"}
        first = clusters.iloc[0]
        member = f"0_甲_乙/{first['doc_id'].replace('/', '__')}"
        assert z.read(member).decode("utf-8") == first["text"]
    assert "ARI" in (out / "report.html").read_text(encoding="utf-8")
