import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from reproduce import REPRODUCING, data_name, portable_args, source_files  # noqa: E402
from step1_vectorize import main as step1_main  # noqa: E402
from step2_cluster import main as step2_main  # noqa: E402
from step3_keywords import main as step3_main  # noqa: E402
from step4_evaluate import main as step4_main  # noqa: E402

ZH = Path(__file__).resolve().parent / "data" / "zh_sample.csv"


def test_portable_args():
    args, data = portable_args("step1", ["--input", "/mnt/d.csv", "--text-col", "標題", "內文", "--out", "o"])
    assert args == ["--text-col", "標題", "內文"] and data == "/mnt/d.csv"
    names = {"names": ["太空 探索", "運動"], "suggestions": ["太空", "體育"]}
    args, _ = portable_args("step3", ["--k", "2", "--names-file", "n.json", "--suggestions-file", "s.json"], names)
    assert args == ["--k", "2", "--names", "太空 探索", "運動", "--suggestions", "太空", "體育"]
    assert data_name("/mnt/user-data/uploads/bbc.csv") == "bbc.csv"
    assert data_name("C:\\data\\corpus\\") == "corpus"


def _run_python_chunks(pkg, monkeypatch):
    """依序執行 reproduce.qmd 的 Python 區塊（等同 Render，但不需 R 與 Quarto）。"""
    monkeypatch.chdir(pkg)
    monkeypatch.setenv(REPRODUCING, "")          # 由 qmd 自己設定
    src = (pkg / "reproduce.qmd").read_text(encoding="utf-8")
    ns = {}
    for chunk in re.findall(r"```\{python\}\n(.*?)```", src, re.S):
        exec(chunk, ns)


def test_reproduce_package_reruns(tmp_path, monkeypatch):
    out = tmp_path / "out"
    nf = tmp_path / "names.json"
    nf.write_text(json.dumps(["甲", "乙", "丙"], ensure_ascii=False), encoding="utf-8")
    step1_main(["--input", str(ZH), "--text-col", "內容", "--label-col", "主題", "--out", str(out)])
    step2_main(["--k-min", "2", "--k-max", "3", "--out", str(out)])
    step3_main(["--k", "3", "--names-file", str(nf), "--out", str(out)])
    step4_main(["--out", str(out)])

    pkg = out / "kmedoids_k3" / "reproduce"
    assert (pkg / "vendor" / "jieba" / "dict.txt.big").exists()          # 中文資料附上 jieba
    assert not (pkg / "data").exists() and not (pkg / "zh_sample.csv").exists()   # 原始資料不附上
    for p in source_files():                                             # 程式碼為 LF，與 skill 內相同
        assert (pkg / "scripts" / p.name).read_bytes() == p.read_bytes().replace(b"\r\n", b"\n")
    qmd = (pkg / "reproduce.qmd").read_text(encoding="utf-8")
    assert "engine: knitr" in qmd and "reticulate::py_require(" in qmd
    assert "DATA = 'zh_sample.csv'" in qmd and str(ZH.parent) not in qmd   # 只用檔名，不含原路徑
    assert "--names-file" not in qmd and "'甲', '乙', '丙'" in qmd
    assert "與報告比對" not in qmd

    report = (out / "kmedoids_k3" / "report.html").read_text(encoding="utf-8")
    assert "reproduce.qmd" in report and "<code>zh_sample.csv</code>" in report

    shutil.copy(ZH, pkg / "zh_sample.csv")                               # 使用者把資料放在 qmd 旁邊
    _run_python_chunks(pkg, monkeypatch)
    rerun = pkg / "output" / "kmedoids_k3"
    assert (rerun / "report.html").exists() and not (rerun / "reproduce").exists()
    assert (rerun / "clusters.csv").read_bytes() == (out / "kmedoids_k3" / "clusters.csv").read_bytes()


def test_missing_data_gives_clear_error(tmp_path, monkeypatch):
    out = tmp_path / "out"
    step1_main(["--input", str(ZH), "--text-col", "內容", "--out", str(out)])
    step2_main(["--k-min", "2", "--k-max", "3", "--out", str(out)])
    step3_main(["--k", "2", "--names", "甲", "乙", "--out", str(out)])
    step4_main(["--out", str(out)])
    try:
        _run_python_chunks(out / "kmedoids_k2" / "reproduce", monkeypatch)
    except FileNotFoundError as e:
        assert "zh_sample.csv" in str(e) and "相同的資料夾" in str(e)
    else:
        raise AssertionError("缺少資料時應提示")


def test_sample_dataset_is_bundled_without_jieba(tmp_path):
    step1_main(["--dataset", "20ng", "--out", str(tmp_path)])
    step2_main(["--k-min", "2", "--k-max", "3", "--out", str(tmp_path)])
    step3_main(["--k", "2", "--names", "甲", "乙", "--out", str(tmp_path)])
    step4_main(["--out", str(tmp_path)])
    pkg = tmp_path / "kmedoids_k2" / "reproduce"
    assert (pkg / "data" / "20ng_subset.csv").exists() and not (pkg / "vendor").exists()
    qmd = (pkg / "reproduce.qmd").read_text(encoding="utf-8")
    assert "DATA" not in qmd and "main(['--dataset', '20ng', '--out', OUT])" in qmd
