import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import DATA_20NG  # noqa: E402
from step1_vectorize import main as step1_main  # noqa: E402
from step2_cluster import main as step2_main  # noqa: E402
from step3_keywords import main as step3_main  # noqa: E402
from step4_evaluate import data_limits, external_metrics, main as step4_main, purity  # noqa: E402


def test_purity():
    c = np.array([0, 0, 0, 1, 1, 2])
    y = np.array(["a", "a", "b", "b", "b", "a"])
    assert purity(c, y) == pytest.approx((2 + 2 + 1) / 6)


def test_external_metrics_perfect_and_permuted():
    y = np.array(["a", "a", "b", "b", "c", "c"])
    e = external_metrics(np.array([2, 2, 0, 0, 1, 1]), y)
    assert e["ARI"] == pytest.approx(1) and e["NMI"] == pytest.approx(1) and e["純度"] == 1


def _limits(**kw):
    base = dict(n=800, k=4, sizes=[200, 200, 200, 200], npmi_per=[0.3] * 4, silhouette=0.2,
                eff_median=30, n_dropped=0, k_range=(2, 15))
    return data_limits(**{**base, **kw})


def test_data_limits():
    assert _limits() == []
    assert any("文件偏短" in s for s in _limits(eff_median=5))
    assert any("群 3（8 篇）" in s for s in _limits(sizes=[300, 300, 192, 8]))
    assert any("差異大" in s for s in _limits(sizes=[520, 100, 100, 80], k=4))
    assert any("群 1" in s for s in _limits(npmi_per=[0.3, -0.1, 0.2, 0.2]))
    low = _limits(npmi_per=[0.45, 0.38, 0.18, 0.37])     # 平均 0.345，群 2 低於平均 0.1 以上
    assert any("群 2（0.180）" in s for s in low) and not any("群 1" in s for s in low)
    assert any("上限" in s for s in _limits(k=15, sizes=[50] * 15 + [50]))
    assert any("未納入" in s for s in _limits(n_dropped=3))


def _pipeline(out, step1_args):
    step1_main([*step1_args, "--out", str(out)])
    step2_main(["--k-min", "3", "--k-max", "4", "--out", str(out)])
    step3_main(["--k", "4", "--names", "甲", "乙", "丙", "丁", "--out", str(out)])
    step4_main(["--out", str(out)])


def test_with_labels(tmp_path, capsys):
    _pipeline(tmp_path, ["--dataset", "20ng"])
    assert not (tmp_path / "metrics.csv").exists()                # 指標只放在報告中
    m = pd.read_csv(tmp_path / "work" / "step4" / "metrics.csv", encoding="utf-8-sig")
    assert {"silhouette", "NPMI", "ARI", "NMI", "純度"} <= set(m["指標"])
    assert (m["範圍"] == "整體").sum() == 5
    report = (tmp_path / "kmedoids_k4" / "report.html").read_text(encoding="utf-8")
    assert "ARI" in report and "交叉表" in report and "<dt>隨機種子</dt><dd>42" in report
    assert "名稱來源" in report
    assert "data:image/png;base64," in report                  # 圖片內嵌
    assert "http://" not in report and "https://" not in report   # 不依賴網路
    assert not (tmp_path / "report.md").exists() and not (tmp_path / "report.html").exists()


def test_without_labels(tmp_path):
    f = tmp_path / "in.csv"
    pd.read_csv(DATA_20NG)[["text"]].rename(columns={"text": "內容"}).to_csv(f, index=False)
    out = tmp_path / "out"
    _pipeline(out, ["--input", str(f), "--text-col", "內容"])
    m = pd.read_csv(out / "work" / "step4" / "metrics.csv", encoding="utf-8-sig")
    assert not {"ARI", "NMI", "純度"} & set(m["指標"])
    assert "略過事後比對" in (out / "kmedoids_k4" / "report.html").read_text(encoding="utf-8")


def test_requires_names(tmp_path):
    step1_main(["--dataset", "20ng", "--out", str(tmp_path)])
    step2_main(["--k-min", "3", "--k-max", "3", "--out", str(tmp_path)])
    step3_main(["--k", "3", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="命名"):
        step4_main(["--out", str(tmp_path)])
