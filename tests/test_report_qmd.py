"""報告由 report.qmd render 而來：qmd 的內容、內附執行器、report_qmd.zip。"""
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from qmd_render import markdown, parse, render  # noqa: E402
from report_qmd import data_candidates, data_name, portable_args, source_files  # noqa: E402
from step1_vectorize import main as step1_main  # noqa: E402
from step2_cluster import main as step2_main  # noqa: E402
from step3_keywords import main as step3_main  # noqa: E402
from step4_evaluate import main as step4_main  # noqa: E402

ZH = Path(__file__).resolve().parent / "data" / "zh_sample.csv"


def test_portable_args_and_data_name():
    args, data = portable_args("step1", ["--input", "/mnt/d.csv", "--text-col", "標題", "內文", "--out", "o"])
    assert args == ["--text-col", "標題", "內文"] and data == "/mnt/d.csv"
    names = {"names": ["太空 探索", "運動"], "suggestions": ["太空", "體育"]}
    args, _ = portable_args("step3", ["--k", "2", "--names-file", "n.json", "--suggestions-file", "s.json"], names)
    assert args == ["--k", "2", "--names", "太空 探索", "運動", "--suggestions", "太空", "體育"]
    assert data_name("/mnt/user-data/uploads/bbc.csv") == "bbc.csv"
    assert data_name("C:\\data\\corpus\\") == "corpus"
    # claude.ai 上傳時加的前綴要去掉；qmd 也接受原名稱與 zip／資料夾兩種形式
    assert data_name("/mnt/user-data/uploads/cb6752b6-bbc-fulltext.zip") == "bbc-fulltext.zip"
    assert data_candidates("/mnt/user-data/uploads/cb6752b6-bbc-fulltext.zip") == [
        "bbc-fulltext.zip", "bbc-fulltext", "cb6752b6-bbc-fulltext.zip", "cb6752b6-bbc-fulltext"]
    assert data_name("/x/deadbeefcafe.csv") == "deadbeefcafe.csv"


def test_runner_follows_chunk_options(tmp_path):
    qmd = tmp_path / "t.qmd"
    qmd.write_text('''---
title: "標題"
subtitle: "副標 \\"引號\\""
---

```{r}
#| include: false
stop("R 區塊只給 Quarto 用")
```

```{python}
#| include: false
x = 41
```

## 第一節

段落有 `code` 與 **粗體**。

- 項目一
- 項目二

```{python}
#| output: false
x += 1
print("不顯示")
```

```{python}
#| echo: false
#| output: asis
print("\\n```{=html}\\n<div class='raw'>" + str(x) + "</div>\\n```\\n")
```

```{python}
print("一般輸出")
```
''', encoding="utf-8")
    page = render(qmd)
    assert "<h1>標題</h1>" in page and "副標 &quot;引號&quot;" in page
    assert "<div class='raw'>42</div>" in page                      # 原始 HTML 原樣保留
    assert "不顯示</" not in page and "x = 41" not in page            # output: false、include: false
    assert page.count('<details class="code-fold">') == 2            # echo: false 的區塊不顯示程式碼
    assert "<pre><code>一般輸出" in page and "stop(" not in page
    assert "<code>code</code>" in page and "<strong>粗體</strong>" in page and "<li>項目二</li>" in page
    assert '<a href="#sec-1">第一節</a>' in page


def test_parse_and_markdown_edge_cases():
    meta, blocks = parse("---\ntitle: \"t\"\n---\ntext\n```{python}\n#| echo: false\n# 註解\nx = 1\n```\n")
    assert meta == {"title": "t"} and blocks[1] == ("chunk", "python", {"echo": "false"}, "# 註解\nx = 1")
    assert markdown("1. 甲\n   續行\n2. 乙", []) == "<ol><li>甲 續行</li><li>乙</li></ol>"


def _unzip(result_dir):
    with zipfile.ZipFile(result_dir / "report_qmd.zip") as z:
        assert all(n.startswith("report_qmd/") for n in z.namelist())
        z.extractall(result_dir / "unzipped")
    return result_dir / "unzipped" / "report_qmd"


def _pipeline(src, out, k=3, names_file=None, label=True):
    step1_main(["--input", str(src), "--text-col", "內容", *(["--label-col", "主題"] if label else []),
                "--out", str(out)])
    step2_main(["--k-min", "2", "--k-max", "3", "--out", str(out)])
    if names_file:
        step3_main(["--k", str(k), "--names-file", str(names_file), "--out", str(out)])
    else:
        step3_main(["--k", str(k), "--names", *["甲乙丙"[:k]], "--out", str(out)])
    step4_main(["--out", str(out)])
    return out / f"kmedoids_k{k}"


def test_report_is_rendered_from_qmd(tmp_path, capsys):
    up = tmp_path / "uploads" / "cb6752b6-zh_sample.csv"             # claude.ai 上傳後的檔名帶前綴
    up.parent.mkdir()
    shutil.copy(ZH, up)
    nf = tmp_path / "names.json"
    nf.write_text(json.dumps(["甲", "乙", "丙"], ensure_ascii=False), encoding="utf-8")
    rd = _pipeline(up, tmp_path / "out", names_file=nf)
    assert "注意" not in capsys.readouterr().out                       # qmd 重跑的指標與互動分析相同

    report = (rd / "report.html").read_text(encoding="utf-8")
    assert "<h2 id=\"sec-1\">分析流程</h2>" in report and report.count('<details class="code-fold">') == 4
    assert "zh_sample.csv｜240 篇文件" in report and "cb6752b6" not in report
    heads = re.findall(r"<h2[^>]*>([^<]+)</h2>", report)
    assert heads == ["分析流程", "摘要", "各群", "群數的選擇", "內部指標", "事後比對", "限制",
                     "詞彙摘要（步驟 1）", "文字雲", "設定紀錄", "重跑本報告"]
    assert "ARI" in report and "<svg" in report and "data:image/png;base64," in report
    assert not (rd / "report_qmd").exists() and not (rd / "reproduce.zip").exists()

    pkg = _unzip(rd)
    assert (pkg / "vendor" / "jieba" / "dict.txt.big").exists()          # 中文資料附上 jieba
    assert not (pkg / "data").exists() and not list(pkg.glob("*.csv"))  # 原始資料不附上
    assert not (pkg / "output").exists()
    for p in source_files():                                             # 程式碼為 LF，與 skill 內相同
        assert (pkg / "scripts" / p.name).read_bytes() == p.read_bytes().replace(b"\r\n", b"\n")
    qmd = (pkg / "report.qmd").read_text(encoding="utf-8")
    assert "engine: knitr" in qmd and "reticulate::py_require(" in qmd and "code-fold: true" in qmd
    assert "['zh_sample.csv', 'cb6752b6-zh_sample.csv']" in qmd and str(tmp_path) not in qmd
    assert "--names-file" not in qmd and "'甲', '乙', '丙'" in qmd

    # 使用者在自己的電腦：把原始資料以原檔名放在 qmd 旁邊後 render，得到同樣的報告
    shutil.copy(ZH, pkg / "zh_sample.csv")
    again = render(pkg / "report.qmd")
    assert re.sub(r"\s+", " ", again) == re.sub(r"\s+", " ", report)


def test_missing_data_falls_back_with_notice(tmp_path, capsys):
    src = tmp_path / "zh_sample.csv"
    shutil.copy(ZH, src)
    out = tmp_path / "out"
    step1_main(["--input", str(src), "--text-col", "內容", "--out", str(out)])
    step2_main(["--k-min", "2", "--k-max", "3", "--out", str(out)])
    step3_main(["--k", "2", "--names", "甲", "乙", "--out", str(out)])
    src.unlink()                                                         # 原始資料在步驟 4 前被移走
    capsys.readouterr()
    step4_main(["--out", str(out)])
    assert "無法由 report.qmd 產生報告" in capsys.readouterr().out
    report = (out / "kmedoids_k2" / "report.html").read_text(encoding="utf-8")
    assert "<h2>各群</h2>" in report and (out / "kmedoids_k2" / "report_qmd.zip").exists()
    try:
        render(_unzip(out / "kmedoids_k2") / "report.qmd")
    except RuntimeError as e:
        assert "找不到 zh_sample.csv" in str(e) and "相同的資料夾" in str(e)
    else:
        raise AssertionError("缺少資料時應提示")


def test_sample_dataset_is_bundled_without_jieba(tmp_path):
    step1_main(["--dataset", "20ng", "--out", str(tmp_path)])
    step2_main(["--k-min", "2", "--k-max", "3", "--out", str(tmp_path)])
    step3_main(["--k", "2", "--names", "甲", "乙", "--out", str(tmp_path)])
    step4_main(["--out", str(tmp_path)])
    pkg = _unzip(tmp_path / "kmedoids_k2")
    assert (pkg / "data" / "20ng_subset.csv").exists() and not (pkg / "vendor").exists()
    qmd = (pkg / "report.qmd").read_text(encoding="utf-8")
    assert "DATA" not in qmd and "main(['--dataset', '20ng', '--out', OUT])" in qmd
