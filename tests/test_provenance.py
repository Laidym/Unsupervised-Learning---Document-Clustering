import base64
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import load_json, record_command  # noqa: E402
from provenance import STEP_FILES, commands, source_files  # noqa: E402
from step1_vectorize import main as step1_main  # noqa: E402
from step2_cluster import main as step2_main  # noqa: E402
from step3_keywords import main as step3_main  # noqa: E402
from step4_evaluate import main as step4_main  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_step3_names_file_becomes_inline_names(tmp_path):
    nf = tmp_path / "n.json"
    record_command(tmp_path, "step3", "step3_keywords.py", ["--k", "2", "--names-file", str(nf), "--out", "o"])
    (tmp_path / "work" / "step3" / "names.json").write_text(
        json.dumps({"names": ["太空 探索", "運動"], "suggestions": None}), encoding="utf-8")
    cmds = commands(tmp_path, load_json)
    assert list(cmds) == ["step3"]
    cmd, note = cmds["step3"]
    assert cmd == "python3 scripts/step3_keywords.py --k 2 --names '太空 探索' 運動 --out o"
    assert "等效指令" in note


def test_report_embeds_code_and_commands(tmp_path):
    nf = tmp_path / "names.json"
    nf.write_text(json.dumps(["甲", "乙", "丙", "丁"]), encoding="utf-8")
    step1_main(["--dataset", "20ng", "--out", str(tmp_path)])
    step2_main(["--k-min", "3", "--k-max", "4", "--out", str(tmp_path)])
    step3_main(["--k", "4", "--names-file", str(nf), "--out", str(tmp_path)])
    step4_main(["--out", str(tmp_path)])
    html = (tmp_path / "kmedoids_k4" / "report.html").read_text(encoding="utf-8")

    # 報告依步驟分段，各步驟的指令與程式碼放在該步驟內
    heads = [html.index(f"<h2>步驟 {n}") for n in (1, 2, 3, 4)] + [html.index("<h2>設定紀錄")]
    assert heads == sorted(heads)
    cmds = {"step1": "step1_vectorize.py --dataset 20ng", "step2": "step2_cluster.py --k-min 3 --k-max 4",
            "step3": "step3_keywords.py --k 4 --names 甲 乙 丙 丁", "step4": "step4_evaluate.py --out"}
    for n in range(4):
        step = f"step{n + 1}"
        assert heads[n] < html.index(cmds[step]) < heads[n + 1]
        for name in STEP_FILES[step]:
            assert heads[n] < html.index(f"<strong>{name}</strong>") < heads[n + 1], name
    assert heads[1] < html.index("<h3>群數的選擇") < heads[2] and heads[2] < html.index("<h3>各群</h3>") < heads[3]
    assert heads[3] < html.index("<h3>內部指標") and heads[0] < html.index("<h3>詞彙摘要") < heads[1]
    # 每個程式檔都嵌入，SHA-256 為 LF 換行後的內容
    found = dict(re.findall(r"<strong>([\w.]+)</strong><span class=\"muted\">[\d,]+ 行</span>"
                            r"<code class=\"hash\">SHA-256 (\w+)", html))
    files = source_files()
    assert set(found) == {p.name for p in files} and "wordcloud.py" in found and "stopwords_zh.txt" in found
    for p in files:
        assert found[p.name] == hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    # 可下載的 zip 內容與程式碼一致
    uri = re.search(r'href="data:application/zip;base64,([^"]+)" download="scripts.zip"', html).group(1)
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(uri)))
    assert sorted(z.namelist()) == sorted(f"scripts/{p.name}" for p in files)
    assert z.read("scripts/textmetrics.py") == (SCRIPTS / "textmetrics.py").read_bytes().replace(b"\r\n", b"\n")
    assert "scikit-learn" in html and "jieba（內附） 0.42.1" in html


def test_every_source_file_belongs_to_one_step():
    mapped = [n for names in STEP_FILES.values() for n in names]
    assert len(mapped) == len(set(mapped))
    assert {p.name for p in source_files()} == set(mapped)
