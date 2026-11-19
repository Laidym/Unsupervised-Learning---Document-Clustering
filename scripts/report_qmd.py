"""報告的原始檔 report.qmd：報告 report.html 就是由它 render 而來。

步驟 4（互動流程最後一步）寫出 report.qmd，內容為：
  - 分析流程：以本次實際使用的參數，依序執行步驟 1–4 的程式（程式碼摺疊顯示，輸出不顯示）
  - 報告各段落：呼叫 report_html 的段落函式，以原始 HTML 輸出（順序同 report_html.SECTIONS）
  - 重跑方式
接著在沙盒以 qmd_render.py 執行這份 qmd 得到 report.html。使用者在自己的電腦以 Quarto（如 RStudio 的
Render）render 同一份 qmd，會從頭重跑分析並得到內容相同的報告。

結果資料夾另附 report_qmd.zip，解壓後為 report_qmd/：
  report.qmd      報告原始檔
  scripts/        本次使用的程式碼（換行一律 LF，與 text-clustering.zip 相同）
  vendor/jieba/   只在資料含中文字時附上
  data/           只在使用內附 20 Newsgroups 範例時附上
原始資料不附上，由使用者放進 report_qmd/（與 report.qmd 同一層），qmd 以檔名讀取，不需改程式碼。
claude.ai 上傳的檔案會被加上 8 碼十六進位前綴（如 cb6752b6-bbc.csv），檔名一律去掉前綴；qmd 也接受
帶前綴的名稱，以及 zip 與解壓後資料夾兩種形式。

qmd 使用 knitr 引擎：R 套件 reticulate 的 py_require() 自動下載與原分析相同版本的 Python 與套件，
使用者不需自行安裝 Python。各步驟以 import 後呼叫 main(argv) 的方式在同一個 Python 中執行。
"""
import json
import platform
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JIEBA_DIR = ROOT / "vendor" / "jieba"
sys.path.insert(0, str(HERE))
from qmd_render import render as render_qmd  # noqa: E402
from report_html import SECTIONS  # noqa: E402

STEPS = (("step1", "step1_vectorize", "步驟 1　前處理與向量化"), ("step2", "step2_cluster", "步驟 2　文件分群"),
         ("step3", "step3_keywords", "步驟 3　群關鍵詞與命名"))
PACKAGES = (("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
            ("matplotlib", "matplotlib"), ("openpyxl", "openpyxl"))
PKG_DIR = "report_qmd"
OUT = "output"
UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-(?=.)")      # claude.ai 上傳時加在檔名前的前綴
OLD_OUTPUTS = ("reproduce.zip", "reproduce")           # 舊版的重現套件


def source_files():
    """scripts/ 內所有 .py 與中文停用詞清單，依檔名排序。"""
    files = sorted(HERE.glob("*.py"))
    stop = HERE / "stopwords_zh.txt"
    return files + ([stop] if stop.exists() else [])


def lf_bytes(path):
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def environment():
    """[(pip 套件名稱, 版本)]：本次分析使用的套件；沒有安裝的略過。"""
    rows = []
    for module, pip_name in PACKAGES:
        try:
            mod = sys.modules.get(module) or __import__(module)
        except ImportError:
            continue
        ver = getattr(mod, "__version__", None)
        if ver:
            rows.append((pip_name, ver))
    return rows


def _take_value(args, flag):
    """移除 flag 與其後一個值，回傳 (其餘參數, 該值或 None)。"""
    if flag not in args:
        return list(args), None
    i = args.index(flag)
    return args[:i] + args[i + 2:], args[i + 1] if i + 1 < len(args) else None


def portable_args(step, args, names_info=None):
    """把實際參數改成可在別處重跑的形式：去掉 --out（由 qmd 指定），步驟 1 的 --input 另外記錄，
    步驟 3 的 --names-file / --suggestions-file 改為直接列出名稱。回傳 (參數, 資料路徑或 None)。"""
    args, _ = _take_value(list(args), "--out")
    data = None
    if step == "step1":
        args, data = _take_value(args, "--input")
    if step == "step3" and names_info:
        for flag, key in (("--names-file", "names"), ("--suggestions-file", "suggestions")):
            if flag in args:
                args, _ = _take_value(args, flag)
                if names_info.get(key):
                    args += [f"--{key}", *names_info[key]]
    return args, data


def step_commands(out_dir, load_json):
    """{步驟: {"args", "data"}}；沒有記錄的步驟不列入。"""
    work = Path(out_dir) / "work"
    names_path = work / "step3" / "names.json"
    names_info = load_json(names_path) if names_path.exists() else None
    out = {}
    for step, _, _ in STEPS:
        f = work / step / "command.json"
        if f.exists():
            args, data = portable_args(step, load_json(f)["args"], names_info)
            out[step] = {"args": args, "data": data}
    return out


def _base_name(path):
    return Path(str(path).replace("\\", "/").rstrip("/")).name


def data_name(path):
    """原始資料的檔名（或資料夾名稱），去掉 claude.ai 上傳時加的前綴；使用者把它放在 qmd 旁邊。"""
    return UPLOAD_PREFIX.sub("", _base_name(path))


def data_candidates(path):
    """qmd 依序嘗試的名稱：去掉前綴的名稱、原名稱，以及各自的 zip／資料夾形式。"""
    names = []
    for n in (data_name(path), _base_name(path)):
        alt = n[:-4] if n.lower().endswith(".zip") else (n + ".zip" if not Path(n).suffix else None)
        names += [n] + ([alt] if alt else [])
    return list(dict.fromkeys(names))


def _chunk(lang, code, options=()):
    head = "".join(f"#| {o}\n" for o in options)
    return f"```{{{lang}}}\n{head}{code.strip(chr(10))}\n```\n"


def _yaml_str(s):
    return json.dumps(str(s), ensure_ascii=False)


def build_qmd(cmds, env, python_version, data, candidates, subtitle, sections):
    """sections：要放進報告的 (標題, 函式名稱)，已略過本次沒有內容的段落。"""
    pins = ", ".join(f'"{p}=={v}"' for p, v in env)
    L = ["---", 'title: "文件分群報告"', f"subtitle: {_yaml_str(subtitle)}", "lang: zh-Hant", "format:",
         "  html:", "    embed-resources: true", "    toc: true", '    toc-title: "目錄"',
         "    code-fold: true", '    code-summary: "程式碼"', "engine: knitr", "execute:", "  warning: false",
         "---", ""]
    L += [_chunk("r", f'''
# 在 RStudio 按 Render 時準備 Python：自動下載與原分析相同版本的 Python 與套件（第一次需要網路）
if (!requireNamespace("reticulate", quietly = TRUE) || utils::packageVersion("reticulate") < "1.41") {{
  install.packages("reticulate", repos = "https://cloud.r-project.org")
}}
reticulate::py_require(c({pins}), python_version = "{python_version}")
''', ["include: false"])]
    setup = ["import os, shutil, sys", "from pathlib import Path", "",
             'os.environ["MPLBACKEND"] = "Agg"                    # 只存圖檔，不開視窗',
             'sys.path.insert(0, str(Path("scripts").resolve()))', ""]
    if data:
        setup += ["# 原始資料，與本檔放在同一個資料夾",
                  f"DATA = next((f for f in {list(candidates)!r} if Path(f).exists()), None)",
                  "if DATA is None:",
                  f'    raise FileNotFoundError("找不到 {data}：請把原始資料放進與 report.qmd 相同的資料夾")']
    setup += [f'OUT = "{OUT}"', "shutil.rmtree(OUT, ignore_errors=True)                # 每次都從頭重跑", "",
              "import report_html as R", "", "def show(html):",
              '    print("\\n```{=html}\\n" + html + "\\n```\\n")']
    L += [_chunk("python", "\n".join(setup), ["include: false"]),
          _chunk("python", 'show("<style>" + R.CSS + "</style>")', ["echo: false", "output: asis"])]

    L += ["## 分析流程", "",
          "本報告由 `report.qmd` 產生：以下四個步驟以本次的參數從頭執行分析，之後各節的數字與圖表都來自這裡的結果。"
          "點「程式碼」可展開；各步驟的完整程式碼在 `scripts/` 資料夾。", ""]
    for step, module, title in STEPS:
        c = cmds[step]
        args = (["'--input'", "DATA"] if c["data"] is not None else []) + [repr(a) for a in c["args"]]
        args += ["'--out'", "OUT"]
        L += [f"### {title}", "", _chunk("python", f"from {module} import main\nmain([{', '.join(args)}])",
                                         ["output: false"])]
    L += ["### 步驟 4　評估與限制", "",
          _chunk("python", "from step4_evaluate import evaluate\nctx = evaluate(OUT)", ["output: false"])]

    for title, name in sections:
        L += [f"## {title}", ""]
        L += [_chunk("python", f"show(R.{name}(ctx))", ["echo: false", "output: asis"])]

    L += ["## 重跑本報告", ""]
    if data:
        L += [f"1. 把原始資料 `{data}` 放進本資料夾（與 `report.qmd` 同一層）。"]
    else:
        L += ["1. 本次使用 skill 內附的 20 Newsgroups 範例資料（已附在 `data/`），不需另外準備資料。"]
    L += ["2. 在 RStudio 開啟 `report.qmd`，按 **Render**。", "",
          "第一次執行會自動安裝 R 套件 reticulate，並下載與原分析相同版本的 Python 與套件"
          f"（Python {python_version}、" + "、".join(f"{p} {v}" for p, v in env) + "），"
          "需要網路，可能要幾分鐘；之後會沿用已下載的環境，不需自行安裝 Python。"
          f"分析的中間結果在 `{OUT}/`。", ""]
    return "\n".join(L)


def _package_files(qmd, cfg1, cmds):
    files = [("report.qmd", qmd.encode("utf-8"))]
    files += [(f"scripts/{p.name}", lf_bytes(p)) for p in source_files()]
    if (cfg1.get("zh_ratio") or 0) > 0 and JIEBA_DIR.exists():
        files += [(f"vendor/jieba/{p.relative_to(JIEBA_DIR).as_posix()}", p.read_bytes())
                  for p in sorted(JIEBA_DIR.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    if "--dataset" in cmds["step1"]["args"]:          # 內附的 20 Newsgroups 範例資料
        files.append(("data/20ng_subset.csv", (ROOT / "data" / "20ng_subset.csv").read_bytes()))
    return files


def build_report(out_dir, result_dir, ctx, load_json):
    """寫出 report.qmd 並 render 成 <result_dir>/report.html，另寫出 report_qmd.zip。

    回傳 (report.html 路徑, report_qmd.zip 路徑, 錯誤訊息或 None)。qmd 無法執行時（例如找不到原始資料），
    改由 ctx 直接產生報告，並回傳錯誤訊息。
    """
    from report_html import render as render_direct

    out_dir, result_dir = Path(out_dir), Path(result_dir)
    cmds = step_commands(out_dir, load_json)
    missing = [s for s, _, _ in STEPS if s not in cmds]
    if missing:
        raise RuntimeError(f"找不到 {'、'.join(missing)} 的執行紀錄，請從步驟 1 重跑。")
    raw = cmds["step1"]["data"]
    data = data_name(raw) if raw else None
    if data:
        cmds["step1"]["data"] = data
    sections = [(t, n) for t, n in SECTIONS if n != "merge" or ctx.get("merge")]
    subtitle = f'{data or Path(ctx["cfg1"]["input"]).name}｜{ctx["n"]:,} 篇文件｜{ctx["algorithm_name"]}｜{ctx["k"]} 群'
    py = ".".join(platform.python_version_tuple()[:2])
    qmd = build_qmd(cmds, environment(), py, data, data_candidates(raw) if raw else (), subtitle, sections)

    for old in OLD_OUTPUTS:
        p = result_dir / old
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    pkg = result_dir / PKG_DIR
    if pkg.exists():
        shutil.rmtree(pkg)
    files = _package_files(qmd, ctx["cfg1"], cmds)
    for name, content in files:
        (pkg / name).parent.mkdir(parents=True, exist_ok=True)
        (pkg / name).write_bytes(content)
    zpath = result_dir / f"{PKG_DIR}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files:
            z.writestr(f"{PKG_DIR}/{name}", content)

    error = None
    hpath = result_dir / "report.html"
    try:
        if raw:                                        # 把原始資料放在 qmd 旁邊（不放進 zip）
            src = Path(raw)
            if src.is_dir():
                shutil.copytree(src, pkg / data)
            elif src.exists():
                shutil.copy2(src, pkg / data)
        hpath.write_text(render_qmd(pkg / "report.qmd"), encoding="utf-8")
        mismatch = _compare_metrics(out_dir / "work" / "step4" / "metrics.csv",
                                    pkg / OUT / "work" / "step4" / "metrics.csv")
        if mismatch:
            error = "report.qmd 重跑的指標與互動分析不同：" + mismatch
    except Exception as e:                             # noqa: BLE001  qmd 失敗時仍要給使用者報告
        error = f"無法由 report.qmd 產生報告（{e}），改直接產生。"
        hpath.write_text(render_direct(ctx), encoding="utf-8")
    finally:
        shutil.rmtree(pkg, ignore_errors=True)
    return hpath, zpath, error


def _compare_metrics(a, b, tol=1e-4):
    """互動分析與 qmd 重跑的指標應相同（同一份程式碼與 seed）；不同時回傳說明。"""
    x = pd.read_csv(a, encoding="utf-8-sig")
    y = pd.read_csv(b, encoding="utf-8-sig")
    if list(x["指標"]) != list(y["指標"]) or ((x["數值"] - y["數值"]).abs() > tol).any():
        return "、".join(f"{r.指標}（{r.範圍}）" for r in x.itertuples() if r.Index >= len(y)
                        or abs(r.數值 - y["數值"].iloc[r.Index]) > tol) or "指標項目不同"
    return None
