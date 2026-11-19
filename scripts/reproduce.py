"""重現套件：讓使用者在自己的電腦以相同的程式碼與參數重跑四個步驟。

步驟 4 在結果資料夾寫出 reproduce/：
  reproduce.qmd   依步驟 1–4 執行本次實際使用的參數；在 RStudio 開啟後按 Render 即可
  scripts/        本次使用的程式碼（換行一律 LF，與 text-clustering.zip 相同）
  vendor/jieba/   只在資料含中文字時附上
  data/           只在使用內附 20 Newsgroups 範例時附上
原始資料不附上，由使用者放進 reproduce/（與 reproduce.qmd 同一層），qmd 以檔名讀取，不需改程式碼。

qmd 使用 knitr 引擎：R 套件 reticulate 的 py_require() 自動下載與原分析相同版本的 Python 與套件，
使用者不需自行安裝 Python。各步驟以 import 後呼叫 main(argv) 的方式在同一個 Python 中執行。
各步驟執行時以 common.record_command 把參數存成 work/stepN/command.json，這裡據此產生 qmd。
"""
import os
import platform
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JIEBA_DIR = ROOT / "vendor" / "jieba"
STEPS = (("step1", "step1_vectorize", "步驟 1　前處理與向量化"), ("step2", "step2_cluster", "步驟 2　文件分群"),
         ("step3", "step3_keywords", "步驟 3　群關鍵詞與命名"), ("step4", "step4_evaluate", "步驟 4　評估與限制"))
PACKAGES = (("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
            ("matplotlib", "matplotlib"), ("openpyxl", "openpyxl"))
OUT = "output"
REPRODUCING = "TEXT_CLUSTERING_REPRODUCING"     # qmd 重跑時設定，避免在重跑結果內再寫一份套件


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


def data_name(path):
    """原始資料的檔名（或資料夾名稱）；使用者把它放在 qmd 旁邊。"""
    return Path(str(path).replace("\\", "/").rstrip("/")).name


def _chunk(lang, code, options=()):
    head = "".join(f"#| {o}\n" for o in options)
    return f"```{{{lang}}}\n{head}{code.strip(chr(10))}\n```\n"


def build_qmd(cmds, env, python_version, data, result_name):
    pins = ", ".join(f'"{p}=={v}"' for p, v in env)
    L = ["---", 'title: "重現分群分析"', "lang: zh-Hant", "format:", "  html:", "    embed-resources: true",
         "    toc: true", "engine: knitr", "execute:", "  echo: true", "  warning: false", "---", ""]
    L += ["本文件以原分析相同的程式碼與參數，重跑產生 `report.html` 的四個步驟。", "", "## 使用方式", ""]
    if data:
        L += [f"1. 把原始資料 `{data}` 放進本資料夾（與 `reproduce.qmd` 同一層），檔名不要更改。",
              "2. 在 RStudio 開啟 `reproduce.qmd`，按 **Render**。"]
    else:
        L += ["本次使用 skill 內附的 20 Newsgroups 範例資料（已附在 `data/`），不需另外準備資料。",
              "在 RStudio 開啟 `reproduce.qmd`，按 **Render** 即可。"]
    L += ["",
          "第一次執行會自動安裝 R 套件 reticulate，並下載與原分析相同版本的 Python 與套件"
          f"（Python {python_version}、" + "、".join(f"{p} {v}" for p, v in env) + "），"
          "需要網路，可能要幾分鐘；之後會沿用已下載的環境。不需自行安裝 Python。", "",
          f"重跑的結果放在 `{OUT}/`，完整報告為 `{OUT}/{result_name}/report.html`。", ""]

    L += ["## 準備 Python 環境", "", _chunk("r", f'''
if (!requireNamespace("reticulate", quietly = TRUE) || utils::packageVersion("reticulate") < "1.41") {{
  install.packages("reticulate", repos = "https://cloud.r-project.org")
}}
reticulate::py_require(c({pins}), python_version = "{python_version}")
''', ["message: false"])]
    setup = ["import os, shutil, sys", "from pathlib import Path", "",
             'os.environ["MPLBACKEND"] = "Agg"                    # 只存圖檔，不開視窗',
             f'os.environ["{REPRODUCING}"] = "1"     # 重跑時不再產生重現套件',
             'sys.path.insert(0, str(Path("scripts").resolve()))', ""]
    if data:
        setup += [f"DATA = {data!r}                     # 原始資料，與本檔放在同一個資料夾",
                  "if not Path(DATA).exists():",
                  '    raise FileNotFoundError(f"找不到 {DATA}：請把原始資料放進與 reproduce.qmd 相同的資料夾")']
    setup += [f'OUT = "{OUT}"', "shutil.rmtree(OUT, ignore_errors=True)                # 每次都從頭重跑"]
    L += [_chunk("python", "\n".join(setup))]
    for step, module, title in STEPS:
        c = cmds[step]
        args = (["'--input'", "DATA"] if c["data"] is not None else []) + [repr(a) for a in c["args"]]
        args += ["'--out'", "OUT"]
        L += [f"## {title}", "", _chunk("python", f'''
from {module} import main
main([{", ".join(args)}])
''')]
        if step == "step2":
            L += [_chunk("r", f'knitr::include_graphics("{OUT}/work/step2/k_selection.png")')]
    L += ["## 重跑的報告", "", f"[開啟重跑產生的報告]({OUT}/{result_name}/report.html)", ""]
    return "\n".join(L)


def write_package(out_dir, result_dir, cfg1, load_json):
    """寫出 <result_dir>/reproduce/，回傳 (路徑, 資料檔名或 None)；qmd 重跑中或缺少任何步驟的指令紀錄時不寫，回傳 None。"""
    if os.environ.get(REPRODUCING):
        return None
    cmds = step_commands(out_dir, load_json)
    if any(step not in cmds for step, _, _ in STEPS):
        return None
    dest = Path(result_dir) / "reproduce"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "scripts").mkdir(parents=True)
    for p in source_files():
        (dest / "scripts" / p.name).write_bytes(lf_bytes(p))
    if (cfg1.get("zh_ratio") or 0) > 0 and JIEBA_DIR.exists():
        shutil.copytree(JIEBA_DIR, dest / "vendor" / "jieba", ignore=shutil.ignore_patterns("__pycache__"))
    if "--dataset" in cmds["step1"]["args"]:          # 內附的 20 Newsgroups 範例資料
        (dest / "data").mkdir()
        shutil.copy2(ROOT / "data" / "20ng_subset.csv", dest / "data" / "20ng_subset.csv")
    raw = cmds["step1"]["data"]
    data = data_name(raw) if raw else None
    if data:
        cmds["step1"]["data"] = data
    py = ".".join(platform.python_version_tuple()[:2])
    qmd = build_qmd(cmds, environment(), py, data, Path(result_dir).name)
    (dest / "reproduce.qmd").write_text(qmd, encoding="utf-8", newline="\n")
    return dest, data
