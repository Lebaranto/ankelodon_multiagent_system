# src/gaia_agent/tools/safe_code_run.py
from __future__ import annotations
import io, os, sys, uuid, base64, traceback, contextlib, tempfile, shutil
import multiprocessing as mp
from typing import Optional, Dict, Any, List
from pydantic import ValidationError
from langchain_core.tools import tool
from utils.code_run import (
    CodeRunRequest, CodeRunResult, EnvInfo,
    PlotArtifact, DataFrameArtifact,
)

# ====================== HELPERS ======================

def _b64_png(fig, dpi: int) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return data

def _clip_df(df, max_rows: int, max_cols: int):
    sub = df.iloc[:max_rows, :max_cols]
    head = sub.to_dict(orient="records")
    dtypes = {str(k): str(v) for k, v in sub.dtypes.to_dict().items()}
    return head, list(df.shape), dtypes

def _env_info() -> EnvInfo:
    try:
        import numpy as _np; nv = _np.__version__
    except Exception:
        nv = None
    try:
        import pandas as _pd; pv = _pd.__version__
    except Exception:
        pv = None
    return EnvInfo(numpy=nv, pandas=pv)

# ====================== CHILD PROCESS ======================

def _child_exec(payload: Dict[str, Any], queue: mp.Queue):
    """
    Изолированное выполнение user-кода:
      - урезанные builtins
      - безопасный open (read-only в sandbox)
      - белый список импортов
      - запрет сети
      - temp cwd + очистка
      - RLIMIT CPU/AS (Unix)
      - захват stdout/stderr
      - сбор matplotlib и pandas.DataFrame (по флагам)
    """
    import builtins, importlib

    code: str = payload["code"]
    limits: Dict[str, Any] = payload["limits"]
    allowed: List[str] = payload["allowed"]
    return_plots: bool = payload["return_plots"]
    return_dfs: bool = payload["return_dfs"]

    # ---------- OS limits (Unix) ----------
    try:
        import resource
        cpu = max(1, int(limits["timeout_seconds"]))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        # мягкий лимит RAM ~1.5GB (подстрой при необходимости)
        one_gb = 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (int(1.5 * one_gb), int(1.5 * one_gb)))
        # ограничим размеры файлов
        resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))
    except Exception:
        pass

    # ---------- Sandbox FS ----------
    workdir = tempfile.mkdtemp(prefix="ci_")
    os.chdir(workdir)

    # ---------- Network ban ----------
    try:
        import socket
        class _NoNet(socket.socket):
            def __init__(self, *a, **kw):
                raise OSError("Network disabled in sandbox")
        socket.socket = _NoNet  # type: ignore
    except Exception:
        pass

    # ---------- Builtins ----------
    safe_names = [
        "abs","all","any","bool","dict","float","int","len","list","max","min",
        "range","str","sum","print","enumerate","zip","map","filter","sorted",
        "reversed","complex","pow","divmod", "round", "next", "set", "tuple", "type", "isinstance", "issubclass",
    ]
    safe_builtins = {n: getattr(builtins, n) for n in safe_names}

    # сохранём реальный open, потом подменим на безопасный
    real_open = open

    def _safe_open(path, mode="r", *a, **kw):
        # Разрешаем ТОЛЬКО чтение, ТОЛЬКО внутри workdir
        if any(m in mode for m in ("w", "a", "+", "x")):
            raise PermissionError("Write access forbidden in sandbox")
        abspath = os.path.abspath(path)
        # запрещаем выход из песочницы и следование symlink наружу
        if not abspath.startswith(workdir + os.sep) and abspath != workdir:
            raise PermissionError("Access outside sandbox forbidden")
        # запретим двоичный write по flags
        return real_open(abspath, mode, *a, **kw)

    # удалим опасные builtins и поставим наш open
    for banned in ["exec","eval","__import__","compile","input","globals","locals","vars","dir","help","__build_class__"]:
        safe_builtins.pop(banned, None)
    safe_builtins["open"] = _safe_open

    # ---------- Import whitelist ----------
    real_import = builtins.__import__
    ALLOWED = set(allowed)
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        base = name.split(".")[0]
        if (name not in ALLOWED) and (base not in ALLOWED):
            raise ImportError(f"Module '{name}' is not allowed")
        return real_import(name, globals, locals, fromlist, level)

    glb: Dict[str, Any] = {"__builtins__": safe_builtins}
    lcl: Dict[str, Any] = {}

    # ---------- Matplotlib headless ----------
    plt = None
    if return_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            plt = _plt
        except Exception:
            plt = None

    # ---------- Preload whitelisted mods ----------
    preloads = [
        "math","random","statistics","datetime","re","json","fractions","decimal",
        "numpy","pandas","cmath",
        "matplotlib","matplotlib.pyplot"
    ]
    for mod in preloads:
        try:
            if (mod in ALLOWED) or (mod.split(".")[0] in ALLOWED):
                glb[mod.split(".")[-1]] = importlib.import_module(mod)
        except Exception:
            pass

    # включаем безопасный импорт
    safe_builtins["__import__"] = _safe_import

    # ---------- Execute ----------
    out_buf, err_buf = io.StringIO(), io.StringIO()
    status = "error"
    result_repr: Optional[str] = None
    plots: List[Dict[str, Any]] = []
    dataframes: List[Dict[str, Any]] = []

    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            exec(code, glb, lcl)
            status = "success"

            # вернём repr результата, если есть _ или result
            if "_" in lcl:
                result_repr = repr(lcl["_"])
            elif "result" in lcl:
                result_repr = repr(lcl["result"])

            # графики
            if plt is not None and return_plots:
                fig_nums = plt.get_fignums()[: int(limits["max_plots"])]
                for num in fig_nums:
                    fig = plt.figure(num)
                    b64 = _b64_png(fig, dpi=int(limits["plot_dpi"]))
                    plots.append({"data_base64": b64, "format": "png"})
                plt.close("all")

            # DataFrame’ы
            if return_dfs:
                try:
                    import pandas as _pd
                    for name, val in list(lcl.items()):
                        if isinstance(val, _pd.DataFrame):
                            if len(dataframes) >= int(limits["max_dataframes"]):
                                break
                            head, shape, dtypes = _clip_df(
                                val,
                                max_rows=int(limits["max_df_rows"]),
                                max_cols=int(limits["max_df_cols"]),
                            )
                            dataframes.append({
                                "name": str(name),
                                "head": head,
                                "shape": shape,
                                "dtypes": dtypes,
                            })
                except Exception:
                    pass

    except Exception:
        status = "error"
        print(traceback.format_exc(), file=err_buf)
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass

    queue.put({
        "status": status,
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
        "result_repr": result_repr,
        "plots": plots,
        "dataframes": dataframes,
    })

# ====================== HOST PROCESS ======================

def run_python_in_subprocess(req: CodeRunRequest) -> CodeRunResult:
    exec_id = str(uuid.uuid4())
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()

    payload = {
        "code": req.code,
        "limits": req.limits.model_dump(),
        "allowed": list(req.allowed_modules),
        "return_plots": bool(req.return_plots),
        "return_dfs": bool(req.return_dataframes),
    }

    p = ctx.Process(target=_child_exec, args=(payload, q), daemon=True)
    p.start()
    p.join(req.limits.timeout_seconds)

    status = "timeout"
    stdout = ""
    stderr = "Timed out."
    result_repr = None
    plots: List[PlotArtifact] = []
    dataframes: List[DataFrameArtifact] = []

    if p.is_alive():
        p.terminate()
        p.join(1)
    else:
        try:
            msg = q.get_nowait()
            status = msg.get("status", "error")
            stdout = (msg.get("stdout") or "")[: req.limits.max_stdout_chars]
            stderr = (msg.get("stderr") or "")[: req.limits.max_stderr_chars]
            result_repr = msg.get("result_repr")
            plots = [PlotArtifact(**p_) for p_ in msg.get("plots", [])]
            dataframes = [DataFrameArtifact(**d_) for d_ in msg.get("dataframes", [])]
        except Exception as e:
            status = "error"
            stderr = f"Worker crashed: {e}"

    return CodeRunResult(
        execution_id=exec_id,
        status=status,
        stdout=stdout,
        stderr=stderr,
        result_repr=result_repr,
        plots=plots,
        dataframes=dataframes,
        env=_env_info(),
    )

# ====================== LangChain TOOL ======================

@tool
def safe_code_run(code:str) -> str:
    """
    Safely execute Python code in an isolated subprocess with security restrictions.
    
    IMPORTANT - To see output, you MUST:
    - Use print() statements for output
    - Assign final result to variable 'result' or '_' 
    - Save data to variables for DataFrame/plot capture
    
    Examples:
    ✅ Good:
    result = 2 + 2
    print(f"Answer: {result}")
    
    ✅ Good:
    import numpy as np
    arr = np.array([1, 2, 3])
    print(arr.mean())
    
    ✅ Good:
    import pandas as pd
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    print(df)
    result = df.sum()
    
    ❌ Bad (no output):
    2 + 2  # This won't show anything
    
    Security features:
    - Whitelisted imports only (numpy, pandas, matplotlib, etc.)
    - Read-only file access within sandbox
    - Network disabled
    - Memory/CPU limits
    - Timeout protection
    
    Returns JSON with: status, stdout, stderr, result_repr, plots, dataframes, env info
    """

    # упаковываем запрос в JSON
    req = CodeRunRequest(
        code=code,
        # для первого запуска дайте запас
        limits=dict(timeout_seconds=35)  # или 45
    ).model_dump_json()

    res = run_python_in_subprocess(CodeRunRequest.model_validate_json(req))
    return res.model_dump_json()