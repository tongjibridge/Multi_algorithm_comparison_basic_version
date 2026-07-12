"""界面状态与后台训练任务。

UI 与训练在不同线程：worker 线程只更新 ``RunState``，主线程用 ``ui.timer`` 轮询渲染，
从而安全地跨线程更新界面。
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core import explain as explain_mod
from app.core import pipeline as pipeline_mod
from app.core.data import DataConfig
from app.core.optimize import OptConfig
from app.core.space import Param


@dataclass
class AppConfig:
    """用户在界面上累积的配置。"""

    df: pd.DataFrame | None = None
    excel_name: str = ""
    columns: list[str] = field(default_factory=list)
    model_spaces: dict[str, list[Param]] = field(default_factory=dict)


@dataclass
class RunState:
    running: bool = False
    done: bool = False
    error: str | None = None
    progress: float = 0.0
    stage: str = ""
    logs: list[str] = field(default_factory=list)
    metrics_rows: list[dict] = field(default_factory=list)
    gallery: list[dict] = field(default_factory=list)  # {model,label,path}
    out_dir: str = ""

    def log(self, msg: str) -> None:
        self.logs.append(str(msg))

    def reset(self) -> None:
        self.running = False
        self.done = False
        self.error = None
        self.progress = 0.0
        self.stage = ""
        self.logs.clear()
        self.metrics_rows.clear()
        self.gallery.clear()


def run_job(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    model_keys: list[str],
    opt_cfg: OptConfig,
    model_spaces: dict[str, list[Param]],
    selected_plots: list[str],
    out_dir: str,
    fmt: str,
    dpi: int,
    top_k: int,
    scheme_key: str,
    state: RunState,
) -> None:
    """后台线程入口：依次训练所选模型并生成输出。"""
    state.running = True
    state.out_dir = out_dir
    results = []
    try:
        n = len(model_keys)
        for i, key in enumerate(model_keys):
            state.stage = f"训练模型 {i + 1}/{n}：{key}"
            state.log(f"\n========== 模型 {i + 1}/{n}：{key} ==========")
            r = pipeline_mod.train_one(
                df, data_cfg, key, opt_cfg, state.log,
                model_space=model_spaces.get(key),
            )
            results.append(r)
            state.log("  生成图表与输出…")
            outs = explain_mod.generate(r, selected_plots, out_dir, fmt, dpi, top_k,
                                        scheme_key=scheme_key, log=state.log)
            for label, path in outs:
                state.gallery.append({"model": r.model_name, "label": label,
                                      "path": path})
            state.log(f"  本模型完成，输出 {len(outs)} 项")
            state.progress = (i + 1) / n
        if results:
            state.metrics_rows = pipeline_mod.metrics_table(results).to_dict("records")
        state.log("\n全部完成 ✅")
    except Exception as exc:  # noqa: BLE001
        state.error = f"{type(exc).__name__}: {exc}"
        state.log("\n[错误] " + state.error)
        state.log(traceback.format_exc())
    finally:
        state.running = False
        state.done = True


def start_job(*args: Any) -> threading.Thread:
    t = threading.Thread(target=run_job, args=args, daemon=True)
    t.start()
    return t
