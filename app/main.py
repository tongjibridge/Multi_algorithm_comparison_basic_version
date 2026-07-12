"""可解释机器学习 GUI —— NiceGUI 入口。

工作流（自上而下）：
  ① 导入 Excel 并指定列  →  ② 选择模型  →  ③ 选择参数优化方法
  →  ④ 勾选输出图表与保存位置  →  ⑤ 运行，查看进度 / 指标 / 图像
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any

# Windows 控制台默认 GBK，强制 UTF-8 避免日志/异常含中文时崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

# 允许 `python app/main.py` 直接运行（把项目根加入 sys.path）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
from nicegui import app as nicegui_app  # noqa: E402
from nicegui import ui  # noqa: E402

from app.core import data as D  # noqa: E402
from app.core.explain import PLOT_OPTIONS  # noqa: E402
from app.core.models import get as get_model, list_models  # noqa: E402
from app.core.optimize import METHOD_FAMILIES, METHODS_BY_FAMILY, OptConfig  # noqa: E402
from app.core.space import Param, clone_space, validate_space  # noqa: E402
from app.core.themes import DEFAULT_SCHEME, list_schemes  # noqa: E402
from app.ui.state import AppConfig, RunState, start_job  # noqa: E402

UPLOAD_DIR = _ROOT / "uploads"
DEFAULT_OUT = _ROOT / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
DEFAULT_OUT.mkdir(exist_ok=True)

cfg = AppConfig()
state = RunState()

DEFAULT_PLOTS = {"regression_fit", "residuals", "importance_native",
                 "importance_perm", "importance_shap", "shap_summary", "results_table"}
IMG_EXT = (".png", ".jpg", ".jpeg")


@ui.page("/")
def index() -> None:  # noqa: C901 —— 单页应用，集中构建
    # 紧凑样式：缩小字号/卡片内边距/控件高度，让全部组件在单屏内可见
    ui.add_head_html(
        "<style>"
        ".cap{font-size:.72rem;color:#666;line-height:1.15}"
        ".sec{font-size:.82rem;font-weight:600}"
        # 显式边框：Quasar 卡片默认靠 box-shadow，会被列的 overflow:hidden 裁掉左右阴影，
        # 导致 Chrome/Edge 下左右边界不可见；改用实线边框（画在元素边缘，不被裁剪）保证一致
        ".tightcard{padding:8px!important;gap:4px!important;"
        "border:1px solid #dfe3e8;box-shadow:0 1px 3px rgba(0,0,0,.06)}"
        ".q-field--dense .q-field__control{min-height:34px}"
        ".pchk .q-checkbox__label{font-size:.72rem;line-height:1}"
        # 压缩上传组件：隐藏"0.0B/0.00%"副标题与文件列表，限制高度
        ".compact-upload .q-uploader__subtitle{display:none}"
        ".compact-upload .q-uploader__list{display:none}"
        ".compact-upload{max-height:54px}"
        ".compact-upload .q-uploader__header{padding:2px 8px}"
        ".result-image{aspect-ratio:4/3;background:#f8fafc}"
        ".result-image .q-img__image{object-fit:contain!important}"
        "</style>"
    )
    ui.query(".nicegui-content").classes("p-0 gap-0")  # 去掉默认外边距，铺满视口

    with ui.row().classes("w-full no-wrap gap-2 p-2").style(
            "height:100vh;box-sizing:border-box"):
        # ============== 左栏：配置（①②③④）+ 数据预览 ============== #
        with ui.column().classes("gap-2 h-full").style(
                "flex:1 1 0;min-width:0;overflow:hidden"):
            ui.label("可解释机器学习 · 训练与解释").classes("text-base font-bold")

            # ① 数据导入与列配置
            with ui.card().classes("w-full tightcard"):
                ui.label("① 数据导入与列配置").classes("sec")
                with ui.row().classes("w-full gap-2 items-center no-wrap"):
                    ui.label("数据表文件路径").classes("text-sm whitespace-nowrap")
                    in_path = ui.input(
                        placeholder="粘贴本地 Excel 路径，或用下方上传").props(
                        "dense clearable").classes("flex-1")
                    in_path.on(
                        "keydown",
                        lambda e: _load_path() if e.args.get("key") == "Enter" else None,
                        args=["key"],
                    )
                ui.upload(label="或拖拽 / 选择 Excel (.xlsx) 上传", auto_upload=True,
                          on_upload=lambda e: _on_upload(e)).props(
                    "accept=.xlsx flat bordered").classes("w-full compact-upload")
                summary_lbl = ui.label("尚未加载数据").classes("cap")
                with ui.row().classes("w-full gap-2 no-wrap"):
                    sel_target = ui.select([], label="目标列").props("dense").classes("flex-1")
                    sel_features = ui.select([], label="特征列", multiple=True).props(
                        "dense").classes("flex-1")
                with ui.row().classes("w-full gap-2 no-wrap"):
                    sel_cat = ui.select([], label="分类列", multiple=True).props(
                        "dense").classes("flex-1")
                    sel_nonstd = ui.select([], label="不标准化列", multiple=True).props(
                        "dense").classes("flex-1")
                with ui.row().classes("gap-2 no-wrap items-center"):
                    in_test = ui.number("测试集比例", value=0.2, min=0.05, max=0.9,
                                        step=0.05).props("dense").classes("w-28")
                    in_seed = ui.number("随机种子", value=42, min=0, step=1).props(
                        "dense").classes("w-28")

            # ②③ 模型 + 优化（并排）
            with ui.row().classes("w-full gap-2 no-wrap items-stretch"):
                with ui.card().classes("tightcard").style("width:40%"):
                    with ui.row().classes("w-full items-center justify-between gap-2 no-wrap"):
                        ui.label("② 模型（多选）").classes("sec")
                        params_btn = ui.button("修改模型参数", icon="tune").props(
                            "flat dense no-caps").classes("text-xs")
                    sel_models = ui.select({k: n for k, n in list_models()}, multiple=True,
                                           value=["xgboost", "random_forest"]).props(
                        "dense").classes("w-full")
                    params_status = ui.label("").classes("cap text-primary")
                with ui.card().classes("tightcard flex-1"):
                    ui.label("③ 参数优化").classes("sec")
                    with ui.row().classes("w-full gap-2 no-wrap"):
                        sel_method_family = ui.select(
                            {k: v for k, v in METHOD_FAMILIES}, value="optuna",
                            label="优化框架").props("dense").classes("flex-1")
                        sel_method = ui.select(
                            {k: v for k, v in METHODS_BY_FAMILY["optuna"]},
                            value="optuna_tpe", label="算法").props(
                            "dense").classes("flex-1")
                    with ui.row().classes("w-full gap-2 no-wrap"):
                        in_trials = ui.number("迭代次数", value=30, min=2, step=1).props(
                            "dense").classes("flex-1")
                        in_cv = ui.number("CV折数", value=5, min=2, max=10, step=1).props(
                            "dense").classes("flex-1")
                        sel_scoring = ui.select({"rmse": "RMSE", "mae": "MAE", "r2": "R²"},
                                                value="rmse", label="评分").props(
                            "dense").classes("flex-1")

            # ④ 输出图表与保存
            with ui.card().classes("w-full tightcard"):
                ui.label("④ 输出图表与保存").classes("sec")
                plot_checks: dict[str, Any] = {}
                with ui.grid(columns=3).classes("w-full gap-x-2 gap-y-0"):
                    for key, label, hint in PLOT_OPTIONS:
                        cb = ui.checkbox(label, value=(key in DEFAULT_PLOTS)).props(
                            "dense").classes("pchk")
                        cb.tooltip(hint)
                        plot_checks[key] = cb
                with ui.row().classes("w-full gap-2 no-wrap items-center"):
                    sel_scheme = ui.select({k: v for k, v in list_schemes()},
                                           value=DEFAULT_SCHEME, label="配色").props(
                        "dense").classes("flex-1")
                    sel_fmt = ui.select(["png", "svg", "pdf"], value="png",
                                        label="格式").props("dense").classes("w-24")
                    in_dpi = ui.number("DPI", value=300, min=72, max=600, step=10).props(
                        "dense").classes("w-24")
                    in_topk = ui.number("TopK", value=6, min=1, max=20, step=1).props(
                        "dense").classes("w-24")
                in_outdir = ui.input("保存目录", value=str(DEFAULT_OUT)).props(
                    "dense").classes("w-full")

            # 数据预览：填满左栏剩余空间（内部滚动），与右栏"结果"区上下对称
            preview_card = ui.card().classes("w-full tightcard").style(
                "flex:1;overflow:auto")
            with preview_card:
                with ui.column().classes("w-full h-full items-center justify-center"):
                    ui.icon("table_chart").classes("text-5xl text-gray-300")
                    ui.label("加载数据后在此预览").classes("cap")

        # ============== 右栏：运行与结果（⑤）+ 结果展示 ============== #
        with ui.column().classes("gap-2 h-full").style(
                "flex:1 1 0;min-width:0;overflow:hidden"):
            with ui.card().classes("w-full tightcard"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    run_btn = ui.button("开始运行", icon="play_arrow").props("dense")
                    open_btn = ui.button("打开输出目录", icon="folder_open",
                                         on_click=lambda: _open_dir(in_outdir.value)).props(
                        "flat dense")
                    stage_lbl = ui.label("").classes("cap")
                progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
                log_view = ui.log(max_lines=500).classes("w-full").style(
                    "height:110px;background:#f6f8fa;color:#24292f;"
                    "border:1px solid #d0d7de;border-radius:4px;"
                    "font-family:ui-monospace,Consolas,monospace;font-size:.72rem")

            # 结果区：高度自适应填满右栏剩余空间，内部滚动（不撑高整页）
            results_card = ui.card().classes("w-full tightcard").style("flex:1;overflow:auto")
            with results_card:
                with ui.column().classes("w-full h-full items-center justify-center"):
                    ui.icon("insights").classes("text-5xl text-gray-300")
                    ui.label("结果将在运行完成后显示").classes("cap")

        # ---------------- 交互逻辑 ---------------- #
        params_dialog = ui.dialog()

        def _sync_params_status() -> None:
            selected = set(sel_models.value or [])
            count = sum(key in selected for key in cfg.model_spaces)
            params_status.text = f"已自定义 {count} 个已选模型" if count else ""

        sel_models.on_value_change(lambda _e: _sync_params_status())

        def _open_model_params() -> None:
            """展示所选模型的搜索范围，并把修改保存在本次页面配置中。"""
            model_keys = list(sel_models.value or [])
            if not model_keys:
                ui.notify("尚未选择模型，请先选择模型", type="warning")
                return

            editors: dict[str, list[tuple[Param, Any, Any, Any]]] = {}
            params_dialog.clear()
            with params_dialog, ui.card().classes("w-full").style(
                    "width:min(900px,92vw);max-height:88vh"):
                ui.label("修改模型参数范围").classes("text-base font-bold")
                ui.label(
                    "数值参数可修改搜索上下限；分类参数请输入 Python 列表格式的候选值。"
                ).classes("cap")
                with ui.scroll_area().classes("w-full").style("height:min(64vh,620px)"):
                    for model_key in model_keys:
                        spec = get_model(model_key)
                        current_space = clone_space(cfg.model_spaces.get(model_key, spec.space))
                        editors[model_key] = []
                        with ui.expansion(spec.name_cn, value=True).classes("w-full"):
                            if not current_space:
                                ui.label("该模型没有可调整的参数范围").classes(
                                    "cap text-gray-500")
                                continue
                            for param in current_space:
                                with ui.row().classes(
                                        "w-full items-center gap-2 no-wrap border-b pb-1"):
                                    ui.label(param.name).classes(
                                        "text-sm font-medium").style("width:180px")
                                    if param.kind in {"float", "int"}:
                                        low = ui.number("下限", value=param.low).props(
                                            "dense").classes("flex-1")
                                        high = ui.number("上限", value=param.high).props(
                                            "dense").classes("flex-1")
                                        if param.kind == "int":
                                            low.props("step=1")
                                            high.props("step=1")
                                        log = ui.checkbox("对数采样", value=param.log).props(
                                            "dense")
                                        if param.kind == "int":
                                            log.disable()
                                        editors[model_key].append((param, low, high, log))
                                    else:
                                        choices = ui.input(
                                            "候选值", value=repr(param.choices)
                                        ).props("dense").classes("flex-1")
                                        editors[model_key].append((param, choices, None, None))

                def _save_model_params() -> None:
                    updated: dict[str, list[Param]] = {}
                    try:
                        for model_key, rows in editors.items():
                            new_space: list[Param] = []
                            for original, first, second, third in rows:
                                if original.kind in {"float", "int"}:
                                    low_value = first.value
                                    high_value = second.value
                                    if low_value is None or high_value is None:
                                        raise ValueError(f"参数 {original.name} 必须填写上下限")
                                    if original.kind == "int":
                                        low_number = float(low_value)
                                        high_number = float(high_value)
                                        if not low_number.is_integer() or not high_number.is_integer():
                                            raise ValueError(
                                                f"整数参数 {original.name} 的上下限必须是整数")
                                        low_value = int(low_number)
                                        high_value = int(high_number)
                                    else:
                                        low_value = float(low_value)
                                        high_value = float(high_value)
                                    new_space.append(Param(
                                        original.name, original.kind,
                                        low=low_value, high=high_value,
                                        log=bool(third.value),
                                    ))
                                else:
                                    parsed = ast.literal_eval((first.value or "").strip())
                                    if not isinstance(parsed, (list, tuple)):
                                        raise ValueError(
                                            f"参数 {original.name} 的候选值必须是列表")
                                    new_space.append(Param(
                                        original.name, "cat", choices=list(parsed)))
                            validate_space(new_space)
                            updated[model_key] = new_space
                    except (SyntaxError, ValueError, TypeError) as exc:
                        ui.notify(f"参数范围有误：{exc}", type="negative")
                        return

                    cfg.model_spaces.update(updated)
                    _sync_params_status()
                    params_dialog.close()
                    ui.notify("模型参数范围已保存，将用于本次运行", type="positive")

                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("取消", on_click=params_dialog.close).props("flat")
                    ui.button("保存", icon="save", on_click=_save_model_params)

            params_dialog.open()

        params_btn.on_click(_open_model_params)

        def _apply_df(df: pd.DataFrame, name: str) -> None:
            cfg.df = df
            cfg.excel_name = name
            cfg.columns = D.column_names(df)
            cols = cfg.columns
            # 默认：目标=最后一列，特征=其余，分类列=object 型
            obj_cols = [c for c in cols if str(df[c].dtype) == "object"]
            target_default = cols[-1]
            feat_default = [c for c in cols if c != target_default]
            sel_target.set_options(cols, value=target_default)
            sel_features.set_options(cols, value=feat_default)
            cat_default = [c for c in obj_cols if c != target_default]
            sel_cat.set_options(feat_default, value=cat_default)
            sel_nonstd.set_options(feat_default, value=cat_default)
            summary_lbl.text = (f"已加载 {name}：{df.shape[0]} 行 × {df.shape[1]} 列，"
                                f"分类列默认 = {cat_default or '无'}")
            # 在左栏底部预览卡片内渲染数据表（前 20 行，内部滚动）
            preview_card.clear()
            with preview_card:
                ui.label(f"{name} · 前 20 行").classes("cap")
                ui.table.from_pandas(df.head(20)).classes("w-full").props("dense")
            ui.notify(f"已加载 {df.shape[0]} 行 × {df.shape[1]} 列", type="positive")

        async def _on_upload(e) -> None:
            try:
                dest = UPLOAD_DIR / e.file.name
                await e.file.save(dest)
                df = pd.read_excel(dest)
            except Exception as exc:  # noqa: BLE001
                ui.notify(f"读取失败：{exc}", type="negative")
                return
            _apply_df(df, e.file.name)

        def _load_path() -> None:
            p = (in_path.value or "").strip().strip('"')
            if not p:
                ui.notify("请输入本地 Excel 路径", type="warning")
                return
            try:
                df = D.load_excel(p)
            except Exception as exc:  # noqa: BLE001
                ui.notify(f"读取失败：{exc}", type="negative")
                return
            _apply_df(df, os.path.basename(p))

        # 目标列变化 → 同步刷新特征候选（排除目标列）
        def _sync_features() -> None:
            if not cfg.columns:
                return
            tgt = sel_target.value
            feats = [c for c in cfg.columns if c != tgt]
            cur = [c for c in (sel_features.value or []) if c in feats]
            sel_features.set_options(feats, value=cur or feats)

        sel_target.on_value_change(lambda _e: _sync_features())

        def _sync_method_options() -> None:
            family = sel_method_family.value or "optuna"
            options = METHODS_BY_FAMILY.get(family, METHODS_BY_FAMILY["optuna"])
            valid = {key for key, _label in options}
            value = sel_method.value if sel_method.value in valid else options[0][0]
            sel_method.set_options({key: label for key, label in options}, value=value)

        sel_method_family.on_value_change(lambda _e: _sync_method_options())

        # pushed 记录已推送的日志条数与结果是否已渲染，避免重复
        pushed = {"n": 0, "rendered": False}

        def _poll() -> None:
            """定时器回调（主线程）：把后台线程写入 RunState 的进展同步到界面。

            训练在子线程，不能直接操作 UI；故子线程只更新 state，由本回调读取并渲染，
            实现安全的跨线程更新。
            """
            # 增量推送新产生的日志行
            while pushed["n"] < len(state.logs):
                log_view.push(state.logs[pushed["n"]])
                pushed["n"] += 1
            progress.value = state.progress
            stage_lbl.text = state.stage
            if state.running:
                run_btn.disable()  # 运行中禁用按钮，防止重复触发
            # 任务结束后只渲染一次结果
            if state.done and not pushed["rendered"]:
                pushed["rendered"] = True
                run_btn.enable()
                _render_results()
                if state.error:
                    ui.notify("运行出错，详见日志", type="negative")
                else:
                    ui.notify("运行完成 ✅", type="positive")

        def _render_results() -> None:
            results_card.clear()
            with results_card:
                ui.label("结果").classes("sec")
                if state.metrics_rows:
                    ui.label("指标对比（测试集）").classes("cap")
                    ui.table.from_pandas(pd.DataFrame(state.metrics_rows)).classes(
                        "w-full").props("dense")
                if state.gallery:
                    by_model: dict[str, list] = {}
                    for item in state.gallery:
                        by_model.setdefault(item["model"], []).append(item)
                    for model_name, items in by_model.items():
                        with ui.expansion(f"{model_name}（{len(items)} 项）",
                                          value=True).classes("w-full"):
                            with ui.grid(columns=2).classes("w-full gap-2"):
                                for it in items:
                                    _render_item(it)

        def _render_item(it: dict) -> None:
            # 单个输出卡片：图片直接预览，其它文件（如 xlsx）显示文件图标；均可下载
            path = it["path"]
            with ui.column().classes("w-full items-center gap-1 border rounded p-1"):
                if path.lower().endswith(IMG_EXT):
                    ui.image(Path(path)).props("fit=contain").classes("w-full result-image")
                else:
                    ui.icon("description").classes("text-4xl text-gray-500")
                ui.label(it["label"]).classes("cap text-center")
                ui.button("下载", icon="download",
                          on_click=lambda p=path: ui.download(
                              Path(p).read_bytes(), filename=Path(p).name)
                          ).props("flat dense")

        def _start() -> None:
            if cfg.df is None:
                ui.notify("请先上传 Excel 数据", type="warning")
                return
            models = list(sel_models.value or [])
            if not models:
                ui.notify("请至少选择一个模型", type="warning")
                return
            plots_sel = [k for k, cb in plot_checks.items() if cb.value]
            if not plots_sel:
                ui.notify("请至少勾选一种输出", type="warning")
                return
            try:
                data_cfg = D.DataConfig(
                    target=sel_target.value,
                    features=list(sel_features.value or []),
                    categorical=list(sel_cat.value or []),
                    non_standardize=list(sel_nonstd.value or []),
                    test_size=float(in_test.value), random_state=int(in_seed.value))
                data_cfg.validate(cfg.df)
            except Exception as exc:  # noqa: BLE001
                ui.notify(f"列配置有误：{exc}", type="negative")
                return
            opt_cfg = OptConfig(method=sel_method.value, n_trials=int(in_trials.value),
                                cv_folds=int(in_cv.value), scoring=sel_scoring.value,
                                random_state=int(in_seed.value))
            out_dir = in_outdir.value or str(DEFAULT_OUT)
            os.makedirs(out_dir, exist_ok=True)
            state.reset()
            pushed["n"] = 0
            pushed["rendered"] = False
            log_view.clear()
            results_card.clear()
            with results_card:
                ui.label("运行中，请稍候…").classes("cap")
            ui.notify(f"开始训练 {len(models)} 个模型…", type="info")
            start_job(cfg.df, data_cfg, models, opt_cfg,
                      {key: clone_space(space) for key, space in cfg.model_spaces.items()},
                      plots_sel, out_dir,
                      sel_fmt.value, int(in_dpi.value), int(in_topk.value),
                      sel_scheme.value, state)

        run_btn.on_click(_start)
        ui.timer(0.4, _poll)


def _open_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        os.startfile(path)  # type: ignore[attr-defined]  # Windows
    except Exception as exc:  # noqa: BLE001
        ui.notify(f"无法打开目录：{exc}", type="warning")


if __name__ in {"__main__", "__mp_main__"}:
    nicegui_app.add_media_files("/uploads", str(UPLOAD_DIR))
    ui.run(title="可解释机器学习", port=8080, reload=False, show=True)
