"""可解释性编排：按用户勾选的图表类型与模型能力，分派生成输出文件。

每个输出独立 try/except，单项失败不影响其余。返回 ``[(标签, 文件路径), ...]``。
"""

from __future__ import annotations

import os
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import plots  # noqa: E402
from . import themes  # noqa: E402
from .pipeline import TrainResult  # noqa: E402

# UI 可勾选的图表：(key, 中文标签, 说明/限制)
PLOT_OPTIONS: list[tuple[str, str, str]] = [
    ("regression_fit", "回归拟合图（训练/测试）", "真实 vs 预测散点 + 指标"),
    ("residuals", "残差分析图", "残差分布与异常值"),
    ("importance_native", "原生特征重要性", "仅树模型"),
    ("importance_perm", "置换重要性", "所有模型"),
    ("importance_shap", "SHAP 特征重要性", "平均 |SHAP|"),
    ("shap_summary", "SHAP 摘要散点图", "特征影响概览"),
    ("shap_dependence", "SHAP 依赖图", "Top-K 特征"),
    ("shap_waterfall", "SHAP 瀑布图", "单样本解释"),
    ("pdp_1d", "PDP / ICE（一维）", "Top-K 特征"),
    ("pdp_2d", "2D PDP 热力图", "Top-2 特征对"),
    ("ale_1d", "ALE（一维）", "Top-K 特征，连续变量"),
    ("results_table", "导出预测结果表 (xlsx)", "训练/测试预测值"),
]

_SHAP_PLOTS = {"importance_shap", "shap_summary", "shap_dependence", "shap_waterfall"}


def _ext(fmt: str) -> str:
    return fmt.lstrip(".").lower()


def _importances(result: TrainResult) -> pd.DataFrame:
    """优先用原生重要性，否则置换重要性，作为 Top-K 特征排序依据。"""
    feats = result.feature_names
    if result.spec.has_native_importance and hasattr(result.model, "feature_importances_"):
        vals = np.asarray(result.model.feature_importances_, dtype=float)
    else:
        from sklearn.inspection import permutation_importance

        r = permutation_importance(result.model, result.prepared.X_test,
                                   result.prepared.y_test, n_repeats=8,
                                   random_state=42)
        vals = r.importances_mean
    return pd.DataFrame({"Feature": feats, "Importance": vals}).sort_values(
        "Importance", ascending=False).reset_index(drop=True)


def _compute_shap(result: TrainResult, log):
    """返回 (shap_values[ndarray], X_used[DataFrame], base_value[float])。"""
    import shap

    model, X_train, X_test = result.model, result.prepared.X_train, result.prepared.X_test
    if result.spec.shap_kind == "tree":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        base = float(np.ravel(explainer.expected_value)[0])
        return np.asarray(sv), X_test, base
    # 非树模型：KernelExplainer（慢）→ 背景/样本下采样以控制耗时
    bg = shap.sample(X_train, min(50, len(X_train)), random_state=42)
    n = min(100, len(X_test))
    X_used = X_test.iloc[:n]
    if log:
        log(f"    KernelExplainer 计算 SHAP（背景{len(bg)}/样本{n}），可能较慢…")
    explainer = shap.KernelExplainer(model.predict, bg)
    sv = explainer.shap_values(X_used, nsamples=100, silent=True)
    base = float(np.ravel(explainer.expected_value)[0])
    return np.asarray(sv), X_used, base


def generate(
    result: TrainResult,
    selected: list[str],
    out_dir: str,
    fmt: str = "png",
    dpi: int = 300,
    top_k: int = 6,
    scheme_key: str = themes.DEFAULT_SCHEME,
    log: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    _log = log or (lambda _m: None)
    ext = _ext(fmt)
    sch = themes.get(scheme_key)  # 当前配色方案：离散色 + 连续 colormap
    model_dir = os.path.join(out_dir, result.model_key)
    os.makedirs(model_dir, exist_ok=True)
    outputs: list[tuple[str, str]] = []

    def _path(name: str) -> str:
        return os.path.join(model_dir, f"{result.model_key}_{name}.{ext}")

    sel = set(selected)
    pr = result.prepared
    imp_df = None
    try:
        imp_df = _importances(result)
        top_feats = imp_df["Feature"].tolist()[:top_k]
    except Exception as exc:  # noqa: BLE001
        _log(f"    [警告] 计算特征重要性失败：{exc}")
        top_feats = result.feature_names[:top_k]

    # --- 回归拟合图（训练/测试分别出图，用不同配色区分）---
    if "regression_fit" in sel:
        for tag, yt, yp, m, color, lbl in [
            ("train", pr.y_train, result.y_train_pred, result.train_metrics,
             sch.train, "训练集"),
            ("test", pr.y_test, result.y_test_pred, result.test_metrics,
             sch.test, "测试集"),
        ]:
            try:
                p = plots.regression_fit(
                    yt, yp, m, f"{result.model_name} 拟合（{tag}）",
                    _path(f"regression_fit_{tag}"), dpi, color, lbl)
                outputs.append((f"回归拟合图-{tag}", p))
            except Exception as exc:  # noqa: BLE001
                _log(f"    [警告] 回归拟合图({tag})失败：{exc}")

    # --- 残差分析图 ---
    if "residuals" in sel:
        for tag, yt, yp in [("train", pr.y_train, result.y_train_pred),
                            ("test", pr.y_test, result.y_test_pred)]:
            try:
                resid = np.asarray(yt) - np.asarray(yp)
                p = plots.residuals(resid, yp, f"{result.model_name} 残差（{tag}）",
                                    _path(f"residuals_{tag}"), dpi,
                                    point_color=sch.train)
                outputs.append((f"残差分析图-{tag}", p))
            except Exception as exc:  # noqa: BLE001
                _log(f"    [警告] 残差图({tag})失败：{exc}")

    # --- 原生特征重要性 ---
    if "importance_native" in sel:
        if result.spec.has_native_importance and hasattr(result.model, "feature_importances_"):
            try:
                d = pd.DataFrame({"Feature": result.feature_names,
                                  "Importance": result.model.feature_importances_})
                p = plots.importance_bar(d, f"{result.model_name} 原生特征重要性",
                                         _path("importance_native"), sch.bar, dpi)
                outputs.append(("原生特征重要性", p))
            except Exception as exc:  # noqa: BLE001
                _log(f"    [警告] 原生重要性失败：{exc}")
        else:
            _log(f"    [跳过] {result.model_name} 无原生特征重要性")

    # --- 置换重要性 ---
    if "importance_perm" in sel:
        try:
            from sklearn.inspection import permutation_importance

            r = permutation_importance(result.model, pr.X_test, pr.y_test,
                                       n_repeats=10, random_state=42)
            d = pd.DataFrame({"Feature": result.feature_names,
                              "Importance": r.importances_mean})
            p = plots.importance_bar(d, f"{result.model_name} 置换重要性",
                                     _path("importance_permutation"), sch.bar2, dpi)
            outputs.append(("置换重要性", p))
        except Exception as exc:  # noqa: BLE001
            _log(f"    [警告] 置换重要性失败：{exc}")

    # --- SHAP 系列 ---
    if sel & _SHAP_PLOTS:
        try:
            import shap

            sv, X_used, base = _compute_shap(result, _log)

            if "importance_shap" in sel:
                d = pd.DataFrame({"Feature": list(X_used.columns),
                                  "Importance": np.abs(sv).mean(axis=0)})
                p = plots.importance_bar(d, f"{result.model_name} SHAP 重要性",
                                         _path("importance_shap"), sch.bar3, dpi)
                outputs.append(("SHAP 特征重要性", p))

            if "shap_summary" in sel:
                plt.figure()
                shap.summary_plot(sv, X_used, show=False, cmap=plt.get_cmap(sch.cmap))
                plt.title(f"{result.model_name} SHAP 摘要", fontsize=14)
                plt.tight_layout()
                p = _path("shap_summary")
                plt.savefig(p, dpi=dpi, bbox_inches="tight")
                plt.close("all")
                outputs.append(("SHAP 摘要散点图", p))

            if "shap_dependence" in sel:
                names = list(X_used.columns)
                for feat in top_feats:
                    if feat not in names:
                        continue
                    try:
                        shap.dependence_plot(feat, sv, X_used,
                                             interaction_index="auto", show=False,
                                             cmap=plt.get_cmap(sch.cmap))
                        plt.tight_layout()
                        p = _path(f"shap_dependence_{feat}")
                        plt.savefig(p, dpi=dpi, bbox_inches="tight")
                        plt.close("all")
                        outputs.append((f"SHAP 依赖图-{feat}", p))
                    except Exception as exc:  # noqa: BLE001
                        _log(f"    [警告] SHAP 依赖图({feat})失败：{exc}")

            if "shap_waterfall" in sel:
                try:
                    expl = shap.Explanation(values=sv[0], base_values=base,
                                            data=X_used.iloc[0].values,
                                            feature_names=list(X_used.columns))
                    plt.figure()
                    shap.plots.waterfall(expl, max_display=15, show=False)
                    plt.tight_layout()
                    p = _path("shap_waterfall_sample0")
                    plt.savefig(p, dpi=dpi, bbox_inches="tight")
                    plt.close("all")
                    outputs.append(("SHAP 瀑布图(样本0)", p))
                except Exception as exc:  # noqa: BLE001
                    _log(f"    [警告] SHAP 瀑布图失败：{exc}")
        except Exception as exc:  # noqa: BLE001
            _log(f"    [警告] SHAP 计算失败：{exc}")

    # --- PDP / ICE 一维 ---
    if "pdp_1d" in sel:
        for feat in top_feats:
            try:
                p = plots.pdp_ice_1d(result.model, pr.X_train, feat,
                                     _path(f"pdp_ice_{feat}"), dpi,
                                     line_color=sch.accent, ice_color=sch.train)
                outputs.append((f"PDP/ICE-{feat}", p))
            except Exception as exc:  # noqa: BLE001
                _log(f"    [警告] PDP/ICE({feat})失败：{exc}")

    # --- 2D PDP ---
    if "pdp_2d" in sel and len(top_feats) >= 2:
        f1, f2 = top_feats[0], top_feats[1]
        try:
            p = plots.pdp_2d(result.model, pr.X_train, f1, f2,
                             _path(f"pdp2d_{f1}_{f2}"), dpi, cmap=sch.cmap)
            outputs.append((f"2D PDP-{f1}×{f2}", p))
        except Exception as exc:  # noqa: BLE001
            _log(f"    [警告] 2D PDP 失败：{exc}")

    # --- ALE 一维 ---
    if "ale_1d" in sel:
        for feat in top_feats:
            try:
                p = plots.ale_1d(result.model, pr.X_train, feat,
                                 _path(f"ale_{feat}"), dpi, color=sch.test)
                outputs.append((f"ALE-{feat}", p))
            except Exception as exc:  # noqa: BLE001
                _log(f"    [警告] ALE({feat})失败：{exc}")

    # --- 导出预测结果表 ---
    if "results_table" in sel:
        try:
            p = os.path.join(model_dir, f"{result.model_key}_results.xlsx")
            with pd.ExcelWriter(p) as w:
                tr = pr.X_train.copy()
                tr["y_true"] = pr.y_train.values
                tr["y_pred"] = result.y_train_pred.values
                tr.to_excel(w, sheet_name="train", index=False)
                te = pr.X_test.copy()
                te["y_true"] = pr.y_test.values
                te["y_pred"] = result.y_test_pred.values
                te.to_excel(w, sheet_name="test", index=False)
            outputs.append(("预测结果表(xlsx)", p))
        except Exception as exc:  # noqa: BLE001
            _log(f"    [警告] 导出结果表失败：{exc}")

    return outputs
