"""绘图层：由参考项目 ``tools.py`` 重构而来。

相对参考实现的变化：
- 统一使用 ``Agg`` 后端（服务端渲染，存文件后在 NiceGUI 展示）；
- 中文字体从 ``app/assets/times+simsun.ttf`` 加载；
- 所有函数模型无关、参数化（保存路径 / 标题 / dpi），返回保存路径；
- ``regression_fit`` 的多面板布局参考自 ``tools.py`` 的 ``plot_regression_fit2``。

SHAP 系列绘图在 :mod:`app.core.explain` 中处理（依赖 shap 对象）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 关键：必须在 import pyplot 之前设置为非交互后端

import matplotlib.gridspec as gridspec  # noqa: E402  多面板拼图用
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# --------------------------------------------------------------------------- #
# 中文字体加载（位于 app/assets，下同一份字体同时含 Times New Roman 与 SimSun）
# --------------------------------------------------------------------------- #
_FONT_PATH = Path(__file__).resolve().parents[1] / "assets" / "times+simsun.ttf"
_FONT_NAME = "sans-serif"
if _FONT_PATH.exists():
    try:
        font_manager.fontManager.addfont(str(_FONT_PATH))  # 注册字体文件
        _FONT_NAME = font_manager.FontProperties(fname=str(_FONT_PATH)).get_name()
    except Exception:  # pragma: no cover —— 字体异常时退回默认，不影响出图
        pass
# 让导出的 PDF/PS 以 TrueType 嵌入字体，避免论文排版乱码
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def _use_font(size: int = 12) -> None:
    """应用中文字体到全局 rcParams。

    注意：``plt.style.use(...)`` 会重置 rcParams（把字体改回 Arial），
    因此本函数必须在任何 ``style.use`` **之后**调用。
    """
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [_FONT_NAME, "DejaVu Sans", "Arial"]
    matplotlib.rcParams["font.size"] = size
    matplotlib.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
# 1. 回归拟合图（真实 vs 预测）—— 多面板版，参考 tools.plot_regression_fit2
# --------------------------------------------------------------------------- #
def regression_fit(y_true, y_pred, metrics: dict, title: str, save_path: str,
                   dpi: int = 300, color: str = "#C15060",
                   point_label: str = "训练集") -> str:
    """绘制"真实值 vs 预测值"主图，并在四周拼接边际直方图与残差面板。

    布局（2×3 GridSpec，5 个子图）参考参考项目 ``plot_regression_fit2``：
      ┌────────────┬───────────────────────┐
      │ 残差直方图  │  真实值边际直方图        │   ← 上排
      ├────────────┼───────────────────────┼──────────┐
      │ 残差散点    │  主图(真实 vs 预测)      │ 预测值   │   ← 下排
      │ (resid×pred)│  +1:1线 +±20%带 +指标   │ 边际直方 │
      └────────────┴───────────────────────┴──────────┘

    参数
    ----
    color       散点/直方图配色（训练集与测试集传入不同颜色以区分）。
    point_label 主图散点的图例文字（如"训练集"/"测试集"）。
    """
    plt.style.use("default")  # 先回到 matplotlib 默认，避免上一张图的 seaborn 样式残留
    _use_font(14)             # 再应用中文字体（顺序不可颠倒）

    c_point = color           # 散点与直方图颜色
    c_line = "#3C5488"        # 1:1 参考线颜色（Nature 蓝）
    c_band = color            # ±20% 误差带颜色

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # 坐标范围：下限取 0 与数据最小值的较小者，上限在数据最大值上再留 1/6 余量
    min_val = float(min(0.0, y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    max_val += (max_val - min_val) / 6 or 1.0

    # 画布与 2×3 网格：列宽=[残差区, 主图, 右直方], 行高=[上直方, 主图]
    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    gs = gridspec.GridSpec(2, 3, width_ratios=[0.6, 3, 0.5],
                           height_ratios=[0.5, 3], wspace=0.4, hspace=0.1)
    ax_main = fig.add_subplot(gs[1, 1])                      # 主散点图
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)       # 上：真实值直方图
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)     # 右：预测值直方图
    ax_left = fig.add_subplot(gs[1, 0], sharey=ax_main)      # 左：残差散点
    ax_top_left = fig.add_subplot(gs[0, 0], sharex=ax_left)  # 左上：残差直方图

    # —— 主图：真实值 vs 预测值散点 ——
    ax_main.scatter(y_true, y_pred, c=c_point, s=60, alpha=0.9,
                    edgecolors="grey", linewidth=0.5, label=point_label)
    # 1:1 完美预测参考线
    ax_main.plot([min_val, max_val], [min_val, max_val],
                 c=c_line, linestyle="--", linewidth=1.5)
    # ±20% 误差带（y=x*0.8 ~ y=x*1.2 之间填充）
    x_line = np.linspace(min_val, max_val, 100)
    ax_main.fill_between(x_line, x_line * 0.8, x_line * 1.2,
                         color=c_band, alpha=0.25, label="±20%")
    # 左上角指标文本框
    txt = (f"$R^2$ = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.3f}\n"
           f"MAE = {metrics['MAE']:.3f}")
    ax_main.text(0.05, 0.95, txt, transform=ax_main.transAxes, fontsize=15,
                 va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))
    ax_main.set_xlim([min_val, max_val])
    ax_main.set_ylim([min_val, max_val])
    ax_main.tick_params(axis="both", which="both", direction="in",
                        top=False, right=False, labelsize=13)
    ax_main.minorticks_off()
    ax_main.set_xlabel("真实值 (True)", fontsize=15)
    ax_main.set_ylabel("预测值 (Predicted)", fontsize=15)
    ax_main.legend(fontsize=13, loc="lower right")

    bins = 15
    # —— 上方边际：真实值分布 ——
    ax_top.hist(y_true, bins=bins, color=c_point, edgecolor="grey", alpha=0.8)
    ax_top.tick_params(axis="both", which="both", direction="in",
                       top=False, right=False, labelsize=12)
    ax_top.minorticks_off()
    ax_top.set_ylabel("计数", fontsize=14)
    # —— 右方边际：预测值分布（横向直方图，与主图共享 y 轴）——
    ax_right.hist(y_pred, bins=bins, orientation="horizontal",
                  color=c_point, edgecolor="grey", alpha=0.8)
    ax_right.tick_params(axis="both", which="both", direction="in",
                         top=False, right=False, labelsize=12)
    ax_right.minorticks_off()
    ax_right.set_xlabel("计数", fontsize=14)
    # —— 左方：残差(真实−预测) vs 预测值（与主图共享 y 轴）——
    resid = y_true - y_pred
    ax_left.scatter(resid, y_pred, c=c_point, s=40, alpha=0.8,
                    edgecolors="grey", linewidth=0.5)
    ax_left.axvline(0, color=c_line, linestyle="--", linewidth=1.2)  # 零残差线
    ax_left.tick_params(axis="both", which="both", direction="in",
                        top=False, right=False, labelsize=12)
    ax_left.minorticks_off()
    ax_left.set_xlabel("残差", fontsize=14)
    # —— 左上：残差分布直方图（与左方残差散点共享 x 轴）——
    ax_top_left.hist(resid, bins=10, color=c_point, edgecolor="grey", alpha=0.8)
    ax_top_left.tick_params(axis="both", which="both", direction="in",
                            top=False, right=False, labelsize=12)
    ax_top_left.minorticks_off()
    ax_top_left.set_ylabel("计数", fontsize=14)

    # 统一给 5 个子图描黑色边框
    for ax in (ax_main, ax_top, ax_right, ax_left, ax_top_left):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

    fig.suptitle(title, fontsize=16, y=0.96)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- #
# 2. 特征重要性条形图（水平条形 + 数值标注）
# --------------------------------------------------------------------------- #
def importance_bar(df_imp: pd.DataFrame, title: str, save_path: str,
                   bar_color: str = "dodgerblue", dpi: int = 300) -> str:
    """绘制特征重要性水平条形图。``df_imp`` 需含 'Feature' 与 'Importance' 两列。"""
    _use_font(14)
    d = df_imp.sort_values("Importance", ascending=True)  # 升序，使最重要的在最上方
    # 高度随特征数自适应，避免特征多时拥挤
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(d) + 2)))
    bars = ax.barh(d["Feature"].astype(str), d["Importance"],
                   color=bar_color, alpha=0.85)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlabel("重要性得分", fontsize=13)
    ax.set_ylabel("特征", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    # 在每个条形末端标注具体数值
    for b in bars:
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f" {b.get_width():.4f}", va="center", ha="left", fontsize=11)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.18)  # 右侧留白给数值标签
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- #
# 3. 残差分析图（残差 vs 预测值，高亮异常值）
# --------------------------------------------------------------------------- #
def residuals(resid, y_pred, title: str, save_path: str, dpi: int = 300,
              point_color: str = "seagreen") -> str:
    """绘制残差散点：以 ±2 倍标准差为界把点分为正常值与异常值。

    ``point_color`` 控制正常值配色；异常值固定红色以保留"告警"语义。
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    _use_font(12)  # 注意顺序：先 style 后 font
    resid = np.asarray(resid, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sd = float(np.std(resid)) or 1.0
    outlier = np.abs(resid) > 2 * sd  # |残差| > 2SD 视为异常值
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axhspan(-sd, sd, color="gold", alpha=0.25, label=f"±1 S.D. (±{sd:.2f})")
    ax.scatter(y_pred[~outlier], resid[~outlier], alpha=0.6, c=point_color,
               edgecolors="k", linewidth=0.5, s=50, label="正常值")
    if outlier.any():
        ax.scatter(y_pred[outlier], resid[outlier], alpha=0.85, c="red",
                   edgecolors="k", linewidth=0.5, s=70, label="异常值 (>2 S.D.)")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)  # 零残差基准线
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel("预测值 (Predicted)", fontsize=13)
    ax.set_ylabel("残差 (真实 − 预测)", fontsize=13)
    ymax = float(np.max(np.abs(resid))) * 1.2 or 1.0
    ax.set_ylim(-ymax, ymax)  # 对称 y 轴，零线居中
    ax.legend(loc="upper right", fontsize=11)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- #
# 4. PDP / ICE（一维，含 95% 置信区间）
# --------------------------------------------------------------------------- #
def _manual_pdp_1d(model, X: pd.DataFrame, feature: str, resolution: int = 50):
    """手动计算一维 PDP 与 ICE。

    思路：把目标特征在 [min, max] 上取 ``resolution`` 个网格点；对每个样本，
    将该特征依次替换为各网格值后预测，得到一条 ICE 曲线；所有 ICE 的均值即 PDP。
    """
    grid = np.linspace(X[feature].min(), X[feature].max(), resolution)
    ice = np.zeros((len(X), resolution))  # 每行一条 ICE 曲线
    for i, (_, row) in enumerate(X.iterrows()):
        tmp = pd.DataFrame([row] * resolution)  # 复制该样本 resolution 份
        tmp[feature] = grid                      # 仅改变目标特征
        ice[i, :] = model.predict(tmp[X.columns])
    return grid, ice.mean(axis=0), ice


def pdp_ice_1d(model, X: pd.DataFrame, feature: str, save_path: str,
               dpi: int = 300, line_color: str = "red",
               ice_color: str = "tab:blue") -> str:
    """绘制单变量 PDP（均值虚线，含 95% 置信区间）叠加所有样本的 ICE 细线。

    ``line_color`` 为 PDP 均值线颜色，``ice_color`` 为 ICE 细线与置信带颜色。
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    _use_font(12)
    grid, pdp, ice = _manual_pdp_1d(model, X, feature)
    std = np.std(ice, axis=0)  # 每个网格点上 ICE 的标准差，用于置信区间
    fig, ax = plt.subplots(figsize=(9, 7))
    for line in ice:  # 所有样本的 ICE 曲线（半透明）
        ax.plot(grid, line, color=ice_color, alpha=0.05, linewidth=0.5)
    ax.plot(grid, pdp, color=line_color, linestyle="--", linewidth=3,
            label="平均效应 (PDP)")
    ax.fill_between(grid, pdp - 1.96 * std, pdp + 1.96 * std,  # 均值 ±1.96·SD
                    color=ice_color, alpha=0.25, label="95% 置信区间")
    ax.set_title(f"PDP / ICE：{feature}", fontsize=15)
    ax.set_xlabel(f"{feature}", fontsize=13)
    ax.set_ylabel("对预测值的依赖", fontsize=13)
    ax.legend()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- #
# 5. 2D PDP 热力图（两特征联合部分依赖）
# --------------------------------------------------------------------------- #
def _manual_pdp_2d(model, X, f1, f2, resolution: int = 20):
    """手动计算二维 PDP：在 (f1, f2) 的网格上，把两特征同时固定为网格值后取预测均值。"""
    g1 = np.linspace(X[f1].min(), X[f1].max(), resolution)
    g2 = np.linspace(X[f2].min(), X[f2].max(), resolution)
    z = np.zeros((resolution, resolution))
    for i, v1 in enumerate(g1):
        for j, v2 in enumerate(g2):
            tmp = X.copy()
            tmp[f1] = v1  # 整列固定为当前网格值
            tmp[f2] = v2
            z[j, i] = float(np.mean(model.predict(tmp[X.columns])))
    return g1, g2, z


def pdp_2d(model, X: pd.DataFrame, f1: str, f2: str, save_path: str,
           dpi: int = 300, cmap: str = "viridis") -> str:
    """绘制两特征联合 PDP 的等值线热力图（``cmap`` 由配色方案指定）。"""
    _use_font(12)
    g1, g2, z = _manual_pdp_2d(model, X, f1, f2)
    XX, YY = np.meshgrid(g1, g2)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    c = ax.contourf(XX, YY, z, cmap=cmap, levels=20)
    fig.colorbar(c, ax=ax, label="部分依赖值 (PDP)")
    ax.set_title(f"2D PDP：{f1} × {f2}", fontsize=15)
    ax.set_xlabel(f1, fontsize=13)
    ax.set_ylabel(f2, fontsize=13)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- #
# 6. ALE（一维，累积局部效应）—— 使用 PyALE
# --------------------------------------------------------------------------- #
def ale_1d(model, X: pd.DataFrame, feature: str, save_path: str,
           dpi: int = 300, color: str = "#2c7fb8") -> str:
    """绘制一维 ALE 曲线（含置信区间），曲线/填充颜色由 ``color`` 指定。"""
    from PyALE import ale

    _use_font(12)
    # PyALE 直接在当前 figure 上作图，返回效应数据
    ale(X=X, model=model, feature=[feature], feature_type="continuous",
        grid_size=50, include_CI=True, C=0.95)
    fig, ax = plt.gcf(), plt.gca()
    if ax.lines:  # 主曲线改色加粗
        ax.lines[0].set_color(color)
        ax.lines[0].set_linewidth(2.5)
    if ax.collections:  # 置信区间填充改色
        ax.collections[0].set_facecolor(color)
        ax.collections[0].set_alpha(0.2)
    ax.set_title(f"累积局部效应 (ALE)：{feature}", fontsize=15)
    ax.set_xlabel(feature, fontsize=13)
    ax.set_ylabel("ALE（对预测的影响）", fontsize=13)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    return save_path
