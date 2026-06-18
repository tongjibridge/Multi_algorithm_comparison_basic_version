"""期刊级配色方案。

色值参考 nature-figure 技能的 PALETTE / NMI pastel / Nature Material / Clinical 等色族。
每组方案同时给出：
- 离散色（训练集 / 测试集散点、三类特征重要性条形、强调线）；
- 连续 colormap（SHAP / PDP / 热力图）。

在界面"④ 输出图表与保存"中选择，经 explain.generate 应用到各图。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorScheme:
    key: str
    name_cn: str
    train: str       # 训练集散点
    test: str        # 测试集散点
    bar: str         # 原生特征重要性条形
    bar2: str        # 置换重要性条形
    bar3: str        # SHAP 重要性条形
    accent: str      # 强调线（1:1 线、零残差线、PDP 均值线）
    cmap: str        # 连续色图（SHAP / PDP / 热力图）


# 5 组方案，色值均取自 nature-figure 技能的命名色族
SCHEMES: dict[str, ColorScheme] = {
    s.key: s for s in [
        # Nature 经典：blue_main / red_strong / blue_secondary …（PALETTE）
        ColorScheme("nature_classic", "Nature 经典蓝红",
                    train="#0F4D92", test="#B64342", bar="#3775BA",
                    bar2="#E9A6A1", bar3="#42949E", accent="#4D4D4D",
                    cmap="viridis"),
        # 暖冷对比：沿用初版拟合图配色（C15060 / 34768C）
        ColorScheme("nature_warmcool", "Nature 暖冷对比",
                    train="#C15060", test="#34768C", bar="#3775BA",
                    bar2="#C15060", bar3="#42949E", accent="#3C5488",
                    cmap="magma"),
        # NMI 低饱和：baseline 蓝紫族（PALETTE_NMI_PASTEL），统一家族观感
        ColorScheme("nmi_pastel", "NMI 低饱和蓝紫",
                    train="#484878", test="#7884B4", bar="#7884B4",
                    bar2="#E4CCD8", bar3="#B4C0E4", accent="#606060",
                    cmap="BuPu"),
        # 材料/物理风：teal / violet / aqua / lilac（PALETTE_NATURE_MATERIAL）
        ColorScheme("material_teal", "材料青紫",
                    train="#33B5A5", test="#7C6CCF", bar="#77D7D1",
                    bar2="#B9A7E8", bar3="#33B5A5", accent="#E53935",
                    cmap="cividis"),
        # 临床多色：week/year 多色（PALETTE_NATURE_CLINICAL）
        ColorScheme("clinical", "临床多色",
                    train="#5B8FD6", test="#D24B40", bar="#7BAA5B",
                    bar2="#E28E2C", bar3="#5B8FD6", accent="#272727",
                    cmap="magma"),
    ]
}

DEFAULT_SCHEME = "nature_classic"


def list_schemes() -> list[tuple[str, str]]:
    """返回 [(key, 中文名), ...] 供 UI 下拉选择。"""
    return [(k, s.name_cn) for k, s in SCHEMES.items()]


def get(key: str) -> ColorScheme:
    return SCHEMES.get(key, SCHEMES[DEFAULT_SCHEME])
