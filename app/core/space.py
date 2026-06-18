"""声明式超参数搜索空间。

用一份 ``list[Param]`` 同时派生：
- Optuna 的 ``suggest_*``；
- GridSearch / RandomizedSearch 的离散网格；
- mealpy（PSO 等元启发式）的 ``Var`` 边界与解码。

这样新增模型只需写一次空间，所有优化方法即可复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Param:
    name: str
    kind: str                     # "float" | "int" | "cat"
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[Any] | None = None


# --------------------------------------------------------------------------- #
# Optuna
# --------------------------------------------------------------------------- #
def optuna_suggest(trial, space: list[Param]) -> dict:
    out: dict[str, Any] = {}
    for p in space:
        if p.kind == "float":
            out[p.name] = trial.suggest_float(p.name, p.low, p.high, log=p.log)
        elif p.kind == "int":
            out[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high))
        elif p.kind == "cat":
            out[p.name] = trial.suggest_categorical(p.name, p.choices)
    return out


# --------------------------------------------------------------------------- #
# Grid / Random
# --------------------------------------------------------------------------- #
def grid_space(space: list[Param], n_levels: int = 3) -> dict:
    grid: dict[str, list] = {}
    for p in space:
        if p.kind == "cat":
            grid[p.name] = list(p.choices)
        elif p.kind == "int":
            vals = np.linspace(p.low, p.high, n_levels)
            grid[p.name] = sorted({int(round(v)) for v in vals})
        elif p.kind == "float":
            if p.log:
                vals = np.geomspace(p.low, p.high, n_levels)
            else:
                vals = np.linspace(p.low, p.high, n_levels)
            grid[p.name] = [float(v) for v in vals]
    return grid


# --------------------------------------------------------------------------- #
# mealpy（PSO 等）
# --------------------------------------------------------------------------- #
def mealpy_bounds(space: list[Param]):
    """构造 mealpy 的边界列表。失败时由调用方回退到其他方法。"""
    from mealpy import CategoricalVar, FloatVar, IntegerVar

    bounds = []
    for p in space:
        if p.kind == "float":
            bounds.append(FloatVar(lb=p.low, ub=p.high, name=p.name))
        elif p.kind == "int":
            bounds.append(IntegerVar(lb=int(p.low), ub=int(p.high), name=p.name))
        elif p.kind == "cat":
            bounds.append(CategoricalVar(valid_sets=list(p.choices), name=p.name))
    return bounds
