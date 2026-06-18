"""模型注册表：统一封装 9 个回归模型（一期）。

每个模型用 ``ModelSpec`` 描述，超参数搜索空间用声明式 ``list[Param]`` 表达，
由 :mod:`app.core.space` 派生出 Optuna / Grid / mealpy 三种优化方法所需的形式。

搜索空间取自参考项目各算法文件的 ``objective`` 函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .space import Param

RANDOM_STATE = 42


@dataclass
class ModelSpec:
    key: str
    name_cn: str
    _build: Callable[[dict], Any]
    space: list[Param] = field(default_factory=list)
    fixed_params: dict = field(default_factory=dict)
    default_params: dict = field(default_factory=dict)
    is_tree: bool = False
    has_native_importance: bool = False
    shap_kind: str = "kernel"  # "tree" | "kernel"

    def build(self, params: dict | None = None):
        params = dict(params or {})
        return self._build({**self.fixed_params, **params})


# --------------------------------------------------------------------------- #
# estimator 工厂（延迟导入第三方库，避免启动开销）
# --------------------------------------------------------------------------- #
def _build_xgb(p):
    from xgboost import XGBRegressor

    return XGBRegressor(**p)


def _build_lgbm(p):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**p)


def _build_cat(p):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**p)


def _build_tabpfn(p):
    # 预训练基础模型：懒加载 tabpfn，并按 GPU 可用性自动选择 device。
    # 需先用 setup_gpu_env 脚本安装 torch(cu113)+tabpfn(+tabpfn-extensions)。
    from tabpfn import TabPFNRegressor

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        device = "cpu"
    return TabPFNRegressor(device=device, ignore_pretraining_limits=True,
                           memory_saving_mode="auto", **p)


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
MODELS: dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    MODELS[spec.key] = spec


# ---- 树模型（有原生重要性，SHAP 用 TreeExplainer）------------------------- #
_register(ModelSpec(
    key="xgboost", name_cn="XGBoost", _build=_build_xgb,
    fixed_params={"objective": "reg:quantileerror", "quantile_alpha": 0.5,
                  "random_state": RANDOM_STATE, "verbosity": 0},
    default_params={"learning_rate": 0.1, "n_estimators": 200,
                    "subsample": 0.8, "max_depth": 5},
    space=[
        Param("learning_rate", "float", 0.01, 0.3, log=True),
        Param("n_estimators", "int", 50, 300),
        Param("subsample", "float", 0.3, 1.0),
        Param("max_depth", "int", 3, 7),
    ],
    is_tree=True, has_native_importance=True, shap_kind="tree",
))

_register(ModelSpec(
    key="lightgbm", name_cn="LightGBM", _build=_build_lgbm,
    fixed_params={"random_state": RANDOM_STATE, "verbose": -1},
    default_params={"learning_rate": 0.1, "n_estimators": 200,
                    "num_leaves": 31, "max_depth": 5},
    space=[
        Param("learning_rate", "float", 0.01, 0.3, log=True),
        Param("n_estimators", "int", 50, 300),
        Param("num_leaves", "int", 10, 200),
        Param("max_depth", "int", 3, 7),
    ],
    is_tree=True, has_native_importance=True, shap_kind="tree",
))

_register(ModelSpec(
    key="catboost", name_cn="CatBoost", _build=_build_cat,
    fixed_params={"random_state": RANDOM_STATE, "verbose": 0},
    default_params={"learning_rate": 0.1, "l2_leaf_reg": 3,
                    "iterations": 400, "depth": 6},
    space=[
        Param("learning_rate", "float", 0.01, 0.3, log=True),
        Param("l2_leaf_reg", "int", 1, 9),
        Param("iterations", "int", 300, 500),
        Param("depth", "int", 3, 10),
    ],
    is_tree=True, has_native_importance=True, shap_kind="tree",
))

_register(ModelSpec(
    key="random_forest", name_cn="随机森林 RF",
    _build=lambda p: RandomForestRegressor(random_state=RANDOM_STATE, **p),
    default_params={"n_estimators": 200, "max_depth": 10, "min_samples_split": 2,
                    "min_samples_leaf": 1, "max_features": "sqrt", "bootstrap": True},
    space=[
        Param("n_estimators", "int", 50, 300),
        Param("max_depth", "int", 3, 15),
        Param("min_samples_split", "int", 2, 10),
        Param("min_samples_leaf", "int", 1, 10),
        Param("max_features", "cat", choices=["sqrt", "log2", None]),
        Param("bootstrap", "cat", choices=[True, False]),
    ],
    is_tree=True, has_native_importance=True, shap_kind="tree",
))

_register(ModelSpec(
    key="gbr", name_cn="梯度提升 GBR",
    _build=lambda p: GradientBoostingRegressor(loss="quantile", alpha=0.5,
                                               random_state=RANDOM_STATE, **p),
    default_params={"n_estimators": 200, "max_depth": 5,
                    "learning_rate": 0.1, "subsample": 0.8},
    space=[
        Param("n_estimators", "int", 50, 300),
        Param("max_depth", "int", 3, 14),
        Param("learning_rate", "float", 0.01, 0.3, log=True),
        Param("subsample", "float", 0.3, 1.0),
    ],
    is_tree=True, has_native_importance=True, shap_kind="tree",
))

_register(ModelSpec(
    key="decision_tree", name_cn="决策树 DT",
    _build=lambda p: DecisionTreeRegressor(random_state=RANDOM_STATE, **p),
    default_params={"max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 3},
    space=[
        Param("max_depth", "int", 10, 30),
        Param("min_samples_split", "int", 2, 30),
        Param("min_samples_leaf", "int", 1, 15),
    ],
    is_tree=True, has_native_importance=True, shap_kind="tree",
))

# ---- 非树模型（无原生重要性，SHAP 用 KernelExplainer）--------------------- #
_register(ModelSpec(
    key="knn", name_cn="KNN",
    _build=lambda p: KNeighborsRegressor(**p),
    default_params={"n_neighbors": 5, "weights": "distance", "p": 2},
    space=[
        Param("n_neighbors", "int", 1, 11),
        Param("weights", "cat", choices=["uniform", "distance"]),
        Param("p", "cat", choices=[1, 2, 3, 4]),
    ],
    is_tree=False, has_native_importance=False, shap_kind="kernel",
))

_register(ModelSpec(
    key="svr", name_cn="SVR",
    _build=lambda p: SVR(**p),
    default_params={"C": 1.0, "kernel": "rbf", "gamma": 0.1},
    space=[
        Param("C", "float", 0.001, 10, log=True),
        Param("kernel", "cat", choices=["linear", "poly", "rbf"]),
        Param("gamma", "float", 0.01, 1, log=True),
    ],
    is_tree=False, has_native_importance=False, shap_kind="kernel",
))

_register(ModelSpec(
    key="mlp", name_cn="MLP 多层感知器",
    _build=lambda p: MLPRegressor(activation="relu", solver="adam",
                                  random_state=RANDOM_STATE, **p),
    default_params={"hidden_layer_sizes": (128, 128), "max_iter": 500, "alpha": 0.001},
    space=[
        Param("hidden_layer_sizes", "cat",
              choices=[(32, 32), (64, 64), (128, 128), (256, 256)]),
        Param("max_iter", "int", 300, 600),
        Param("alpha", "float", 1e-4, 1e-2, log=True),
    ],
    is_tree=False, has_native_importance=False, shap_kind="kernel",
))

# ---- 预训练基础模型（二期，需 torch GPU；运行较慢）------------------------ #
_register(ModelSpec(
    key="tabpfn", name_cn="TabPFN（预训练·需GPU）",
    _build=_build_tabpfn,
    default_params={},   # 基础模型无超参可调
    space=[],            # 空搜索空间 → 优化层直接用默认参数，不做 CV 搜索
    is_tree=False, has_native_importance=False, shap_kind="kernel",
))


def list_models() -> list[tuple[str, str]]:
    """返回 [(key, 中文名), ...] 供 UI 多选使用。"""
    return [(k, s.name_cn) for k, s in MODELS.items()]


def get(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"未知模型: {key}")
    return MODELS[key]
