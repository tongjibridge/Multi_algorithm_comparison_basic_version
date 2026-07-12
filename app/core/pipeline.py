"""统一训练管道：模型无关地串联 数据准备 → 参数优化 → 训练 → 评估。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import data as data_mod
from . import models as models_mod
from . import optimize as opt_mod
from .metrics import regression_metrics
from .space import Param


@dataclass
class TrainResult:
    model_key: str
    model_name: str
    spec: models_mod.ModelSpec
    best_params: dict
    model: Any
    prepared: data_mod.PreparedData
    y_train_pred: pd.Series
    y_test_pred: pd.Series
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]

    # 便捷访问
    @property
    def feature_names(self) -> list[str]:
        return self.prepared.feature_names


def train_one(
    df: pd.DataFrame,
    data_cfg: data_mod.DataConfig,
    model_key: str,
    opt_cfg: opt_mod.OptConfig,
    log: Callable[[str], None] | None = None,
    model_space: list[Param] | None = None,
) -> TrainResult:
    """训练单个模型并返回完整结果。"""
    spec = models_mod.get(model_key)
    if model_space is not None:
        spec = spec.with_space(model_space)
    _log = log or (lambda _m: None)

    _log(f"[{spec.name_cn}] 划分数据集 (test_size={data_cfg.test_size})")
    X_train_raw, X_test_raw, y_train, y_test = data_mod.split_raw(df, data_cfg)

    _log(f"[{spec.name_cn}] 参数优化：{opt_cfg.method}")
    best_params = opt_mod.optimize(spec, X_train_raw, y_train, data_cfg, opt_cfg, _log)
    _log(f"[{spec.name_cn}] 最优参数：{best_params}")

    # 在训练集上 fit 预处理器，得到列顺序一致的 DataFrame（供解释性分析）
    pre = data_mod.Preprocessor(data_cfg.categorical, data_cfg.non_standardize)
    X_train = pre.fit_transform(X_train_raw, y_train)
    X_test = pre.transform(X_test_raw)
    prepared = data_mod.PreparedData(
        X_train_raw=X_train_raw, X_test_raw=X_test_raw,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        preprocessor=pre, feature_names=list(X_train.columns),
    )

    _log(f"[{spec.name_cn}] 训练最终模型")
    model = spec.build(best_params)
    model.fit(X_train, y_train)

    y_train_pred = pd.Series(model.predict(X_train), index=y_train.index)
    y_test_pred = pd.Series(model.predict(X_test), index=y_test.index)
    train_m = regression_metrics(y_train, y_train_pred)
    test_m = regression_metrics(y_test, y_test_pred)
    _log(f"[{spec.name_cn}] 测试集 R²={test_m['R2']:.4f}  RMSE={test_m['RMSE']:.4f}")

    return TrainResult(
        model_key=model_key, model_name=spec.name_cn, spec=spec,
        best_params=best_params, model=model, prepared=prepared,
        y_train_pred=y_train_pred, y_test_pred=y_test_pred,
        train_metrics=train_m, test_metrics=test_m,
    )


def metrics_table(results: list[TrainResult]) -> pd.DataFrame:
    """多模型指标对比表（测试集）。"""
    rows = []
    for r in results:
        rows.append({
            "模型": r.model_name,
            "R²(测试)": round(r.test_metrics["R2"], 4),
            "RMSE(测试)": round(r.test_metrics["RMSE"], 4),
            "MAE(测试)": round(r.test_metrics["MAE"], 4),
            "R²(训练)": round(r.train_metrics["R2"], 4),
            "RMSE(训练)": round(r.train_metrics["RMSE"], 4),
        })
    return pd.DataFrame(rows)
