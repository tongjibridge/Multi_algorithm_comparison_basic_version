"""数据层：Excel 读取、列配置、编码、标准化、数据集划分。

设计要点
--------
- 列含义完全由用户在界面指定（目标 / 特征 / 分类列 / 不标准化列），不做硬编码假设。
- ``Preprocessor`` 同时用于两处，保证训练与交叉验证的预处理逻辑一致且**无数据泄漏**：
  1. 最终模型训练：手动 ``fit_transform`` 得到列顺序不变的 DataFrame，供 SHAP / PDP / ALE 使用；
  2. 超参数优化：作为 sklearn ``Pipeline`` 的第一步，在每个 CV 折内独立 fit，避免泄漏。
- 分类编码沿用参考实现的 ``category_encoders.OrdinalEncoder``（类别→整数序号）。
- 标准化沿用 ``StandardScaler``，但跳过用户指定的"不标准化列"（通常即分类列）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:  # category_encoders 为可选依赖时的兜底
    import category_encoders as ce
except Exception:  # pragma: no cover
    ce = None


# --------------------------------------------------------------------------- #
# 配置与结果容器
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """用户在界面上指定的数据列配置与划分参数。"""

    target: str
    features: list[str]
    categorical: list[str] = field(default_factory=list)
    non_standardize: list[str] = field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42

    def validate(self, df: pd.DataFrame) -> None:
        cols = set(df.columns)
        if self.target not in cols:
            raise ValueError(f"目标列 '{self.target}' 不在数据中")
        if not self.features:
            raise ValueError("至少需要选择一个特征列")
        missing = [c for c in self.features if c not in cols]
        if missing:
            raise ValueError(f"特征列不存在: {missing}")
        if self.target in self.features:
            raise ValueError("目标列不能同时作为特征列")
        bad_cat = [c for c in self.categorical if c not in self.features]
        if bad_cat:
            raise ValueError(f"分类列必须是已选特征: {bad_cat}")
        if not (0.05 <= self.test_size <= 0.9):
            raise ValueError("测试集比例需在 0.05~0.9 之间")
        # 目标必须可转为数值（回归任务）
        try:
            pd.to_numeric(df[self.target])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"目标列 '{self.target}' 不是数值，无法用于回归") from exc


@dataclass
class PreparedData:
    """划分 + 预处理后的训练/测试数据，供训练与可解释性分析使用。"""

    X_train_raw: pd.DataFrame  # 原始特征（未编码），用于按折预处理
    X_test_raw: pd.DataFrame
    X_train: pd.DataFrame  # 已编码+标准化（列顺序与原始一致）
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: "Preprocessor"
    feature_names: list[str]


# --------------------------------------------------------------------------- #
# 预处理器
# --------------------------------------------------------------------------- #
class Preprocessor(BaseEstimator, TransformerMixin):
    """编码分类列 + 标准化数值列，输出**列顺序不变**的 DataFrame。

    与 sklearn 兼容，可直接放进 ``Pipeline`` 在每个 CV 折内独立 fit，从而避免泄漏。
    """

    def __init__(self, categorical: list[str] | None = None,
                 non_standardize: list[str] | None = None):
        # sklearn 约定：__init__ 只能原样存储参数（规范化放到 fit），否则 clone 失败
        self.categorical = categorical
        self.non_standardize = non_standardize

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        self.non_standardize_ = list(self.non_standardize or [])
        # 1) 分类编码器
        cats = [c for c in (self.categorical or []) if c in self.columns_]
        if cats and ce is not None:
            self.encoder_ = ce.OrdinalEncoder(cols=cats, handle_unknown="value",
                                              handle_missing="value")
            X_enc = self.encoder_.fit_transform(X, y)
        else:
            self.encoder_ = None
            X_enc = X
        X_enc = X_enc.astype(float)
        # 2) 标准化器（跳过不标准化列）
        self.standardize_ = [c for c in self.columns_ if c not in self.non_standardize_]
        if self.standardize_:
            self.scaler_ = StandardScaler().fit(X_enc[self.standardize_])
        else:
            self.scaler_ = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        X_enc = self.encoder_.transform(X) if self.encoder_ is not None else X
        X_enc = X_enc.astype(float)
        if self.scaler_ is not None:
            scaled = pd.DataFrame(
                self.scaler_.transform(X_enc[self.standardize_]),
                columns=self.standardize_, index=X_enc.index,
            )
            for c in self.non_standardize_:
                if c in X_enc.columns:
                    scaled[c] = X_enc[c].values
            X_enc = scaled
        # 恢复原始列顺序
        return X_enc[self.columns_]


# --------------------------------------------------------------------------- #
# 顶层函数
# --------------------------------------------------------------------------- #
def load_excel(path: str) -> pd.DataFrame:
    """读取 Excel，返回 DataFrame。"""
    return pd.read_excel(path)


def column_names(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns]


def split_raw(df: pd.DataFrame, cfg: DataConfig):
    """按配置切出 X/y 并做 train/test 划分（返回未编码的原始特征）。"""
    cfg.validate(df)
    X = df[cfg.features].copy()
    y = pd.to_numeric(df[cfg.target]).astype(float)
    X = X.loc[y.notna()]
    y = y.loc[y.notna()]
    return train_test_split(X, y, test_size=cfg.test_size,
                            random_state=cfg.random_state)


def prepare(df: pd.DataFrame, cfg: DataConfig) -> PreparedData:
    """完整准备：划分 → 在训练集上 fit 预处理器 → 变换训练/测试集。"""
    X_train_raw, X_test_raw, y_train, y_test = split_raw(df, cfg)
    pre = Preprocessor(categorical=cfg.categorical,
                       non_standardize=cfg.non_standardize)
    X_train = pre.fit_transform(X_train_raw, y_train)
    X_test = pre.transform(X_test_raw)
    return PreparedData(
        X_train_raw=X_train_raw, X_test_raw=X_test_raw,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        preprocessor=pre, feature_names=list(X_train.columns),
    )
