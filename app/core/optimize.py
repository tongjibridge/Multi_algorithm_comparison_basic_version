"""参数优化层：统一接口支持多种优化方法（对应需求 R3）。

支持的方法 (``method`` 取值)：
- ``manual``        直接使用模型的默认参数，不搜索；
- ``optuna_tpe``    Optuna + TPE 采样（贝叶斯式，默认）；
- ``optuna_random`` Optuna + 随机采样；
- ``optuna_cmaes``  Optuna + CMA-ES 进化采样；
- ``grid``          GridSearchCV 网格搜索；
- ``random``        RandomizedSearchCV 随机搜索；
- ``pso``           mealpy 粒子群（元启发式）。

所有方法都通过 ``Pipeline([Preprocessor, model])`` 做 K 折交叉验证，保证**无数据泄漏**。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from . import space as space_mod
from .data import DataConfig, Preprocessor
from .models import ModelSpec

# UI 可见的方法清单：(key, 中文名)
METHODS: list[tuple[str, str]] = [
    ("optuna_tpe", "Optuna · TPE (贝叶斯, 推荐)"),
    ("optuna_random", "Optuna · 随机采样"),
    ("optuna_cmaes", "Optuna · CMA-ES 进化"),
    ("grid", "网格搜索 GridSearch"),
    ("random", "随机搜索 RandomizedSearch"),
    ("pso", "粒子群 PSO (mealpy)"),
    ("manual", "手动 / 默认参数 (不搜索)"),
]

# scoring：sklearn 评分器名（均为越大越好，内部取负得到待最小化的损失）
SCORERS = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


@dataclass
class OptConfig:
    method: str = "optuna_tpe"
    n_trials: int = 40        # optuna / random / pso 的预算
    cv_folds: int = 5
    scoring: str = "rmse"
    random_state: int = 42


def _pipe(spec: ModelSpec, params: dict, cfg: DataConfig) -> Pipeline:
    """把"预处理器 + 模型"组成 sklearn 管道。

    预处理器作为管道第一步，CV 时会在每个折内独立 fit，从而避免用测试折信息
    去拟合编码器/标准化器（数据泄漏）。
    """
    return Pipeline([
        ("pre", Preprocessor(cfg.categorical, cfg.non_standardize)),
        ("model", spec.build(params)),
    ])


def _kf(opt: OptConfig) -> KFold:
    return KFold(n_splits=opt.cv_folds, shuffle=True, random_state=opt.random_state)


def _cv_loss(spec, params, X, y, cfg, opt) -> float:
    """给定一组超参数，返回 K 折交叉验证的"待最小化损失"。

    sklearn 的 neg_* 评分越大越好，这里取负号转成越小越好，供各优化器统一最小化。
    """
    scores = cross_val_score(_pipe(spec, params, cfg), X, y,
                             cv=_kf(opt), scoring=SCORERS[opt.scoring])
    return -float(np.mean(scores))


# --------------------------------------------------------------------------- #
# 各方法实现
# --------------------------------------------------------------------------- #
def _run_optuna(spec, X, y, cfg, opt, sampler, log):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 目标函数：每次 trial 由采样器给出一组超参数，返回其 CV 损失
    def objective(trial):
        params = space_mod.optuna_suggest(trial, spec.space)
        return _cv_loss(spec, params, X, y, cfg, opt)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    # 每若干个 trial 向界面日志汇报一次进度
    done = {"n": 0}

    def _cb(study, trial):
        done["n"] += 1
        if log and (done["n"] % 5 == 0 or done["n"] == opt.n_trials):
            log(f"    优化进度 {done['n']}/{opt.n_trials}，当前最优 "
                f"{opt.scoring.upper()}={study.best_value:.4f}")

    study.optimize(objective, n_trials=opt.n_trials, callbacks=[_cb])
    return dict(study.best_params)


def _run_search(spec, X, y, cfg, opt, randomized, log):
    # sklearn 的 *SearchCV 通过 "step名__参数名" 定位管道内某一步的超参数，
    # 故给每个模型超参数加上 "model__" 前缀
    grid = {f"model__{k}": v for k, v in space_mod.grid_space(spec.space).items()}
    pipe = _pipe(spec, {}, cfg)
    kf = _kf(opt)
    if randomized:
        search = RandomizedSearchCV(
            pipe, grid, n_iter=opt.n_trials, cv=kf, scoring=SCORERS[opt.scoring],
            random_state=opt.random_state, n_jobs=-1, error_score="raise")
    else:
        search = GridSearchCV(pipe, grid, cv=kf, scoring=SCORERS[opt.scoring],
                              n_jobs=-1, error_score="raise")
    search.fit(X, y)
    # 去掉 "model__" 前缀，还原成可直接喂给 spec.build 的参数名
    return {k.replace("model__", "", 1): v for k, v in search.best_params_.items()}


def _run_pso(spec, X, y, cfg, opt, log):
    """粒子群优化：把超参数空间映射为 mealpy 边界，最小化 CV 损失。"""
    from mealpy import PSO, Problem

    bounds = space_mod.mealpy_bounds(spec.space)  # 由声明式空间生成 mealpy 变量边界

    # mealpy 的 Problem：obj_func 接收编码解，decode 回参数字典后算 CV 损失
    class _P(Problem):
        def obj_func(self, sol):
            params = self.decode_solution(sol)
            return _cv_loss(spec, params, X, y, cfg, opt)

    # 用总预算粗略折算迭代代数与种群规模
    epoch = max(5, opt.n_trials // 4)
    pop = max(8, min(20, opt.n_trials))
    problem = _P(bounds=bounds, minmax="min", log_to=None)  # 最小化损失
    model = PSO.OriginalPSO(epoch=epoch, pop_size=pop)
    model.solve(problem)
    # g_best.solution 是最优编码解，解码回超参数字典返回
    return dict(problem.decode_solution(model.g_best.solution))


# --------------------------------------------------------------------------- #
# 顶层入口
# --------------------------------------------------------------------------- #
def optimize(spec: ModelSpec, X, y, cfg: DataConfig, opt: OptConfig, log=None) -> dict:
    """返回最优超参数 dict（不含固定参数）。"""
    method = opt.method
    if method == "manual" or not spec.space:
        return dict(spec.default_params)

    if method.startswith("optuna"):
        import optuna
        sampler = {
            "optuna_tpe": optuna.samplers.TPESampler(seed=opt.random_state),
            "optuna_random": optuna.samplers.RandomSampler(seed=opt.random_state),
            "optuna_cmaes": optuna.samplers.CmaEsSampler(seed=opt.random_state),
        }.get(method, optuna.samplers.TPESampler(seed=opt.random_state))
        return _run_optuna(spec, X, y, cfg, opt, sampler, log)

    if method == "grid":
        return _run_search(spec, X, y, cfg, opt, randomized=False, log=log)
    if method == "random":
        return _run_search(spec, X, y, cfg, opt, randomized=True, log=log)

    if method == "pso":
        try:
            return _run_pso(spec, X, y, cfg, opt, log)
        except Exception as exc:  # noqa: BLE001 —— PSO 失败时回退到 TPE
            if log:
                log(f"    [警告] PSO 失败 ({type(exc).__name__})，回退到 Optuna·TPE")
            import optuna
            return _run_optuna(spec, X, y, cfg, opt,
                               optuna.samplers.TPESampler(seed=opt.random_state), log)

    # 未知方法 → 默认参数
    return dict(spec.default_params)
