# -*- coding: utf-8 -*-
"""
一次性重建脚本：为 ui2.py 补齐 6 个模型的推理产物
（BayesianRidge / DecisionTree / kNN / SVR / MLP / ELM）。

背景：
- 这些模型的训练脚本只保存了 optuna study（最优超参），模型本身和 scaler 没有落盘；
- BayesianRidge2 目录里的旧产物（bay.pkl/encoder/scaler）是旧 UHPC 数据集的，不可用，
  因此这里按用户要求在 database2 上重跑 optuna 调参；
- DT / kNN / SVR / MLP 为确定性模型，用 study 最优参数 + 与训练脚本完全相同的
  确定性流程（KFold rs=45 第一折 + encode_database + data_norm_get）重建，结果可复现；
- ELM 随机初始化不可精确复现，固定 np.random 种子重建。

产物写入各 savemodel/<name>2/ 目录：
    ui_model.pkl    训练好的模型（可直接 predict；ELM 因无法 pickle 用 hpelm
                    原生格式存为 ui_model.lm）
    ui_encoder.pkl  category_encoders OrdinalEncoder
    ui_scaler.pkl   StandardScaler
均为新文件名，不覆盖任何现有文件。TabPFN(tabpfn2) 与 xRFM(xrfm2) 产物齐全，无需重建。

用法（在项目根目录）：
    .venv/Scripts/python.exe rebuild_ui_models.py
"""

import os
import sys
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from tools import data_norm_get  # noqa: E402
from encode import encode_database  # noqa: E402

CATEGORICAL_COLUMNS = [0, 1]
NON_STANDARDIZE_FEATURES = ["FRP fiber type", "FRP fiber surface type"]


def canonical_split(df):
    """与训练脚本一致的最终训练/测试划分：KFold(5, shuffle, rs=45) 的第一折。"""
    y = df.iloc[:, -1]
    x = df.iloc[:, :-1]
    kf = KFold(n_splits=5, shuffle=True, random_state=45)
    train_index, test_index = next(kf.split(y))
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return x_train, x_test, y_train, y_test


def build_pipeline(x_train, x_test, y_train, y_test):
    """编码 + 标准化，返回 (encoder, scaler, X_train_scaled, X_test_scaled)。"""
    x_train_enc, encoder, _ = encode_database(
        x_train, y_train, CATEGORICAL_COLUMNS
    )
    x_test_enc = encoder.transform(x_test)
    scaler_tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "_ui_rebuild_scaler.pkl")
    X_train_scaled, X_test_scaled = data_norm_get(
        x_train_enc,
        x_test_enc,
        y_train,
        y_test,
        save_path=scaler_tmp,
        non_standardize_features=NON_STANDARDIZE_FEATURES,
    )
    scaler = pickle.load(open(scaler_tmp, "rb"))
    return encoder, scaler, X_train_scaled, X_test_scaled


def save_artifacts(model_dir, model, encoder, scaler):
    # 只写 ui_* 前缀的新文件，绝不触碰训练脚本留下的原始产物
    joblib.dump(model, os.path.join(model_dir, "ui_model.pkl"))
    joblib.dump(encoder, os.path.join(model_dir, "ui_encoder.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "ui_scaler.pkl"))


def report(name, y_test, y_pred):
    print(
        f"  [rebuild] {name}: test R2={r2_score(y_test, y_pred):.4f} "
        f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.4f}"
    )


def main():
    df = pd.read_excel(r"./database2.xlsx")
    x_train, x_test, y_train, y_test = canonical_split(df)
    encoder, scaler, X_train_scaled, X_test_scaled = build_pipeline(
        x_train, x_test, y_train, y_test
    )
    n_features = X_train_scaled.shape[1]
    print(f"database2: {df.shape}, train={len(y_train)}, test={len(y_test)}, "
          f"features={n_features}")

    # ---------- 确定性重建：DT / kNN / SVR / MLP ----------
    deterministic = [
        ("DecisionTree2", "DecisionTree_model_final.pkl",
         lambda p: DecisionTreeRegressor(**p, random_state=42)),
        ("kNN2", "kNN_model_final.pkl",
         lambda p: KNeighborsRegressor(**p)),
        ("SVR2", "SVR_model_final.pkl",
         lambda p: SVR(**p)),
        ("MLP2", "MLP_model_final.pkl",
         lambda p: MLPRegressor(**p, random_state=42)),
    ]
    for model_dir, study_file, factory in deterministic:
        if os.path.exists(os.path.join("savemodel", model_dir, "ui_model.pkl")):
            print(f"[SKIP] {model_dir}: ui_model.pkl 已存在")
            continue
        study = joblib.load(os.path.join("savemodel", model_dir, study_file))
        model = factory(study.best_params)
        model.fit(X_train_scaled, y_train)
        save_artifacts(os.path.join("savemodel", model_dir), model, encoder, scaler)
        print(f"[OK] {model_dir}: {study.best_params}")
        report(model_dir, y_test, model.predict(X_test_scaled))

    # ---------- ELM：固定种子重建（hpelm 随机投影不可精确复现） ----------
    # 注意：hpelm 对象含 lambda 无法 pickle，必须用它原生的 .lm 格式
    from hpelm import ELM

    if not os.path.exists("savemodel/ELM2/ui_model.lm"):
        study = joblib.load("savemodel/ELM2/ELM_model_final.pkl")
        np.random.seed(42)
        elm = ELM(n_features, 1)
        elm.add_neurons(**study.best_params)
        elm.train(X_train_scaled.values, y_train.values, "r")
        lm_path = os.path.abspath("savemodel/ELM2/ui_model.lm")
        if os.path.exists(lm_path):
            os.remove(lm_path)
        elm.save(lm_path)
        joblib.dump(encoder, "savemodel/ELM2/ui_encoder.pkl")
        joblib.dump(scaler, "savemodel/ELM2/ui_scaler.pkl")
        print(f"[OK] ELM2: {study.best_params} (seed=42 重建, .lm 格式)")
        # hpelm 的 load 是实例方法：先构造同维度 ELM 再加载
        elm_verify = ELM(n_features, 1)
        elm_verify.load(lm_path)
        report("ELM2", y_test, elm_verify.predict(
            X_test_scaled.values).flatten())
    else:
        print("[SKIP] ELM2: ui_model.lm 已存在")

    # ---------- BayesianRidge：在 database2 上重跑 optuna（用户要求） ----------
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    x, y = df.iloc[:, :-1], df.iloc[:, -1]

    def objective(trial):
        param_grid = {
            "alpha_1": trial.suggest_float("alpha_1", 1e-6, 1e-3, log=True),
            "alpha_2": trial.suggest_float("alpha_2", 1e-6, 1e-3, log=True),
            "lambda_1": trial.suggest_float("lambda_1", 1e-10, 1e-4, log=True),
            "lambda_2": trial.suggest_float("lambda_2", 1e-10, 1e-4, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 400),
        }
        from sklearn.metrics import mean_pinball_loss

        error, count = 0.0, 0
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(y):
            xt, xe = x.iloc[train_index], x.iloc[test_index]
            yt, ye = y.iloc[train_index], y.iloc[test_index]
            xt_enc, enc, _ = encode_database(xt, yt, CATEGORICAL_COLUMNS)
            xe_enc = enc.transform(xe)
            Xt_s, Xe_s = data_norm_get(
                xt_enc, xe_enc, yt, ye,
                non_standardize_features=NON_STANDARDIZE_FEATURES,
            )
            m = BayesianRidge(**param_grid)
            m.fit(Xt_s, yt)
            error += mean_pinball_loss(ye, m.predict(Xe_s))
            count += 1
        return error / count

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    print(f"[OK] BayesianRidge optuna 完成: {study.best_params} "
          f"(pinball={study.best_value:.4f})")

    bay = BayesianRidge(**study.best_params)
    bay.fit(X_train_scaled, y_train)
    save_artifacts("savemodel/BayesianRidge2", bay, encoder, scaler)
    report("BayesianRidge2", y_test, bay.predict(X_test_scaled))
    # 保存新 study，便于追溯
    joblib.dump(study, "savemodel/BayesianRidge2/ui_study.pkl")

    print("\n全部重建完成：ui_model.pkl / ui_encoder.pkl / ui_scaler.pkl 已就位。")


if __name__ == "__main__":
    main()
