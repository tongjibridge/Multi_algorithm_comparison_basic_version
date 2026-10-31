import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# 以脚本所在目录为基准定位 savemodel，避免受启动目录影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def savemodel(*parts):
    return os.path.join(BASE_DIR, "savemodel", *parts)


# database2.xlsx 的 9 个输入特征（不含目标列 τ$_{u}$）
FEATURE_NAMES = [
    "FRP fiber type",
    "FRP fiber surface type",
    "Processing time",
    "Temperature",
    "Elastic modulus of FRP fiber",
    "f$_{cu}$",
    "l",
    "d",
    "c",
]

# 独热/序数编码列不参与标准化，与训练脚本 non_standardize_features 一致
NON_STANDARDIZE_FEATURES = ["FRP fiber type", "FRP fiber surface type"]


def data_norm(data, scaler):
    """与训练侧 data_norm_get 一致：7 个数值列标准化，2 个类别列原样保留。"""
    existing_non_standardize = [
        col for col in NON_STANDARDIZE_FEATURES if col in data.columns
    ]
    standardize_features = [
        col for col in data.columns if col not in existing_non_standardize
    ]
    scaled = scaler.transform(data[standardize_features])
    scaled_df = pd.DataFrame(
        scaled, columns=standardize_features, index=data.index
    )
    data_df = pd.concat([scaled_df, data[existing_non_standardize]], axis=1)
    return data_df[data.columns]


def build_input_df(values):
    """9 个 UI 输入 -> 单行 DataFrame，列名与 database2.xlsx 完全一致。"""
    return pd.DataFrame([dict(zip(FEATURE_NAMES, values))])


def build_xrfm_model(device, exponent, bandwidth, reg, verbose=True):
    # 与 xrfm_reg.py 的 build_model 完全一致（best.pt 的保存结构）
    from xrfm import xRFM

    rfm_params = {
        "model": {
            "kernel": "l2",
            "exponent": exponent,  # ↓ 降低非线性强度（更稳）
            "bandwidth": bandwidth,  # ↑ 让核更“平滑”（更强泛化）
            "diag": True,  # ✅ 更偏轴对齐/特征选择，表格数据更不容易过拟合
            "bandwidth_mode": "adaptive",  # 保留，但下面配合更强正则
        },
        "fit": {
            "reg": reg,  # ↑ 明显增大正则（最关键）
            "iters": 7,  # ↓ 少迭代一点，减少“学太细”
            "early_stop_rfm": True,  # ↓ 更早停（防止后期开始记噪声）
            "verbose": verbose,
        },
    }
    model = xRFM(
        rfm_params=rfm_params,
        device=device,
        min_subset_size=10000,
        task_type="regression",  # ⭐ 回归模式
        tuning_metric="mse",  # ⭐ 用 MSE 而不是 accuracy
        split_method="top_vector_agop_on_subset",
        split_temperature=None,
        overlap_fraction=0.1,
        n_threads=1,
    )
    return model


# ---------------- 各模型加载器：返回 predict(X_df) -> np.ndarray ----------------
# 产物说明：
#   BayesianRidge/DT/kNN/SVR/MLP/ELM 使用 rebuild_ui_models.py 重建的
#   ui_model.pkl(.lm)/ui_encoder.pkl/ui_scaler.pkl；
#   TabPFN/xRFM 使用训练脚本原生保存的产物。

_MODEL_CACHE = {}


def _load_simple(model_dir, model_file):
    """通用：encoder -> scaler -> sklearn 系模型（含 hpelm 特例走 .lm）。"""
    encoder = joblib.load(os.path.join(model_dir, "ui_encoder.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "ui_scaler.pkl"))
    if model_file.endswith(".lm"):
        from hpelm import ELM

        model = ELM(len(FEATURE_NAMES), 1)
        model.load(os.path.join(model_dir, model_file))

        def predict(X_df):
            X_enc = encoder.transform(X_df)
            X_scaled = data_norm(X_enc, scaler)
            return model.predict(X_scaled.values.astype(np.float64)).flatten()

    else:
        model = joblib.load(os.path.join(model_dir, model_file))

        def predict(X_df):
            X_enc = encoder.transform(X_df)
            X_scaled = data_norm(X_enc, scaler)
            return model.predict(X_scaled)

    return predict


def _load_tabpfn():
    from tabpfn.model_loading import load_fitted_tabpfn_model

    device = "cuda" if _torch_cuda_available() else "cpu"
    model = load_fitted_tabpfn_model(
        savemodel("tabpfn2", "tabpfn_model_final.tabpfn_fit"), device=device
    )
    encoder = joblib.load(savemodel("tabpfn2", "encoder.pkl"))
    scaler = joblib.load(savemodel("tabpfn2", "scaler.pkl"))

    def predict(X_df):
        X_enc = encoder.transform(X_df)
        X_scaled = data_norm(X_enc, scaler)
        return np.asarray(model.predict(X_scaled)).flatten()

    return predict


def _load_xrfm():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_xrfm_model(
        device, exponent=1.3, bandwidth=5.0, reg=5e-3, verbose=False
    )
    model_dir = savemodel("xrfm2")
    X_train_scaled_df = joblib.load(
        os.path.join(model_dir, "X_train_scaled_df.pkl")
    )
    X_train_tensor = torch.tensor(
        X_train_scaled_df.values, dtype=torch.float32, device=device
    )
    ckpt = torch.load(os.path.join(model_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt, X_train_tensor)
    encoder = joblib.load(os.path.join(model_dir, "encoder.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    def predict(X_df):
        X_enc = encoder.transform(X_df)
        X_scaled = data_norm(X_enc, scaler)
        return np.asarray(model.predict(X_scaled)).flatten()

    return predict


def _torch_cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


MODEL_LOADERS = {
    "BayesianRidge": lambda: _load_simple(
        savemodel("BayesianRidge2"), "ui_model.pkl"
    ),
    "DecisionTree": lambda: _load_simple(
        savemodel("DecisionTree2"), "ui_model.pkl"
    ),
    "ELM": lambda: _load_simple(savemodel("ELM2"), "ui_model.lm"),
    "MLP": lambda: _load_simple(savemodel("MLP2"), "ui_model.pkl"),
    "kNN": lambda: _load_simple(savemodel("kNN2"), "ui_model.pkl"),
    "SVR": lambda: _load_simple(savemodel("SVR2"), "ui_model.pkl"),
    "TabPFN": _load_tabpfn,
    "xRFM": _load_xrfm,
}

MODEL_CHOICES = list(MODEL_LOADERS.keys())


def get_predict(model_choose):
    """按需加载并缓存模型的 predict 函数。"""
    if model_choose not in _MODEL_CACHE:
        _MODEL_CACHE[model_choose] = MODEL_LOADERS[model_choose]()
    return _MODEL_CACHE[model_choose]


def calculate_features(
    jin_type,
    jin_surface,
    processing_time,
    temperature,
    elastic_modulus,
    fcu,
    bond_lenth,
    diameter,
    cover_ratio,
    model_choose,
):
    """
    输入 9 个特征，用所选模型预测 FRP-混凝土粘结强度 τu。

    参数：
    - jin_type: FRP 纤维类型 (B/C/G)
    - jin_surface: FRP 纤维表面类型 (SC+SW/RB/...)
    - processing_time: 暴露时间 (days)
    - temperature: 温度 (℃)
    - elastic_modulus: FRP 纤维弹性模量 (GPa)
    - fcu: 混凝土抗压强度 (MPa)
    - bond_lenth: 粘结长度 l
    - diameter: 筋直径 d
    - cover_ratio: 保护层厚度 c
    """
    if model_choose not in MODEL_LOADERS:
        raise gr.Error(
            f"未知模型: {model_choose}，可选: {', '.join(MODEL_CHOICES)}"
        )

    X_df = build_input_df(
        [
            jin_type,
            jin_surface,
            processing_time,
            temperature,
            elastic_modulus,
            fcu,
            bond_lenth,
            diameter,
            cover_ratio,
        ]
    )

    try:
        predict = get_predict(model_choose)
        result = predict(X_df)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"{model_choose} 推理失败: {type(e).__name__}: {e}")

    return float(result.flatten()[0])


try:
    my_theme = gr.Theme.from_hub("JohnSmith9982/small_and_pretty")
except Exception:
    # 离线/无法访问 huggingface.co 时回退到内置主题
    my_theme = gr.themes.Soft()
# 创建Gradio界面
with gr.Blocks(
    theme=my_theme,
    title="bond strenth Predictor of FRP-Concrete by machine learning",
) as demo:
    gr.Markdown(
        """
        # 🏗️ Bond Strenth Predictor of FRP-Concrete in marine environment
        ### Input various parameters and use machine learning models to predict the bond strength between FRP and Concrete in marine environment.
        """
    )

    with gr.Row(scale=1):
        with gr.Column():
            with gr.Group():
                jin_type = gr.Radio(
                    choices=["B", "C", "G"],
                    label="FRP fiber type",
                    value="B",
                    info="Select FRP fiber type",
                )

                # 源数据中 "SW+RB "（尾部多空格）与 "SW+RB" 是两个类别，
                # 界面上合并为一项；编码器按 "SW+RB" 映射
                jin_surface = gr.Radio(
                    choices=[
                        "SC+SW",
                        "RB",
                        "SW",
                        "SW+RB",
                        "SW+RB+SC",
                        "SW+SC",
                        "SC",
                    ],
                    label="FRP fiber surface type",
                    value="SC+SW",
                    info="Select FRP fiber surface type",
                )
                fcu = gr.Number(
                    label="Concrete compressive strength fcu (MPa)",
                    value=70.0,
                    minimum=0,
                    info="Input fcu (MPa), data range: 29.3-84.1",
                )
                fiber_E = gr.Number(
                    label="Elastic modulus of FRP fiber (GPa)",
                    value=50.0,
                    minimum=0,
                    info="Elastic modulus (GPa), data range: 38.1-170.0",
                )
        with gr.Column():
            with gr.Group():
                # 数值输入的 data range 提示取自 database2.xlsx（318 组）的上下限
                pro_time = gr.Number(
                    label="Exposure time(days)",
                    value=30,
                    minimum=0,
                    info="Input exposure time (days), data range: 15-270",
                )
                temperature = gr.Number(
                    label="Temperature (℃)",
                    value=60.0,
                    minimum=0,
                    info="Input temperature (℃), data range: 20-80",
                )

                with gr.Row():
                    diameter = gr.Number(
                        label="diameter",
                        value=12.0,
                        minimum=0,
                        info="diameter (mm), data range: 8-16",
                    )
                    bond_lenth = gr.Number(
                        label="Bond length",
                        value=60.0,
                        minimum=0,
                        info="Bond length (mm), data range: 15-93",
                    )

                cover_ratio = gr.Number(
                    label="Cover thickness",
                    value=94,
                    minimum=0,
                    info="Cover thickness (mm), data range: 46-95",
                )

    with gr.Row(scale=1):
        with gr.Group():
            gr.Markdown("### 📈 Calculation and results")
            model_choose = gr.Radio(
                choices=MODEL_CHOICES,
                label="Model Selection",
                value="BayesianRidge",
                info="Select the model to use",
            )
            calculate_btn = gr.Button(
                "🚀 Start calculating", variant="primary", size="lg"
            )

            output = gr.Number(label="Bond Strength (MPa)")

    # 绑定计算函数（10 个输入与 calculate_features 的 10 个参数一一对应）
    calculate_btn.click(
        fn=calculate_features,
        inputs=[
            jin_type,
            jin_surface,
            pro_time,
            temperature,
            fiber_E,
            fcu,
            bond_lenth,
            diameter,
            cover_ratio,
            model_choose,
        ],
        outputs=output,
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()
