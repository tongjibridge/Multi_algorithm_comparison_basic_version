import gradio as gr
import pickle
import pandas as pd
import joblib
from tabpfn.model_loading import load_fitted_tabpfn_model


def data_norm(data, scaler):
    # 定义不需要标准化的特征（独热编码特征）
    non_standardize_features = ["B", "C", "G", "带肋", "黏砂", "光圆"]

    # 确保这些特征确实存在于数据中
    existing_non_standardize_features = [
        col for col in non_standardize_features if col in data.columns
    ]

    # 分离需要标准化和不需要标准化的特征
    standardize_features = [
        col for col in data.columns if col not in existing_non_standardize_features
    ]

    # 对需要标准化的特征进行标准化
    data_standardize_scaled = scaler.transform(data[standardize_features])
    data_standardize_scaled_df = pd.DataFrame(
        data_standardize_scaled, columns=standardize_features, index=data.index
    )
    # 合并不需要标准化的特征和标准化后的特征
    data_df = pd.concat(
        [data_standardize_scaled_df, data[existing_non_standardize_features]], axis=1
    )
    data_df = data_df[data.columns]
    return data_df


def calculate_features(
    jin_type,
    jin_surface,
    diameter,
    fcu,
    bond_ratio,
    fiber_content,
    cover_ratio,
    model_choose,
):
    """
    计算所有特征并返回最大值

    参数：
    - jin_type: 筋类型 (B/C/G)
    - jin_surface: 筋表面 (带肋/黏砂/光圆)
    - diameter: 筋直径
    - fcu: 混凝土强度
    - bond_ratio: 粘结长度/直径
    - fiber_content: 钢纤维掺量
    - cover_ratio: 保护层厚度/直径
    """

    # 筋类型独热编码
    jin_type_onehot = {"B": [1, 0, 0], "C": [0, 1, 0], "G": [0, 0, 1]}

    # 筋表面独热编码
    jin_surface_onehot = {"Rib": [1, 0, 0], "SC": [0, 1, 0], "Smo": [0, 0, 1]}

    # 获取独热编码
    type_features = jin_type_onehot[jin_type]
    surface_features = jin_surface_onehot[jin_surface]
    feature_names = [
        "B",
        "C",
        "G",
        "带肋",
        "黏砂",
        "光圆",
        "筋直径",
        "混凝土强度fcu",
        "粘结长度 直径",
        "钢纤维掺量",
        "保护层厚度 直径",
    ]
    # 组合所有特征
    all_features = (
        type_features
        + surface_features
        + [diameter, fcu, bond_ratio, fiber_content, cover_ratio]
    )
    X_df = {}
    for i in range(len(feature_names)):
        X_df[feature_names[i]] = [all_features[i]]
    X_df = pd.DataFrame(X_df)

    if model_choose == "CatBoost":
        from_file = joblib.load("./savemodel/cat/CatBoost.pkl")
        scaler = pickle.load(open("./savemodel/cat/scaler.pkl", "rb"))
        X_df = data_norm(X_df, scaler)
        result = from_file.predict(X_df)
    elif model_choose == "XGBoost":
        from_file = pickle.load(open("./savemodel/xgb/xgb.pkl", "rb"))
        scaler = pickle.load(open("./savemodel/xgb/scaler.pkl", "rb"))
        X_df = data_norm(X_df, scaler)
        result = from_file.predict(X_df)
    elif model_choose == "LightGBM":
        from_file = joblib.load("./savemodel/LightGBM/lgb.pkl")
        scaler = pickle.load(open("./savemodel/LightGBM/scaler.pkl", "rb"))
        X_df = data_norm(X_df, scaler)
        result = from_file.predict(X_df, num_iteration=from_file.best_iteration)
    elif model_choose == "tabpfn":
        from_file = load_fitted_tabpfn_model(
            "./savemodel/tabpfn/tabpfn_model_final.tabpfn_fit", device="cpu"
        )
        scaler = pickle.load(open("./savemodel/tabpfn/scaler.pkl", "rb"))
        X_df = data_norm(X_df, scaler)

        result = from_file.predict(X_df)
    return result.flatten()[0]


# 创建Gradio界面
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="bond strenth Predictor of FRP-UHPC by machine learning",
) as demo:
    gr.Markdown(
        """
        # 🏗️ Bond Strenth Predictor of FRP-UHPC by machine learning
        ### Input various parameters and use machine learning models to predict the bond strength between FRP and UHPC.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Input parameters")

            with gr.Group():
                gr.Markdown("#### Categorical parameters")
                jin_type = gr.Radio(
                    choices=["B", "C", "G"],
                    label="FRP fiber type",
                    value="B",
                    info="Select FRP fiber type",
                )

                jin_surface = gr.Radio(
                    choices=["Rib", "SC", "Smo"],
                    label="FRP fiber surface type",
                    value="Rib",
                    info="Select FRP fiber surface type",
                )

            with gr.Group():
                gr.Markdown("#### Numerical parameters")
                diameter = gr.Number(
                    label="FRP fiber diameter (mm)",
                    value=20,
                    minimum=0,
                    info="Input FRP fiber diameter",
                )

                fcu = gr.Number(
                    label="UHPC compressive strength fcu (MPa)",
                    value=30.0,
                    minimum=0,
                    info="Input UHPC compressive strength fcu (MPa)",
                )

                bond_ratio = gr.Number(
                    label="Bond length/diameter",
                    value=10.0,
                    minimum=0,
                    info="Bond length to diameter ratio",
                )

        with gr.Column(scale=1):
            with gr.Group():
                fiber_content = gr.Number(
                    label="Steel fiber content (%)",
                    value=0.0,
                    minimum=0,
                    info="Steel fiber content volume percentage",
                )

                cover_ratio = gr.Number(
                    label="Cover thickness/diameter",
                    value=2.5,
                    minimum=0,
                    info="Cover thickness to diameter ratio",
                )
                model_choose = gr.Radio(
                    choices=["LightGBM", "XGBoost", "CatBoost", "tabpfn"],
                    label="Model Selection",
                    value="LightGBM",
                    info="Select the model to use",
                )
            calculate_btn = gr.Button(
                "🚀 Start calculating", variant="primary", size="lg"
            )
            gr.Markdown("### 📈 Calculation results")
            output = gr.Number(label="Bond Strength (MPa)")

    # 绑定计算函数
    calculate_btn.click(
        fn=calculate_features,
        inputs=[
            jin_type,
            jin_surface,
            diameter,
            fcu,
            bond_ratio,
            fiber_content,
            cover_ratio,
            model_choose,
        ],
        outputs=output,
    )

    # 添加示例
    gr.Examples(
        examples=[
            ["B", "Rib", 20, 30, 10, 0, 2.5],
            ["C", "Rib", 12, 153.76, 5, 2, 5.75],
            ["G", "Smo", 16, 25, 8, 0.5, 2.0],
        ],
        inputs=[
            jin_type,
            jin_surface,
            diameter,
            fcu,
            bond_ratio,
            fiber_content,
            cover_ratio,
            model_choose,
        ],
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()
