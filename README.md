# Multi_algorithm_comparison_basic_version — MarineEnvironments 分支

> **本分支是论文的官方代码仓库分支：**
>
> **Prediction of FRP Reinforced Concrete Bond Durability Performance in Marine Environments Based on Tree-structured Parzen Estimator and Machine Learning**
>
> （基于树形结构 Parzen 估计器与机器学习预测海洋环境下 FRP 增强混凝土粘结耐久性能）
>
> 论文中的全部模型训练、超参数优化（TPE）、可解释性分析与在线预测界面代码均在本分支维护。

## 论文对应内容

- **任务**：以 FRP 纤维类型/表面类型、暴露时间、温度、弹性模量、混凝土强度、粘结长度、直径、保护层厚度等 9 个特征，预测海洋环境下 FRP-混凝土粘结强度 τu。
- **调参方法**：所有模型均使用 Optuna 的 **TPE（Tree-structured Parzen Estimator）** 采样器进行超参数优化。
- **对比模型**：BayesianRidge、DecisionTree、ELM、kNN、LightGBM、MLP、RF、SVR、XGBoost、CatBoost、GBR、Stacking、TabM、TabPFN、xRFM 等。
- **数据**：`database2.xlsx`（318 组海洋环境 FRP-混凝土粘结试验数据，未入库，请参见论文）。
- **在线预测**：`ui2.py` 提供 Gradio 交互界面，支持 BayesianRidge / DecisionTree / ELM / MLP / kNN / SVR / TabPFN / xRFM 共 8 个模型的一键推理。

## 项目功能

- **算法比较**：包含多种主流机器学习算法（如 CatBoost、决策树、KNN、LightGBM、随机森林、SVR 等）的实现与调参。
- **自动调参**：使用 Optuna 进行自动化超参数优化，提高模型性能。
- **SHAP 可视化**：支持模型可解释性分析，包含 SHAP 值的多种可视化方法。
- **数据预处理**：提供标准化、编码、特征分析等常用预处理功能。
- **可视化工具**：内建多种可视化函数，包括回归拟合图、残差图、特征重要性图等。
- **在线预测界面**：`ui2.py` 基于 Gradio 的多模型粘结强度预测 Web UI。

## 文件说明

- **不同机器学习模型的实现文件，包含调参函数 `objective`**：
  - `CatBoost.py`：CatBoost 回归模型实现
  - `DT.py`：决策树回归模型实现
  - `KNN.py`：K近邻回归模型实现
  - `LightGBM-optuna.py`：LightGBM 回归模型实现（使用 Optuna 调参）
  - `RF.py`：随机森林回归模型实现
  - `SVR.py`：支持向量回归模型实现
  - `GBR.py`：梯度提升回归模型实现
  - `MLP.py`：多层感知器回归模型实现
  - `elm.py`：极限学习机回归模型实现
  - `xrfm_reg.py`：xRFM 回归模型实现
  - `tabm_reg.py`：TabM 回归模型实现
  - `tabpfn_reg.py`：TabPFN 回归模型实现
  - `bay.py`：贝叶斯回归模型实现
- `corr.py`：用于分析数据集特征之间的相关性，包括连续变量、离散变量和多共线性分析。
- `tools.py`：提供多种可视化和预处理工具函数，如数据标准化、SHAP 可视化、残差图绘制等。
- `paras_read.py`：用于提取最优模型参数。
- `encode.py`：对数据集中的分类变量进行编码。
- `ui.py`：旧版（UHPC 数据集）预测界面。
- `ui2.py`：海洋环境数据集（database2）多模型预测界面，本分支主要维护的推理入口。
- `rebuild_ui_models.py`：一次性重建脚本，为 `ui2.py` 补齐 BayesianRidge / DecisionTree / ELM / MLP / kNN / SVR 的推理产物（`savemodel/*2/ui_*.pkl`）。
- `result/`, `savemodel/`：用于保存模型结果和训练完成的模型（均已加入 `.gitignore`，不入库）。

## 依赖库

该项目可能依赖以下库，请确保安装以下 Python 包：
建议优先使用uv进行环境管理

```bash
pip install optuna scikit-learn pandas numpy matplotlib seaborn lightgbm catboost shap
```

TabM / TabPFN / xRFM / ELM 等模型还需要：

```bash
pip install tabpfn tabpfn-extensions rtdl-revisiting-models xrfm hpelm torch
```

## 使用方法

1. **准备数据集**：将 `database2.xlsx` 放到项目根目录（数据未随仓库分发）。
2. **运行模型文件**：如 `LightGBM-optuna.py`，该文件将使用 Optuna（TPE）自动调参并保存最优模型。
3. **重建推理产物**：运行 `python rebuild_ui_models.py`，生成 `ui2.py` 所需的 `ui_model.pkl / ui_encoder.pkl / ui_scaler.pkl`（ELM 为 `ui_model.lm`）。
4. **启动预测界面**：`python ui2.py`，在浏览器中选择模型并输入参数，即可预测粘结强度。
5. **分析结果**：使用 `paras_read.py` 提取最优参数，使用 `tools.py` 中的函数进行可视化分析。
6. **相关性分析**：运行 `corr.py` 以分析数据集中特征之间的关系。

## 参考

微信公众号

* Python+遥感学习日志
* Lvy的口袋
* Python机器学习AI
* AI智能Python学习
* Python机器学习ML
* 3S&ML
* 会一点GIS的地灾研究生

### 模型方法引用

本分支使用的部分模型方法出处如下（其余方法引用请参见论文正文）：

**xRFM**（可解释递归特征机，`xrfm_reg.py`、`ui2.py`）：

> xRFM: Accurate, scalable, and interpretable feature learning.
> *arXiv preprint* arXiv:2508.10053, 2025.
> [https://arxiv.org/abs/2508.10053](https://arxiv.org/abs/2508.10053)

```bibtex
@article{xrfm2025,
  title   = {xRFM: Accurate, scalable, and interpretable feature learning},
  journal = {arXiv preprint arXiv:2508.10053},
  year    = {2025}
}
```

**TabM**（参数高效集成的表格深度学习，`tabm_reg.py`）：

> Gorishniy, Y., Rubachev, I., & Babenko, A.
> TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.
> *arXiv preprint* arXiv:2410.24210, 2024.
> [https://arxiv.org/abs/2410.24210](https://arxiv.org/abs/2410.24210)

```bibtex
@article{tabm2024,
  title   = {TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling},
  author  = {Gorishniy, Yury and Rubachev, Ivan and Babenko, Artem},
  journal = {arXiv preprint arXiv:2410.24210},
  year    = {2024}
}
```

## 许可证

本项目使用 [MIT License](LICENSE)，请在使用前确认遵循该许可协议。

---
