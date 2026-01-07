# Multi_algorithm_comparison_basic_version

本项目旨在比较多种机器学习算法在特定数据集或任务上的性能，以便研究人员和开发者能够快速了解各算法在不同场景下的优劣。

## 项目功能

- **算法比较**：包含多种主流机器学习算法（如 CatBoost、决策树、KNN、LightGBM、随机森林、SVR 等）的实现与调参。
- **自动调参**：使用 Optuna 进行自动化超参数优化，提高模型性能。
- **SHAP 可视化**：支持模型可解释性分析，包含 SHAP 值的多种可视化方法。
- **数据预处理**：提供标准化、编码、特征分析等常用预处理功能。
- **可视化工具**：内建多种可视化函数，包括回归拟合图、残差图、特征重要性图等。

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
  - `xrfm_reg.py`：XRFM 回归模型实现
  - `tabm_reg.py`：TabM 回归模型实现
  - `tabpfn_reg.py`：TabPFN 回归模型实现
  - `bay.py`：贝叶斯回归模型实现
- `corr.py`：用于分析数据集特征之间的相关性，包括连续变量、离散变量和多共线性分析。
- `tools.py`：提供多种可视化和预处理工具函数，如数据标准化、SHAP 可视化、残差图绘制等。
- `paras_read.py`：用于提取最优模型参数。
- `encode.py`：对数据集中的分类变量进行编码。
- `ui.py`：可能包含用户交互或特征计算逻辑。
- `result/`, `savemodel/`：用于保存模型结果和训练完成的模型。

## 依赖库

该项目可能依赖以下库，请确保安装以下 Python 包：
建议优先使用uv进行环境管理

```bash
pip install optuna scikit-learn pandas numpy matplotlib seaborn lightgbm catboost shap
```

## 使用方法

1. **准备数据集**：将数据集整理为 `.csv` 或其他可读格式，并确保数据集路径正确。
2. **运行模型文件**：如 `LightGBM-optuna.py`，该文件将使用 Optuna 自动调参并保存最优模型。
3. **分析结果**：使用 `paras_read.py` 提取最优参数，使用 `tools.py` 中的函数进行可视化分析。
4. **相关性分析**：运行 `corr.py` 以分析数据集中特征之间的关系。

## 参考

微信公众号

* Python+遥感学习日志
* Lvy的口袋
* Python机器学习AI
* AI智能Python学习
* Python机器学习ML
* 3S&ML
* 会一点GIS的地灾研究生

## 许可证

本项目使用 [MIT License](LICENSE)，请在使用前确认遵循该许可协议。

---
