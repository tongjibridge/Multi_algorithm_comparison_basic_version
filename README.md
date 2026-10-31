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
- **对比模型**（共 12 个）：单体模型 BayesianRidge、DecisionTree、ELM、kNN、MLP、SVR、TabPFN、xRFM；集成模型 RF、LightGBM、XGBoost、TabM。
- **数据**：`database2.xlsx`（318 组海洋环境 FRP-混凝土粘结试验数据，未入库，请参见论文）。
- **在线预测**：`ui2.py` 提供 Gradio 交互界面，支持 BayesianRidge / DecisionTree / ELM / MLP / kNN / SVR / TabPFN / xRFM 共 8 个模型的一键推理。

## 项目功能

- **算法比较**：包含多种主流机器学习算法（如决策树、KNN、LightGBM、随机森林、SVR、xRFM、TabPFN、TabM 等）的实现与调参。
- **自动调参**：使用 Optuna 进行自动化超参数优化，提高模型性能。
- **SHAP 可视化**：支持模型可解释性分析，包含 SHAP 值的多种可视化方法。
- **数据预处理**：提供标准化、编码、特征分析等常用预处理功能。
- **可视化工具**：内建多种可视化函数，包括回归拟合图、残差图、特征重要性图等。
- **在线预测界面**：`ui2.py` 基于 Gradio 的多模型粘结强度预测 Web UI。

## 文件说明

- **不同机器学习模型的实现文件，包含调参函数 `objective`**：
  - `DT.py`：决策树回归模型实现
  - `KNN.py`：K近邻回归模型实现
  - `LightGBM-optuna.py`：LightGBM 回归模型实现（使用 Optuna 调参）
  - `RF.py`：随机森林回归模型实现
  - `SVR.py`：支持向量回归模型实现
  - `MLP.py`：多层感知器回归模型实现
  - `elm.py`：极限学习机回归模型实现
  - `xrfm_reg.py`：xRFM 回归模型实现
  - `tabm_reg.py`：TabM 回归模型实现（依赖 `tabm_model_tool.py`）
  - `tabpfn_reg.py`：TabPFN 回归模型实现
  - `bay.py`：贝叶斯回归模型实现
- `tools.py`：提供多种可视化和预处理工具函数，如数据标准化、SHAP 可视化、残差图绘制等。
- `paras_read.py`：用于提取最优模型参数。
- `encode.py`：对数据集中的分类变量进行编码。
- `ui2.py`：海洋环境数据集（database2）多模型预测界面，本分支主要维护的推理入口。
- `rebuild_ui_models.py`：一次性重建脚本，为 `ui2.py` 补齐 BayesianRidge / DecisionTree / ELM / MLP / kNN / SVR 的推理产物（`savemodel/*2/ui_*.pkl`）。
- `tabm_model_tool.py`：TabM 训练辅助工具（模型与配置封装）。
- `result/`, `savemodel/`：用于保存模型结果和训练完成的模型（均已加入 `.gitignore`，不入库）。
- 未用于本论文的旧版/作图脚本（CatBoost、GBR、Stacking、XGBoost-mealpy、旧版 ui、corr/dataplot/distributeplot/runall 等）已移至本地 `unused/` 文件夹，不入库。

## 依赖库

该项目可能依赖以下库，请确保安装以下 Python 包：
建议优先使用uv进行环境管理

```bash
pip install optuna scikit-learn pandas numpy matplotlib seaborn lightgbm xgboost shap
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

与论文一致的超参数优化方法、12 个对比模型（8 个单体 + 4 个集成）、可解释性方法及主要工具的规范引用如下：

**超参数优化（TPE，对应论文 [37]–[39]）**

- Bergstra J, Bardenet R, Bengio Y, et al. Algorithms for hyper-parameter optimization[C]//Advances in Neural Information Processing Systems. 2011, 24.
- Bergstra J, Yamins D, Cox D. Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures[C]//International Conference on Machine Learning. PMLR, 2013: 115-123.
- Watanabe S. Tree-structured parzen estimator: Understanding its algorithm components and their roles for better empirical performance[J]. arXiv preprint arXiv:2304.11127, 2023.
- Akiba T, Sasaki S, Koyama M, et al. Optuna: A next-generation hyperparameter optimization framework[C]//Advances in Neural Information Processing Systems. 2019, 32.

**单体模型（8 个）**

- Bishop C M. Pattern recognition and machine learning[M]. New York: Springer, 2006. —— Bayesian Ridge（Bay）
- Breiman L, Friedman J H, Olshen R A, et al. Classification and regression trees[M]. Belmont: Wadsworth, 1984. —— Decision Tree（DT）
- Huang G B, Zhu Q Y, Siew C K. Extreme learning machine: Theory and applications[J]. Neurocomputing, 2006, 70(1-3): 489-501. —— ELM
- Hornik K. Approximation capabilities of multilayer feedforward networks[J]. Neural Networks, 1991, 4(2): 251-257. —— MLP
- Cover T, Hart P. Nearest neighbor pattern classification[J]. IEEE Transactions on Information Theory, 1967, 13(1): 21-27. —— KNN
- Smola A J, Schölkopf B. A tutorial on support vector regression[J]. Statistics and Computing, 2004, 14: 199-222. —— SVR
- Hollmann N, Müller S, Purucker L, et al. Accurate predictions on small data with a tabular foundation model[J]. Nature, 2025, 637: 319-326. https://doi.org/10.1038/s41586-024-08328-6 —— TabPFN
- xRFM: Accurate, scalable, and interpretable feature learning[J]. arXiv preprint arXiv:2508.10053, 2025. [arXiv:2508.10053](https://arxiv.org/abs/2508.10053) —— xRFM（底层方法 Recursive Feature Machine 见：Radhakrishnan A, Beaglehole D, Singh A, et al. Feature learning in neural networks and kernel machines that recursively learn features[J]. arXiv preprint arXiv:2212.13881, 2022）

**集成模型（4 个）**

- Breiman L. Random forests[J]. Machine Learning, 2001, 45(1): 5-32. —— RF
- Ke G, Meng Q, Finley T, et al. LightGBM: A highly efficient gradient boosting decision tree[C]//Advances in Neural Information Processing Systems. 2017, 30. —— LightGBM
- Chen T, Guestrin C. XGBoost: A scalable tree boosting system[C]//Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016: 785-794. https://doi.org/10.1145/2939672.2939785 —— XGBoost
- Gorishniy Y, Rubachev I, Babenko A. TabM: Advancing tabular deep learning with parameter-efficient ensembling[J]. arXiv preprint arXiv:2410.24210, 2024. [arXiv:2410.24210](https://arxiv.org/abs/2410.24210) —— TabM

**可解释性分析（SHAP / PDP / ICE / ALE）**

- Lundberg S M, Lee S I. A unified approach to interpreting model predictions[C]//Advances in Neural Information Processing Systems. 2017, 30. —— SHAP
- Friedman J H. Greedy function approximation: A gradient boosting machine[J]. The Annals of Statistics, 2001, 29(5): 1189-1232. —— PDP
- Goldstein A, Kapelner A, Bleich J, et al. Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation[J]. Journal of Computational and Graphical Statistics, 2015, 24(1): 50-65. —— ICE
- Apley D W, Zhu J. Visualizing the effects of predictor variables in black box supervised learning models[J]. Journal of the Royal Statistical Society: Series B, 2020, 82(4): 1059-1086. —— ALE

**工具**

- Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.

<details>
<summary>BibTeX（点击展开）</summary>

```bibtex
@inproceedings{bergstra2011tpe,
  title     = {Algorithms for Hyper-Parameter Optimization},
  author    = {Bergstra, James and Bardenet, R{'e}mi and Bengio, Yoshua and K{'e}gl, Bal{'a}zs},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {24},
  year      = {2011}
}
@article{watanabe2023tpe,
  title   = {Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance},
  author  = {Watanabe, Shuhei},
  journal = {arXiv preprint arXiv:2304.11127},
  year    = {2023}
}
@inproceedings{akiba2019optuna,
  title     = {Optuna: A Next-generation Hyperparameter Optimization Framework},
  author    = {Akiba, Takuya and Sasaki, Shotaro and Koyama, Masanori and Ohta, Shinji and Kido, Shuhei},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {32},
  year      = {2019}
}
@book{bishop2006prml,
  title     = {Pattern Recognition and Machine Learning},
  author    = {Bishop, Christopher M.},
  publisher = {Springer},
  year      = {2006}
}
@book{breiman1984cart,
  title     = {Classification and Regression Trees},
  author    = {Breiman, Leo and Friedman, Jerome H. and Olshen, Richard A. and Stone, Charles J.},
  publisher = {Wadsworth},
  year      = {1984}
}
@article{huang2006elm,
  title   = {Extreme Learning Machine: Theory and Applications},
  author  = {Huang, Guang-Bin and Zhu, Qin-Yu and Siew, Chee-Kheong},
  journal = {Neurocomputing},
  volume  = {70},
  number  = {1-3},
  pages   = {489--501},
  year    = {2006}
}
@article{hornik1991mlp,
  title   = {Approximation Capabilities of Multilayer Feedforward Networks},
  author  = {Hornik, Kurt},
  journal = {Neural Networks},
  volume  = {4},
  number  = {2},
  pages   = {251--257},
  year    = {1991}
}
@article{cover1967knn,
  title   = {Nearest Neighbor Pattern Classification},
  author  = {Cover, Thomas and Hart, Peter},
  journal = {IEEE Transactions on Information Theory},
  volume  = {13},
  number  = {1},
  pages   = {21--27},
  year    = {1967}
}
@article{smola2004svr,
  title   = {A Tutorial on Support Vector Regression},
  author  = {Smola, Alex J. and Sch{"o}lkopf, Bernhard},
  journal = {Statistics and Computing},
  volume  = {14},
  pages   = {199--222},
  year    = {2004}
}
@article{hollmann2025tabpfn,
  title   = {Accurate Predictions on Small Data with a Tabular Foundation Model},
  author  = {Hollmann, Noah and M{"u}ller, Samuel and Purucker, Lennart and Arjun, Arjun and Ungru, Aaditya and Hutter, Frank and others},
  journal = {Nature},
  volume  = {637},
  pages   = {319--326},
  year    = {2025},
  doi     = {10.1038/s41586-024-08328-6}
}
@article{xrfm2025,
  title   = {xRFM: Accurate, scalable, and interpretable feature learning},
  journal = {arXiv preprint arXiv:2508.10053},
  year    = {2025}
}
@article{radhakrishnan2022rfm,
  title   = {Feature Learning in Neural Networks and Kernel Machines that Recursively Learn Features},
  author  = {Radhakrishnan, Adityanarayanan and Beaglehole, Daniel and Singh, Ankit and Belkin, Mikhail},
  journal = {arXiv preprint arXiv:2212.13881},
  year    = {2022}
}
@article{breiman2001rf,
  title   = {Random Forests},
  author  = {Breiman, Leo},
  journal = {Machine Learning},
  volume  = {45},
  number  = {1},
  pages   = {5--32},
  year    = {2001}
}
@inproceedings{ke2017lightgbm,
  title     = {LightGBM: A Highly Efficient Gradient Boosting Decision Tree},
  author    = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}
@inproceedings{chen2016xgboost,
  title     = {XGBoost: A Scalable Tree Boosting System},
  author    = {Chen, Tianqi and Guestrin, Carlos},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  year      = {2016},
  doi       = {10.1145/2939672.2939785}
}
@article{tabm2024,
  title   = {TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling},
  author  = {Gorishniy, Yury and Rubachev, Ivan and Babenko, Artem},
  journal = {arXiv preprint arXiv:2410.24210},
  year    = {2024}
}
@inproceedings{lundberg2017shap,
  title     = {A Unified Approach to Interpreting Model Predictions},
  author    = {Lundberg, Scott M. and Lee, Su-In},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}
@article{friedman2001pdp,
  title   = {Greedy Function Approximation: A Gradient Boosting Machine},
  author  = {Friedman, Jerome H.},
  journal = {The Annals of Statistics},
  volume  = {29},
  number  = {5},
  pages   = {1189--1232},
  year    = {2001}
}
@article{goldstein2015ice,
  title   = {Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation},
  author  = {Goldstein, Alex and Kapelner, Adam and Bleich, Justin and Pitkin, Emil},
  journal = {Journal of Computational and Graphical Statistics},
  volume  = {24},
  number  = {1},
  pages   = {50--65},
  year    = {2015}
}
@article{apley2020ale,
  title   = {Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models},
  author  = {Apley, Daniel W. and Zhu, Junyi},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {82},
  number  = {4},
  pages   = {1059--1086},
  year    = {2020}
}
@article{pedregosa2011sklearn,
  title   = {Scikit-learn: Machine Learning in Python},
  author  = {Pedregosa, Fabian and Varoquaux, Ga{"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and others},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011}
}
```

</details>

## 许可证

本项目使用 [MIT License](LICENSE)，请在使用前确认遵循该许可协议。

---
