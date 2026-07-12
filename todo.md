# 可解释机器学习程序 — 开发计划 (todo.md)

> 目标：基于 `Multi_algorithm_comparison_basic_version/` 中"多算法预测混凝土粘结强度"的实践代码，
> 开发一个通用的**可解释机器学习 GUI 程序**。
>
> 技术栈：Python · [NiceGUI](https://nicegui.io/) (GUI) · uv (环境管理) · scikit-learn / XGBoost / LightGBM / CatBoost / SHAP / PyALE / Optuna 等。

---

## 一、需求拆解（来自用户）

| # | 需求 | 说明 |
|---|------|------|
| R1 | **用户选择模型** | 单选/多选；覆盖现有 15 个算法，支持横向对比 |
| R2 | **导入 Excel 作为数据源** | 上传 `.xlsx`；可指定特征列/目标列/分类列/不标准化列 |
| R3 | **选择参数优化方法** | UI 先选 Optuna/mealpy，再选具体算法；后端保留 GridSearch、RandomizedSearch、手动/默认参数兼容路径 |
| R4 | **选择输出的结果图像 + 保存位置** | 勾选需要的图表种类；指定输出目录；训练后预览并保存 |

---

## 二、对现有代码的分析结论

### 2.1 现有资产（可复用）
- **统一的数据管道**：读 Excel → `encode.py`(CatBoost/Ordinal 编码) → `tools.data_norm_get`(StandardScaler 标准化) → 5 折 KFold。
- **完整的绘图库** `tools.py`：回归拟合图、特征重要性(原生+置换)、残差图、PDP/ICE(1D/2D/3D)、3D 散点、SHAP 全套、ALE(1D/2D)。
- **优化方法范例**：Optuna `objective(trial)`（多数文件）、mealpy PSO（`XGBoost_mealpy.py`）。
- **预测型 UI 范例** `ui2cn.py`（Gradio，单样本预测）——交互逻辑可参考，但需用 NiceGUI 重写并扩展为"训练+解释"。
- 中文字体文件 `times+simsun.ttf`（绘图需要）。

### 2.2 主要问题（需在新程序中解决）
1. **严重代码重复**：15 个模型文件各自重复约 900 行近乎相同的管道 → 必须抽象为"模型无关的统一管道 + 模型注册表"。
2. **硬编码**：数据路径(`./database2.xlsx`)、分类列(`[0,1]`)、特征名、保存目录全部写死 → 改为运行时配置。
3. **绘图后端**：`matplotlib.use("TkAgg")` 在无界面/服务端会出错 → 统一改为 `Agg`，图像存文件后在 NiceGUI 中展示。
4. **SHAP 解释器选择**：树模型用 `TreeExplainer`，非树模型(SVR/MLP/KNN/ELM/Bayesian)需 `KernelExplainer`（现为注释）→ 按模型类型自动选择。
5. **原生特征重要性**仅树模型有 → 非树模型回退到置换重要性/SHAP 重要性。
6. **长耗时任务**：训练+SHAP 交互值计算很慢 → 需后台线程 + 进度反馈，避免阻塞 UI。

### 2.3 模型清单与能力矩阵
| 模型 | 文件 | 类型 | 原生重要性 | SHAP 解释器 |
|------|------|------|:---:|------|
| XGBoost | XGBoost.py | 树 | ✓ | TreeExplainer |
| LightGBM | LightGBM-optuna.py | 树 | ✓ | TreeExplainer |
| CatBoost | CatBoost.py | 树 | ✓ | TreeExplainer |
| 随机森林 RF | RF.py | 树 | ✓ | TreeExplainer |
| 梯度提升 GBR | GBR.py | 树 | ✓ | TreeExplainer |
| 决策树 DT | DT.py | 树 | ✓ | TreeExplainer |
| KNN | KNN.py | 非树 | ✗ | KernelExplainer |
| SVR | SVR.py | 非树 | ✗ | KernelExplainer |
| MLP | MLP.py | 非树 | ✗ | KernelExplainer |
| ELM | elm.py | 非树 | ✗ | KernelExplainer |
| 贝叶斯岭回归 | bay.py | 线性 | 系数 | KernelExplainer/Linear |
| TabM | tabm_reg.py | 深度表格 | ✗ | KernelExplainer |
| TabPFN | tabpfn_reg.py | 深度表格 | ✗ | KernelExplainer |
| xRFM | xrfm_reg.py | 核方法 | ✗ | KernelExplainer |
| Stacking | stack.py | 集成 | ✗ | KernelExplainer |

> 一期(MVP)建议优先支持 6 个树模型 + KNN/SVR/MLP（SHAP 兼容性好、依赖轻），TabM/TabPFN/xRFM(需 torch)放二期。

---

## 三、目标架构

```
ExplainableML/
├── todo.md                      # 本文件
├── README.md
├── pyproject.toml               # uv 管理，迁移所需依赖 + nicegui
├── app/
│   ├── main.py                  # NiceGUI 入口
│   ├── core/
│   │   ├── data.py              # Excel 读取、列配置、编码、标准化、数据集划分
│   │   ├── models.py            # 模型注册表：工厂 + 默认参数 + 各优化方法的搜索空间
│   │   ├── optimize.py          # 优化方法注册表：optuna/grid/random/mealpy/manual
│   │   ├── pipeline.py          # 模型无关的统一训练流程（CV + 训练 + 预测）
│   │   ├── metrics.py           # MSE/RMSE/MAE/R2 等指标
│   │   ├── plots.py             # 由 tools.py 重构：所有绘图函数，后端=Agg，模型无关
│   │   ├── explain.py           # 重要性 / PDP·ICE / SHAP / ALE 编排（按模型能力分派）
│   │   └── persistence.py       # 模型、scaler、encoder、结果表的保存/加载
│   ├── ui/
│   │   ├── layout.py            # 页面布局与各步骤面板
│   │   ├── state.py             # 全局配置/运行状态
│   │   └── tasks.py             # 后台训练任务 + 进度/日志回传
│   └── assets/
│       └── times+simsun.ttf     # 绘图中文字体
├── data/                        # 示例数据（可放 database2.xlsx 的副本）
└── outputs/                     # 默认结果输出目录（gitignore）
```

---

## 四、开发任务（按阶段）

> 状态口径（2026-07-11）：`[x]` 表示当前代码已实现；`[ ]` 表示未实现或仅部分实现。

### 阶段 0 — 项目脚手架
- [x] uv 工程、`pyproject.toml`、`app/core/` 与 `app/ui/` 分层。
- [x] Agg 绘图后端和 `app/assets/times+simsun.ttf` 中文字体。

### 阶段 1 — 数据层 `core/data.py`
- [x] Excel 读取、列名提取、目标/特征/分类/不标准化列配置。
- [x] OrdinalEncoder + StandardScaler，训练集 fit、测试集 transform。
- [x] train/test split、随机种子、优化层可配置 K 折 CV。
- [ ] 已校验列冲突、空特征和非数值目标；特征缺失值统一策略待补充。

### 阶段 2 — 模型注册表 `core/models.py`
- [x] `ModelSpec` 统一构建器、默认参数、搜索空间和解释能力。
- [x] 9 个稳定模型：6 树模型 + KNN/SVR/MLP。
- [x] 可选 TabPFN 已注册（懒加载、无原生重要性、尚未实跑）；共 10 个注册模型。
- [ ] TabM、xRFM、Stacking、ELM、贝叶斯岭回归待迁移。

### 阶段 3 — 优化层 `core/optimize.py`
- [x] 统一优化接口、预算、CV、评分和随机种子。
- [x] Optuna：TPE / Random / CMA-ES。
- [x] mealpy：PSO / GWO / HHO / ARO / INFO。
- [x] 后端保留 Grid / Random / Manual；当前 UI 不展示。
- [x] CV 内拟合预处理器，支持 RMSE / MAE / R²。

### 阶段 4 — 训练管道
- [x] 数据划分 → 预处理 → 优化 → 训练 → 预测。
- [x] MSE / RMSE / MAE / R² 训练与测试指标。
- [x] 训练/测试预测 xlsx 和多模型指标对比。

### 阶段 5 — 绘图与解释
- [x] 拟合图、残差图、原生/置换/SHAP 重要性。
- [x] SHAP summary / dependence / waterfall。
- [x] 1D PDP/ICE、2D PDP、ALE 1D；支持 png/svg/pdf、dpi、Top-K 和配色。
- [x] TreeExplainer / KernelExplainer 自动分派及非树模型回退。
- [ ] 1D PDP 置信区间、3D PDP/散点、SHAP interaction、ALE 2D 待实现。

### 阶段 6 — NiceGUI 前端
- [x] 数据、模型、优化、输出、运行五步单页流程。
- [x] Excel 上传、本地路径回车加载、列配置和数据预览。
- [x] 模型多选、Optuna/mealpy 两级下拉、预算/CV/评分。
- [x] 后台线程、进度、日志、指标表、下载和打开输出目录。
- [x] 4:3 结果容器 + contain 等比例完整预览。

### 阶段 7 — 测试与文档
- [x] 冒烟测试覆盖 RF（树）+ KNN（非树）的训练、解释和指标对比。
- [x] README 包含安装、启动、操作、测试和 mealpy 扩展说明。
- [ ] data/optimize/pipeline 单元测试待补充。
- [ ] KernelExplainer 已下采样；长任务取消、超时和完整性能策略待实现。

---

## 五、当前任务进度

### 当前实现快照（2026-07-11）
- [x] 阶段 0–4：脚手架、数据、10 个注册模型、优化器和训练管道。
- [x] 阶段 5（部分）：常用解释图、PDP/ICE、2D PDP、ALE 1D 和结果表。
- [x] 阶段 6：NiceGUI 完整主流程、后台运行、日志、指标和画廊。
- [x] 阶段 7（部分）：RF + KNN 核心冒烟测试、README、浏览器交互验证。
- [x] 最近更新：两级优化器下拉、5 个 mealpy 算法、4:3 完整图片预览、路径回车加载。

### 后续待办
- [ ] 特征缺失值处理和核心单元测试。
- [ ] 长任务取消/超时与 KernelExplainer 性能优化。
- [ ] TabPFN 实跑验证；迁移 Stacking / ELM / 贝叶斯岭回归 / TabM / xRFM。
- [ ] 高级解释图：3D PDP/散点、SHAP interaction、ALE 2D。
- [ ] 模型、预处理器和解释结果持久化。
- [ ] 环境对齐：`.python-version` 为 3.10，当前 `.venv` 实测为 Python 3.11.12。

---

## 六、已确认的决策（更新至 2026-07-11）

1. **模型范围**：9 个稳定模型 + 1 个可选且未验证的 TabPFN。
2. **运行形态**：本地浏览器 NiceGUI，不打包桌面 exe。
3. **数据与任务**：列含义由用户配置，仅支持回归。
4. **优化器 UI**：仅展示 Optuna 和 mealpy；Grid / Random / Manual 保留为后端兼容路径。
5. **环境口径**：`pyproject.toml` 支持 Python 3.10–3.12；目标版本为 3.10，当前 `.venv` 为 3.11.12；默认依赖不含 torch。

---

## 七、9 个模型的超参数搜索空间（从现有 `objective` 提取）

| 模型 | 搜索空间 | 固定项 |
|------|----------|--------|
| XGBoost | learning_rate(0.01–0.3,log)·n_estimators(50–300)·subsample(0.3–1.0)·max_depth(3–7) | objective=reg:quantileerror, quantile_alpha=0.5 |
| LightGBM | learning_rate(0.01–0.3,log)·n_estimators(50–300)·num_leaves(10–200)·max_depth(3–7) | verbose=-1 |
| CatBoost | learning_rate(0.01–0.3,log)·l2_leaf_reg(1–9)·iterations(300–500)·depth(3–10) | verbose=0 |
| RandomForest | n_estimators(50–300)·max_depth(3–15)·min_samples_split(2–10)·min_samples_leaf(1–10)·max_features([sqrt,log2,None])·bootstrap([T,F]) | — |
| GBR | n_estimators(50–300)·max_depth(3–14)·learning_rate(0.01–0.3,log)·subsample(0.3–1.0) | loss=quantile, alpha=0.5 |
| DecisionTree | max_depth(10–30)·min_samples_split(2–30)·min_samples_leaf(1–15) | — |
| KNN | n_neighbors(1–11)·weights([uniform,distance])·p([1,2,3,4]) | — |
| SVR | C(0.001–10,log)·kernel([linear,poly,rbf])·gamma(0.01–1,log) | — |
| MLP | hidden_layer_sizes([(32,32),(64,64),(128,128),(256,256)])·max_iter(300–600)·alpha(1e-4–1e-2,log) | activation=relu, solver=adam |
