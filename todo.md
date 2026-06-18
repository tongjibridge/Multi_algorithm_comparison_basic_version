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
| R3 | **选择参数优化方法** | Optuna(TPE/Random/CMA-ES)、GridSearch、RandomizedSearch、mealpy 元启发式、手动/默认参数 |
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

### 阶段 0 — 项目脚手架
- [ ] 用 uv 初始化新工程的 `pyproject.toml`，迁移现有依赖并新增 `nicegui`（去掉仅预测用的 gradio，按需保留）。
- [ ] 建立 `app/` 目录结构与空模块；配置 logging。
- [ ] 把 `times+simsun.ttf` 复制到 `app/assets/`，绘图统一加载该字体。
- [ ] 全局设置 `matplotlib.use("Agg")`。

### 阶段 1 — 核心数据层 `core/data.py` (对应 R2)
- [ ] Excel 读取（openpyxl），返回列名列表供 UI 配置。
- [ ] 列配置：目标列、特征列、分类列(索引/列名)、不标准化列。
- [ ] 复用/封装 `encode.py` 的分类编码（fit/transform 分离，保存 encoder）。
- [ ] 复用 `tools.data_norm_get` 的标准化逻辑（保存 scaler）。
- [ ] 数据集划分：KFold（折数可配）或 train/test split；随机种子可配。
- [ ] 输入校验与友好报错（缺失值、非数值目标、列选择冲突等）。

### 阶段 2 — 模型注册表 `core/models.py` (对应 R1)
- [ ] 定义统一接口：`build(params) -> estimator`、`default_params`、`search_space(method)`、`capabilities`(是否树模型/原生重要性/SHAP 类型)。
- [ ] 为每个模型登记上述元数据（从现有各 `*.py` 的 `objective` 提取搜索空间）。
- [ ] 一期落地 6 树模型 + KNN/SVR/MLP；二期补 TabM/TabPFN/xRFM/Stacking/ELM/Bayesian。

### 阶段 3 — 参数优化层 `core/optimize.py` (对应 R3)
- [ ] 统一优化接口：`optimize(model_key, X, y, method, n_trials/iters, cv) -> best_params`。
- [ ] 适配器：
  - [ ] Optuna —— TPESampler / RandomSampler / CmaEsSampler（复用现有 `objective` 模式）。
  - [ ] GridSearchCV / RandomizedSearchCV（sklearn）。
  - [ ] mealpy 元启发式（PSO 等，参考 `XGBoost_mealpy.py` 的 `Problem` 封装）。
  - [ ] 手动/默认参数（跳过搜索）。
- [ ] 统一目标函数（默认 5 折 CV + `mean_pinball_loss` 或 RMSE，可选）。

### 阶段 4 — 统一训练管道 `core/pipeline.py`
- [ ] 串联：数据划分 → 编码 → 标准化 → 优化得最优参 → 训练最终模型 → 训练/测试预测。
- [ ] 计算并返回指标（`core/metrics.py`）。
- [ ] 导出缩放后/未缩放结果表到 xlsx（参考现有逻辑）。
- [ ] 多模型批量运行 + 汇总对比表（指标横向对比）。

### 阶段 5 — 绘图与可解释性 `core/plots.py` + `core/explain.py` (对应 R4)
- [ ] 将 `tools.py` 全部绘图函数重构为模型无关、参数化（保存路径/标题/字体/dpi）。
- [ ] 按"图表种类"组织为可勾选清单：
  - [ ] 回归拟合图（训练/测试）
  - [ ] 特征重要性（原生 / 置换 / SHAP）
  - [ ] 残差分析图
  - [ ] PDP / ICE（1D 含置信区间）
  - [ ] 2D PDP 热力图 / 3D PDP 曲面 / 3D 散点 / 固定值 3D PDP
  - [ ] SHAP：summary(条形+散点)、dependence、waterfall、interaction 热力图、interaction dependence
  - [ ] ALE（1D / 2D）
- [ ] `explain.py` 按模型能力自动选择 SHAP 解释器（Tree vs Kernel）并对非树模型给出回退方案。
- [ ] 输出格式可配（png/svg/pdf）、dpi 可配。

### 阶段 6 — NiceGUI 前端 `app/ui/` (整合 R1–R4)
- [ ] 分步向导式布局：① 数据导入与列配置 → ② 模型选择 → ③ 优化方法与参数 → ④ 图表勾选与保存位置 → ⑤ 运行。
- [ ] 文件上传组件（Excel）；列选择下拉/多选；数据预览表格。
- [ ] 模型多选 + 优化方法选择 + 试验次数/折数等参数输入。
- [ ] 图表勾选清单 + 输出目录选择（本地路径输入/选择）+ 图片格式/dpi。
- [ ] "开始运行"→ 后台线程执行（`ui/tasks.py`），实时进度条 + 日志输出，避免阻塞。
- [ ] 结果区：指标对比表 + 生成图像画廊预览 + "打开输出目录"。
- [ ] 异常捕获与用户提示。

### 阶段 7 — 集成、测试与文档
- [ ] 用 `database2.xlsx` 做端到端冒烟测试（至少 1 树模型 + 1 非树模型，跑通全流程）。
- [ ] 核心层单元测试（data/optimize/pipeline 的关键路径）。
- [ ] 编写运行说明（`uv run` 启动方式）、更新 README。
- [ ] 性能：大数据集/慢解释器的超时与可取消机制。

---

## 五、当前任务进度

- [x] **任务1**：分析项目，规划开发任务并输出到 `todo.md`（本文件）。
- [x] **任务2**：在 `ExplainableML/` 内建立无远程的 git 仓库，并通过 `.gitignore` 忽略 `Multi_algorithm_comparison_basic_version/`。

### 一期 MVP 开发进度（2026-06-11，已端到端验证）

- [x] **阶段0** 脚手架：uv + Python 3.11、`app/` 分层、Agg 后端、中文字体。
- [x] **阶段1** 数据层 `data.py`：Excel 读取、列配置、`Preprocessor`（编码+标准化，防泄漏）、划分。
- [x] **阶段2** 模型注册表 `models.py`：9 个模型（声明式搜索空间 `space.py`）。
- [x] **阶段3** 优化层 `optimize.py`：Optuna(TPE/随机/CMA-ES) / Grid / Random / PSO / 手动。
- [x] **阶段4** 训练管道 `pipeline.py`：统一流程 + 指标 + 多模型对比表。
- [x] **阶段5** 绘图与解释 `plots.py` / `explain.py`：拟合/残差/重要性(原生·置换·SHAP)/SHAP摘要·依赖·瀑布/PDP·ICE/2D PDP/ALE/结果表。
- [x] **阶段6** NiceGUI 前端 `main.py`：①数据(上传+本地路径) ②模型 ③优化 ④图表+保存 ⑤运行(后台线程+进度+日志+画廊)。
- [x] **阶段7（部分）** 冒烟测试 `tests/smoke_core.py`、README、浏览器端到端验证（XGBoost+RF 跑通，输出各 9 项）。

> 验证结果：默认数据集上 XGBoost 测试 R²≈0.79、RF≈0.78；UI 完整渲染、图像/指标/下载正常。

- [x] **配色方案** `themes.py`：参考 nature-figure 技能定义 5 组期刊级配色（Nature 经典蓝红 / Nature 暖冷对比 / NMI 低饱和蓝紫 / 材料青紫 / 临床多色），每组含离散色板 + 连续 colormap，UI"④"可选，统一应用到拟合/重要性/残差/PDP/SHAP/ALE 各图。
- [x] **拟合图升级**：`regression_fit` 改为参考 `plot_regression_fit2` 的五面板布局（主散点 + 真实/预测边际直方图 + 残差散点 + 残差直方图）。

### 后续（二期 / 待办）

- [ ] 核心层单元测试（data/optimize/pipeline 关键路径，pytest）。
- [ ] 长任务的"取消/超时"机制；KernelExplainer 大数据下的性能优化。
- [x] **TabPFN**（二期·首个）：已在 `models.py` 注册（懒加载、自动选 GPU、空搜索空间不调参）。
  - 环境照搬参考 `F:\…\.venv`：Python 3.10 + torch 1.12.1+cu113 + tabpfn 2.2.1，tabpfn-extensions 复制源码到 `vendor/` 以 `--no-deps` 安装；一键脚本 `setup_gpu_env.bat/.sh`。
  - ⚠️ tabpfn/extensions 声明 torch≥2.1，与 1.12.1 冲突，故 `--no-deps` 绕过（与参考环境一致）；装完勿再 `uv sync`（会被裁剪）。
  - ⚠️ 尚未实跑/测试 TabPFN 训练与推理（运行慢，按要求只改代码）。SHAP/PDP 对 TabPFN 很慢，建议不勾选或仅用置换重要性。
- [ ] 二期其余模型：Stacking / ELM / 贝叶斯（轻量，无需 torch）/ TabM / xRFM（需 torch）。
- [ ] 可选：打包为桌面应用、训练好的模型/解释结果的持久化复用。

---

## 六、已确认的决策（2026-06-11）

1. **模型一期范围**：6 树模型（XGBoost/LightGBM/CatBoost/RF/GBR/DT）+ KNN/SVR/MLP，共 **9 个**。TabM/TabPFN/xRFM/Stacking/ELM/Bayesian 列为二期。
2. **运行形态**：本地浏览器 GUI（NiceGUI 默认），不打包桌面 exe。
3. **数据假设**：不固定列含义，**完全由用户在界面指定**目标列 / 特征列 / 分类列 / 不标准化列。
4. **任务类型**：**仅回归**（粘结强度预测）。

> 环境：uv + Python 3.11；沿用参考项目的依赖锁定组合（numpy 1.26.4 / scikit-learn 1.5.1 / xgboost 2.0.0 / catboost 1.2.7 / shap 0.42.1 等），不引入 torch。

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
