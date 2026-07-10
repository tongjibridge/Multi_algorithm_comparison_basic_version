# 可解释机器学习 GUI（ExplainableML）

基于 [NiceGUI](https://nicegui.io/) 的本地可解释机器学习程序：**导入 Excel → 选择模型 → 选择参数优化方法 → 勾选输出图表 → 一键训练并生成可解释性分析**。

源自 `Multi_algorithm_comparison_basic_version/`（多算法预测 FRP 筋-混凝土粘结强度的实践）的重构与通用化，把原本 15 个高度重复的脚本抽象为「统一管道 + 模型注册表 + 可选优化方法 + 可选输出」。

> 仅回归任务；列含义完全由用户在界面指定。

## 功能（对应需求）

- **R1 选择模型（可多选）**：XGBoost、LightGBM、CatBoost、随机森林、梯度提升 GBR、决策树、KNN、SVR、MLP，共 9 个，支持横向对比。
- **R2 导入 Excel**：上传 `.xlsx` 或填本地路径；自定义目标列 / 特征列 / 分类列 / 不标准化列；自动检测分类列并预览数据。
- **R3 参数优化方法**：Optuna（TPE / 随机 / CMA-ES）、网格搜索、随机搜索、粒子群 PSO（mealpy）、手动默认参数；统一 K 折交叉验证、无数据泄漏。
- **R4 输出图表与保存**：勾选回归拟合图、残差图、特征重要性（原生 / 置换 / SHAP）、SHAP 摘要 / 依赖 / 瀑布、PDP·ICE、2D PDP、ALE、预测结果表；**5 组期刊级配色方案**（参考 nature-figure，离散色 + 连续 colormap 统一应用）；自定义保存目录、图片格式（png/svg/pdf）、DPI、Top-K。

## 环境与安装

需要 [uv](https://docs.astral.sh/uv/) 与 Python 3.10（本项目固定 3.10，与 TabPFN 的 GPU 环境对齐）。

```bash
uv sync
```

> ⚠️ 本机注意：若 uv 默认缓存（`C:\Users\…\AppData\Local\uv`，软链到 D 盘）报
> "too many temporary files" 或 os error 183，请改用项目同盘缓存：
> `UV_CACHE_DIR=E:/uv-cache uv sync`。

### 可选：启用 TabPFN（GPU）

TabPFN 需要 torch GPU 版，环境照搬参考机器（torch 1.12.1+cu113）。一键安装：

```bash
./setup_gpu_env.sh        # 或双击 setup_gpu_env.bat（Windows）
```

脚本会重建 Python 3.10 环境、装 torch(cu113) 与 tabpfn，并以源码方式安装从参考机复制到 `vendor/` 的 tabpfn-extensions。
> 注意：tabpfn/extensions 声明 torch≥2.1，与 1.12.1 冲突，脚本用 `--no-deps` 绕过；**装完请勿再单独 `uv sync`**（会删除手动安装的 torch/tabpfn），用 `start` 启动即可。TabPFN 推理较慢，SHAP/PDP 建议不勾选。

## 运行

**最简单**：双击根目录 `start.bat`（Windows），或在 bash 中执行 `./start.sh`。
脚本会在首次运行时自动 `uv sync` 安装依赖，之后直接用虚拟环境解释器启动（不触碰可能损坏的 uv 默认缓存）。

也可手动启动：

```bash
uv run python -m app.main
# 或直接用虚拟环境解释器： .venv/Scripts/python.exe -m app.main
```

启动后浏览器会自动打开 <http://localhost:8080>，按 ①→⑤ 操作即可。

## 使用流程

1. **① 数据导入**：上传 Excel 或填本地路径 → 自动预览、识别分类列 → 按需调整目标/特征/分类/不标准化列、测试集比例、随机种子。
2. **② 模型选择**：勾选一个或多个模型。
3. **③ 参数优化**：选方法、预算（trials/迭代）、CV 折数、评分（RMSE/MAE/R²）。
4. **④ 输出图表与保存**：勾选所需图表，设置保存目录、图片格式、DPI、Top-K。
5. **⑤ 运行**：实时进度与日志；完成后查看指标对比表、图像画廊（可下载），或「打开输出目录」。

## 项目结构

```
app/
├── main.py            # NiceGUI 入口与页面
├── core/
│   ├── data.py        # Excel 读取、列配置、编码、标准化、划分（Preprocessor 防泄漏）
│   ├── space.py       # 声明式搜索空间 → 派生 Optuna/Grid/mealpy
│   ├── models.py      # 9 个模型的注册表（工厂、默认参数、空间、能力）
│   ├── optimize.py    # 优化方法：Optuna/Grid/Random/PSO/Manual
│   ├── pipeline.py    # 统一训练管道（划分→优化→训练→评估）
│   ├── metrics.py     # MSE/RMSE/MAE/R²
│   ├── plots.py       # 绘图（Agg 后端，中文字体），由 tools.py 重构
│   └── explain.py     # 按勾选与模型能力分派生成输出（含 SHAP 全套）
├── ui/state.py        # 后台训练任务与跨线程状态
└── assets/            # 中文字体 times+simsun.ttf

tests/smoke_core.py    # 核心管道冒烟测试
todo.md                # 开发计划与进度
```

## 扩展 mealpy 优化器

mealpy 优化器集中在 `app/core/optimize.py` 注册。界面会直接读取其中的
`METHODS_BY_FAMILY`，因此新增算法通常不需要修改 `app/main.py`，也不需要为每个模型
重新定义搜索空间；`app/core/space.py` 会把模型已有的声明式搜索空间转换为 mealpy 边界。

下面以新增 WOA（鲸鱼优化算法）为例。

### 1. 注册界面选项

在 `METHODS_BY_FAMILY["mealpy"]` 中增加一项。内部标识必须以 `mealpy_` 开头，
这样 `optimize()` 才会自动分派到 mealpy 执行路径：

```python
"mealpy": [
    # ...已有算法
    ("mealpy_woa", "WOA 鲸鱼优化"),
],
```

### 2. 注册日志显示名称

在 `_MEALPY_ALGORITHM_NAMES` 中加入同一个内部标识：

```python
_MEALPY_ALGORITHM_NAMES = {
    # ...已有算法
    "mealpy_woa": "WOA",
}
```

### 3. 创建优化器实例

在 `_build_mealpy_optimizer()` 中增加对应分支：

```python
if method == "mealpy_woa":
    from mealpy import WOA
    return WOA.OriginalWOA(epoch=epoch, pop_size=pop)
```

保存后，重启程序即可在“mealpy”算法列表中看到 WOA。项目会继续复用统一的 K 折
交叉验证目标函数、参数解码和最优参数返回逻辑。

不同 mealpy 算法的模块名、类名和必填构造参数可能不同，请以项目当前安装版本的
mealpy API 为准。如果算法还需要额外参数，就在 `_build_mealpy_optimizer()` 的对应
分支中一并传入。若内部标识不以 `mealpy_` 开头，还需要额外修改 `optimize()` 的分派条件，
因此推荐始终遵循 `mealpy_<算法名>` 的命名规则。

完成后可先验证实例是否能创建，再运行核心冒烟测试：

```powershell
uv run python -c "from app.core.optimize import _build_mealpy_optimizer; print(_build_mealpy_optimizer('mealpy_woa', epoch=5, pop=8))"
uv run python tests/smoke_core.py
```

最后在界面中选择新算法，以较小的优化预算运行一次，确认日志中显示正确名称且能返回
最优参数。mealpy 执行异常时，当前实现会记录警告并回退到 Optuna TPE，验证时应同时检查
运行日志，避免把回退结果误认为新算法已经正常运行。

## 测试

```bash
uv run python tests/smoke_core.py
```

会用参考数据集跑通「训练 → 评估 → 出图」并校验输出文件存在。

## 备注

- 绘图使用 `Agg` 后端，图存文件后在界面展示；中文字体来自 `app/assets/times+simsun.ttf`。
- 非树模型（KNN/SVR/MLP）的 SHAP 用 `KernelExplainer`（较慢，已对背景/样本下采样）；树模型用 `TreeExplainer`。
- 二期可扩展：TabM / TabPFN / xRFM / Stacking / ELM / 贝叶斯（需 torch 等较重依赖）。
