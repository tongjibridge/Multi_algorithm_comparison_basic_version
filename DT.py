# 导入所需的库

import pandas as pd  # 导入pandas库，用于数据处理和分析，特别是DataFrame操作
import numpy as np  # 导入numpy库，用于进行数值计算，特别是数组操作
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘制图表
import matplotlib  # 导入matplotlib主库，用于更底层的绘图设置

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    mean_pinball_loss,
)  # 从sklearn导入评估回归模型性能的指标
import joblib  # 导入joblib库，用于模型的保存和加载

# from scipy.stats import gaussian_kde  # 从scipy.stats导入高斯核密度估计，可用于绘制密度图（此脚本中未直接使用，但可能为备用）
# from sklearn.inspection import PartialDependenceDisplay, partial_dependence  # 从sklearn导入部分依赖图的工具（此脚本未使用，改用手动实现）
# import time  # 导入time库，用于计时（此脚本中未直接使用）
import os  # 导入os库，用于操作系统相关功能，如创建文件夹

import warnings  # 导入warnings库，用于控制警告信息的显示
from collections import (
    defaultdict,
)  # 从collections导入defaultdict，用于创建带有默认值的字典


# from scipy.interpolate import griddata  # 从scipy.interpolate导入griddata，用于插值（此脚本中未直接使用）
from matplotlib import font_manager
import optuna
from sklearn.model_selection import KFold
import logging
from tools import (
    plot_regression_fit,
    plot_importance_combined,
    plot_residuals_styled,
    data_norm_get,
)

# --- 全局设置 ---
# 忽略特定类型的警告，避免在输出中显示不必要的警告信息
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils._bunch")
warnings.filterwarnings("ignore", category=UserWarning)
# 解决 DecisionTree 在 GridSearchCV 中可能出现的 verbosity 警告
warnings.filterwarnings(
    "ignore", message="Found 'n_estimators' in params. Will use it instead of argument"
)
matplotlib.use(
    "TkAgg"
)  # 设置matplotlib的后端，'TkAgg'是一个图形界面后端，确保在某些环境下可以正常显示绘图窗口

font_path = "times+simsun.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
# 创建日志器
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

# --- （函数定义区） ---


def objective(trial):
    global index
    param_grid = {
        "max_depth": trial.suggest_int("max_depth", 10, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
    }

    error = 0
    count = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train_scaled_df, X_test_scaled_df = data_norm_get(
            x_train,
            x_test,
            y_train,
            y_test,
            non_standardize_features=non_standardize_features,
        )

        model = DecisionTreeRegressor(**param_grid, random_state=42)
        model.fit(X_train_scaled_df, y_train)
        y_pred = model.predict(X_test_scaled_df)

        # 计算评估指标
        error += mean_pinball_loss(y_test, y_pred)
        count += 1

    return error / count


logger.info(
    "-------------------------------------准备数据---------------------------------------"
)
# 从指定的Excel文件中读取数据
# 注意：请确保文件路径正确无误
df = pd.read_excel(r"./fpr筋机器学习预处理.xlsx")
non_standardize_features = ["B", "C", "G", "带肋", "黏砂", "光圆"]
y = df.iloc[:, -1]  # 提取最后一列作为目标变量y
x = df.iloc[:, :-1]  # 提取从第二列开始的所有列作为特征变量x
feature_names_from_df = x.columns.tolist()  # 获取特征名称列表

logger.info(
    "-------------------------------------划分数据集---------------------------------------"
)
# 将数据集划分为训练集和测试集，测试集占30%，设置随机种子以保证结果可复现
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=45
# )


# --- 【DecisionTree 修改】 ---
logger.info(
    "-------------------------------------定义DecisionTree模型超参数范围---------------------------------------"
)
# 定义要进行网格搜索的DecisionTree超参数范围


logger.info(
    "-------------------------------------搜索最佳超参数---------------------------------------"
)
# 实例化GridSearchCV对象，用于自动寻找最佳超参数组合
sampler = optuna.samplers.CmaEsSampler()
study = optuna.create_study(direction="minimize", sampler=sampler)  # 最小化MAE

study.optimize(objective, n_trials=100, show_progress_bar=True)


logger.info(
    "-------------------------------------输出最佳模型---------------------------------------"
)
# 输出最优参数
print("最优参数:", study.best_params)
logger.info("最佳MAE:", study.best_value)

logger.info(
    "-------------------------------------保存最佳模型---------------------------------------"
)
model_save_dir = r"./savemodel/DecisionTree/"  # 定义模型保存的目录
os.makedirs(model_save_dir, exist_ok=True)  # 创建目录，如果目录已存在则不报错
# --- 【DecisionTree 修改】 ---
model_path = os.path.join(
    model_save_dir, "DecisionTree_model_final.pkl"
)  # 定义模型的完整保存路径
joblib.dump(study, model_path)  # 将找到的最佳模型保存到文件
logger.info(f"模型已保存至: {model_path}")  # 打印保存成功信息
study = joblib.load(model_path)  # 从文件加载模型

kf = KFold(n_splits=5, shuffle=True, random_state=45)
index = kf.split(y)
train_index, test_index = next(index)
x_train, x_test = x.iloc[train_index], x.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
X_train_scaled_df, X_test_scaled_df = data_norm_get(
    x_train, x_test, y_train, y_test, non_standardize_features=non_standardize_features
)


best_model = DecisionTreeRegressor(**study.best_params, random_state=42)
best_model.fit(X_train_scaled_df, y_train)
logger.info(
    "-------------------------------应用模型--------------------------------------"
)
y_test_pred = best_model.predict(X_test_scaled_df)  # 使用加载的模型对测试集进行预测
y_train_pred = best_model.predict(X_train_scaled_df)  # 使用加载的模型对训练集进行预测

results_plot_save_dir = r"./result/DecisionTree/"  # 定义结果图保存的目录
os.makedirs(results_plot_save_dir, exist_ok=True)  # 创建目录，如果目录已存在则不报错
# 将数据写入xlsx表格，其中X_train_scaled_df、y_train、y_train_pred在train表格，X_test_scaled_df、y_test、y_test_pred在test表格
train_df = pd.concat([X_train_scaled_df, y_train, pd.Series(y_train_pred)], axis=1)
train_df.columns = feature_names_from_df + ["y_train", "y_train_pred"]
test_df = pd.concat([X_test_scaled_df, y_test, pd.Series(y_test_pred)], axis=1)
test_df.columns = feature_names_from_df + ["y_test", "y_test_pred"]
with pd.ExcelWriter(
    os.path.join(results_plot_save_dir, "DecisionTree_scaled_results.xlsx")
) as writer:
    train_df.to_excel(writer, sheet_name="train", index=False)
    test_df.to_excel(writer, sheet_name="test", index=False)
# 保存未缩放的数据
train_df = pd.concat([x_train, y_train, pd.Series(y_train_pred)], axis=1)
train_df.columns = feature_names_from_df + ["y_train", "y_train_pred"]
test_df = pd.concat([x_test, y_test, pd.Series(y_test_pred)], axis=1)
test_df.columns = feature_names_from_df + ["y_test", "y_test_pred"]
with pd.ExcelWriter(
    os.path.join(results_plot_save_dir, "DecisionTree_results.xlsx")
) as writer:
    train_df.to_excel(writer, sheet_name="train", index=False)
    test_df.to_excel(writer, sheet_name="test", index=False)


logger.info(
    "-------------------------------------训练模型性能---------------------------------------"
)
train_mse = mean_squared_error(y_train, y_train_pred)  # 计算训练集的均方误差(MSE)
train_rmse = np.sqrt(train_mse)  # 计算训练集的均方根误差(RMSE)
train_mae = mean_absolute_error(y_train, y_train_pred)  # 计算训练集的平均绝对误差(MAE)
train_r2 = r2_score(y_train, y_train_pred)  # 计算训练集的决定系数(R2)
logger.info(
    f"MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}"
)

logger.info(
    "-------------------------------------验证模型性能---------------------------------------"
)
test_mse = mean_squared_error(y_test, y_test_pred)  # 计算测试集的均方误差(MSE)
test_rmse = np.sqrt(test_mse)  # 计算测试集的均方根误差(RMSE)
test_mae = mean_absolute_error(y_test, y_test_pred)  # 计算测试集的平均绝对误差(MAE)
test_r2 = r2_score(y_test, y_test_pred)  # 计算测试集的决定系数(R2)
logger.info(
    f"MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}"
)

logger.info(
    "----------------------------------------结果绘图-----------------------------------------"
)


# --- 【DecisionTree 修改】 ---
train_path = os.path.join(
    results_plot_save_dir, "DecisionTree_训练集精度_final.png"
)  # 训练集拟合图的保存路径
test_path = os.path.join(
    results_plot_save_dir, "DecisionTree_验证集精度_final.png"
)  # 验证集拟合图的保存路径
# 调用函数绘制训练集的拟合图
plot_regression_fit(
    y_train,
    y_train_pred,
    train_r2,
    train_rmse,
    train_mae,
    "Train Set",
    "DecisionTree Model Performance (Train Set)",
    train_path,
)
# 调用函数绘制测试集的拟合图
plot_regression_fit(
    y_test,
    y_test_pred,
    test_r2,
    test_rmse,
    test_mae,
    "Test Set",
    "DecisionTree Model Performance (Test Set)",
    test_path,
)
plt.rcdefaults()  # 恢复matplotlib的默认设置

# --- 【DecisionTree 修改】 ---
logger.info(
    "----------------------------------------计算并绘制DecisionTree原生特征重要性图-----------------------------------------"
)

# importances = best_model.feature_importances_
# # 创建一个包含特征名称和重要性分数的DataFrame
# gbdt_importance_df = pd.DataFrame(
#     {"Feature": feature_names_from_df, "Importance": importances}
# )
# save_path_gbdt = os.path.join(
#     results_plot_save_dir, "DecisionTree_特征重要性组合图_final.png"
# )  # 定义保存路径
# # 调用函数绘制组合特征重要性图
# plot_importance_combined(
#     gbdt_importance_df,
#     "DecisionTree模型计算的特征重要性",
#     save_path_gbdt,
#     bar_color="dodgerblue",
# )

logger.info(
    "----------------------------------------计算并绘制Permutation Importance图-----------------------------------------"
)
scores = defaultdict(
    list
)  # 创建一个默认值为列表的字典，用于存储每个特征的置换重要性分数
# 遍历每一个特征
for feat_name in feature_names_from_df:
    X_t = X_test_scaled_df.copy()  # 复制一份测试集数据
    # 随机打乱当前特征列的顺序
    X_t[feat_name] = np.random.permutation(X_t[feat_name].values)
    # 计算打乱后模型的R2分数
    shuff_acc = r2_score(y_test, best_model.predict(X_t))
    # 计算重要性：(原始R2 - 打乱后R2) / 原始R2，如果原始R2接近0则直接用差值
    scores[feat_name].append(
        (test_r2 - shuff_acc) / test_r2 if test_r2 > 1e-6 else test_r2 - shuff_acc
    )
# 对特征按重要性得分从高到低排序
sorted_scores = sorted(
    [(np.mean(score_list), feat) for feat, score_list in scores.items()], reverse=True
)
perm_feature_names = [feat for _, feat in sorted_scores]  # 获取排序后的特征名称
perm_feature_scores = [score for score, _ in sorted_scores]  # 获取排序后的重要性分数
# 创建一个包含置换重要性结果的DataFrame
perm_importance_df = pd.DataFrame(
    {"Feature": perm_feature_names, "Importance": perm_feature_scores}
)
# --- 【DecisionTree 修改】 ---
save_path_perm = os.path.join(
    results_plot_save_dir, "DecisionTree_特征重要性_Permutation_final.png"
)  # 定义保存路径
# 调用函数绘制组合特征重要性图（使用置换重要性数据）
plot_importance_combined(
    perm_importance_df,
    "特征重要性 (Permutation Importance for DecisionTree)",
    save_path_perm,
    bar_color="lightcoral",
)

logger.info(
    "----------------------------------------绘制残差分析图-----------------------------------------"
)
train_residuals = y_train - y_train_pred  # 计算训练集的残差
test_residuals = y_test - y_test_pred  # 计算测试集的残差
# --- 【DecisionTree 修改】 ---
train_res_path = os.path.join(
    results_plot_save_dir, "DecisionTree_训练集残差分析图_final.png"
)  # 训练集残差图保存路径
test_res_path = os.path.join(
    results_plot_save_dir, "DecisionTree_验证集残差分析图_final.png"
)  # 测试集残差图保存路径
# 调用函数绘制训练集残差图
plot_residuals_styled(
    train_residuals, y_train_pred, train_res_path, "DecisionTree 训练集残差分析"
)
# 调用函数绘制测试集残差图
plot_residuals_styled(
    test_residuals, y_test_pred, test_res_path, "DecisionTree 验证集残差分析"
)

# # =================================================================================
# # ============ 使用手动计算方法绘制 PDP 和 ICE 相关图 (适用于DecisionTree) ============
# # =================================================================================
# logger.info(
#     "------------------------开始 PDP 和 ICE 相关绘图 (手动实现)------------------------"
# )
# # 定义PDP/ICE图的保存目录
# # --- 【DecisionTree 修改】 ---
# pdp_ice_save_dir = os.path.join(results_plot_save_dir, "DecisionTree_PDP_ICE_Plots_final")
# os.makedirs(pdp_ice_save_dir, exist_ok=True)  # 创建目录
# # 定义双变量PDP图的保存目录
# pdp_2way_save_dir = os.path.join(pdp_ice_save_dir, "2Way_PDP_All_Combinations")
# os.makedirs(pdp_2way_save_dir, exist_ok=True)  # 创建目录
# # 定义3D PDP图的保存目录
# pdp_3d_save_dir = os.path.join(pdp_ice_save_dir, "3D_PDP_All_Combinations")
# os.makedirs(pdp_3d_save_dir, exist_ok=True)  # 创建目录

# n_top_features_for_pdp = 11  # 设置用于PDP分析的最重要特征的数量
# if n_top_features_for_pdp > len(
#     feature_names_from_df
# ):  # 如果特征总数不足，则取全部特征
#     n_top_features_for_pdp = len(feature_names_from_df)
# # 根据DecisionTree置换重要性排序，选取最重要的N个特征
# top_features_pdp_names = perm_importance_df["Feature"].tolist()[:n_top_features_for_pdp]
# plt.style.use("seaborn-v0_8-whitegrid")  # 设置绘图风格
# plt.rc("font", family="Microsoft YaHei")  # 设置中文字体

# # --- 1. 绘制单变量 PDP (含置信区间) 和 ICE 组合图 ---
# logger.info("\n开始绘制单变量 PDP (含95%置信区间) 和 ICE 组合图...")
# # 遍历最重要的N个特征
# for feature_name in top_features_pdp_names:
#     logger.info(f"正在计算特征 '{feature_name}' 的PDP/ICE数据...")
#     try:
#         # 使用手动编写的函数计算1D PDP和ICE数据
#         grid_vals, pdp_vals, ice_lines_vals = manual_pdp_1d(
#             best_model, X_train_scaled_df, feature_name
#         )

#         # 在每个网格点上计算所有ICE线的标准差，用于构建置信区间
#         pdp_std = np.std(ice_lines_vals, axis=0)

#         # 开始绘图
#         fig, ax = plt.subplots(figsize=(10, 8))

#         # 绘制所有样本的ICE线 (半透明蓝色细线)
#         for ice_line in ice_lines_vals:
#             ax.plot(grid_vals, ice_line, color="tab:blue", alpha=0.05, linewidth=0.5)

#         # 绘制PDP线 (红色虚线)，代表平均效应
#         ax.plot(
#             grid_vals,
#             pdp_vals,
#             color="red",
#             linestyle="--",
#             linewidth=3,
#             label="平均效应 (PDP)",
#         )

#         # 绘制95%置信区间 (平均值 ± 1.96 * 标准差)
#         ax.fill_between(
#             grid_vals,
#             pdp_vals - 1.96 * pdp_std,
#             pdp_vals + 1.96 * pdp_std,
#             color="skyblue",
#             alpha=0.4,
#             label="95% 置信区间",
#         )

#         ax.set_title(f"PDP/ICE 组合图\n特征: {feature_name}", fontsize=16)  # 设置标题
#         ax.set_xlabel(f"{feature_name} (标准化值)", fontsize=12)  # 设置x轴标签
#         ax.set_ylabel("对预测值的依赖性", fontsize=12)  # 设置y轴标签
#         ax.legend()  # 显示图例
#         # 保存图表
#         # --- 【DecisionTree 修改】 ---
#         plt.savefig(
#             os.path.join(pdp_ice_save_dir, f"DecisionTree_Manual_PDP_ICE_{feature_name}.png"),
#             dpi=300,
#             bbox_inches="tight",
#         )
#         plt.close(fig)  # 关闭图表
#         logger.info(f"成功绘制特征 '{feature_name}' 的PDP/ICE图。")

#     except Exception as e:
#         logger.info(f"绘制手动 PDP/ICE for {feature_name} 出错: {e}")  # 打印错误信息

# # --- 2. 绘制双变量 (2D 和 3D) PDP 图 ---
# logger.info("\n开始绘制双变量 PDP (2D 热力图 和 3D 曲面图)...")
# if len(top_features_pdp_names) >= 2:  # 确保至少有两个特征可以进行组合
#     # 遍历最重要的N个特征中的所有两两组合
#     for feat1, feat2 in combinations(top_features_pdp_names, 2):
#         logger.debug(f"正在计算特征对 '{feat1}' vs '{feat2}' 的2D PDP数据...")
#         try:
#             # 使用手动编写的函数计算2D PDP数据
#             grid_x, grid_y, pdp_z = manual_pdp_2d(
#                 best_model, X_train_scaled_df, (feat1, feat2)
#             )

#             # 创建用于绘图的网格坐标
#             XX, YY = np.meshgrid(grid_x, grid_y)
#             # 注意：pdp_z的维度可能需要转置以匹配meshgrid的坐标系
#             ZZ = pdp_z.T

#             # 绘制 2D 热力图
#             fig_2d, ax_2d = plt.subplots(figsize=(8, 7))
#             # 使用contourf填充等值线图
#             c = ax_2d.contourf(XX, YY, ZZ, cmap="viridis", levels=20)
#             fig_2d.colorbar(c, ax=ax_2d, label="部分依赖值")  # 添加颜色条
#             ax_2d.set_title(f"2D PDP: {feat1} vs {feat2}", fontsize=16)  # 设置标题
#             ax_2d.set_xlabel(f"{feat1} (标准化值)", fontsize=12)  # 设置x轴标签
#             ax_2d.set_ylabel(f"{feat2} (标准化值)", fontsize=12)  # 设置y轴标签
#             # --- 【DecisionTree 修改】 ---
#             plt.savefig(
#                 os.path.join(
#                     pdp_2way_save_dir, f"DecisionTree_Manual_PDP_2D_{feat1}_{feat2}.png"
#                 ),
#                 dpi=300,
#             )  # 保存
#             plt.close(fig_2d)  # 关闭图表

#             # 绘制 3D 曲面图
#             fig_3d = plt.figure(figsize=(12, 9))
#             ax_3d = fig_3d.add_subplot(111, projection="3d")  # 创建3D子图
#             # 绘制3D曲面
#             surf = ax_3d.plot_surface(
#                 XX, YY, ZZ, cmap="viridis", edgecolor="none", antialiased=True
#             )
#             fig_3d.colorbar(
#                 surf, shrink=0.5, aspect=20, label="部分依赖值", pad=0.1
#             )  # 添加颜色条
#             ax_3d.set_xlabel(f"{feat1} (标准化值)", fontsize=10, labelpad=10)  # x轴标签
#             ax_3d.set_ylabel(f"{feat2} (标准化值)", fontsize=10, labelpad=10)  # y轴标签
#             ax_3d.set_zlabel(
#                 "对预测值的依赖性 (PDP)", fontsize=10, labelpad=10, rotation=180
#             )  # z轴标签
#             ax_3d.set_title(
#                 f"三维部分依赖图 (3D PDP)\n{feat1} vs {feat2}", fontsize=14
#             )  # 标题
#             ax_3d.view_init(elev=20, azim=45)  # 设置视角
#             # --- 【DecisionTree 修改】 ---
#             plt.savefig(
#                 os.path.join(pdp_3d_save_dir, f"DecisionTree_Manual_PDP_3D_{feat1}_{feat2}.png"),
#                 dpi=300,
#             )  # 保存
#             plt.close(fig_3d)  # 关闭图表

#             logger.info(f"成功绘制特征对 '{feat1}' vs '{feat2}' 的2D和3D PDP图。")

#         except Exception as e:
#             logger.info(
#                 f"绘制手动 2D/3D PDP for {feat1} & {feat2} 出错: {e}"
#             )  # 打印错误信息

# # --- 绘制三特征3D散点图 ---
# logger.info("\n开始绘制三特征 (3D) 散点图...")
# pdp_3d_scatter_save_dir = os.path.join(
#     pdp_ice_save_dir, "3D_Scatter_Three_Features"
# )  # 定义保存目录
# os.makedirs(pdp_3d_scatter_save_dir, exist_ok=True)  # 创建目录

# if len(top_features_pdp_names) >= 3:  # 确保至少有3个特征
#     # 最多选择前4个重要特征进行组合，避免组合数过多
#     n_features_for_3d_scatter = min(len(top_features_pdp_names), 4)

#     # 遍历所有三个特征的组合
#     for features_tuple in combinations(
#         top_features_pdp_names[:n_features_for_3d_scatter], 3
#     ):
#         try:
#             # 定义保存路径
#             # --- 【DecisionTree 修改】 ---
#             save_path = os.path.join(
#                 pdp_3d_scatter_save_dir,
#                 f"DecisionTree_3D_Scatter_{features_tuple[0]}_{features_tuple[1]}_{features_tuple[2]}.png",
#             )
#             # 调用函数绘制3D散点图
#             plot_3d_scatter_three_features(
#                 X_test_scaled_df, y_test_pred, features_tuple, save_path
#             )
#         except Exception as e:
#             logger.info(
#                 f"绘制 3D 散点图 for {features_tuple} 出错: {e}"
#             )  # 打印错误信息

# # --- 调用：绘制固定特征值的3D PDP图 ---
# logger.info("\n开始绘制固定特征值的3D PDP图...")
# pdp_3d_fixed_save_dir = os.path.join(
#     pdp_ice_save_dir, "3D_PDP_Fixed_Value"
# )  # 定义保存目录
# os.makedirs(pdp_3d_fixed_save_dir, exist_ok=True)  # 创建目录

# if len(top_features_pdp_names) >= 3:  # 确保至少有3个特征
#     # 最多选择前4个重要特征进行组合
#     n_features_for_3d_fixed = min(len(top_features_pdp_names), 4)

#     # 遍历所有三个特征的组合
#     for features_tuple in combinations(
#         top_features_pdp_names[:n_features_for_3d_fixed], 3
#     ):
#         # 对每个组合，轮流固定其中的一个特征
#         for feature_to_fix in features_tuple:
#             try:
#                 features_list = list(features_tuple)  # 元组转列表
#                 # 获取另外两个变化的特征
#                 varying_feats = [f for f in features_list if f != feature_to_fix]
#                 # 定义保存路径
#                 # --- 【DecisionTree 修改】 ---
#                 save_path = os.path.join(
#                     pdp_3d_fixed_save_dir,
#                     f"DecisionTree_3DPDP_{varying_feats[0]}_{varying_feats[1]}_Fix_{feature_to_fix}.png",
#                 )
#                 # 将固定的值设为该特征的中位数
#                 fixed_val = X_train_scaled_df[feature_to_fix].median()

#                 # 调用函数绘制固定特征值的3D PDP图
#                 plot_3d_pdp_fixed_value(
#                     best_model,
#                     X_train_scaled_df,
#                     features_list,
#                     save_path,
#                     fixed_feature=feature_to_fix,
#                     fixed_value=fixed_val,
#                 )
#             except Exception as e:
#                 logger.info(
#                     f"绘制固定值3D PDP for {features_list} (固定 {feature_to_fix}) 出错: {e}"
#                 )  # 打印错误信息

# # --- 【DecisionTree 修改】 ---
# logger.info("------------------------开始 SHAP 分析 (DecisionTree)------------------------")
# shap_save_dir = os.path.join(
#     results_plot_save_dir, "DecisionTree_SHAP_Plots_final"
# )  # 定义SHAP图的保存目录
# os.makedirs(shap_save_dir, exist_ok=True)  # 创建目录
# # shap.TreeExplainer 同样适用于 DecisionTree 模型
# explainer = shap.TreeExplainer(best_model)
# shap_values = explainer(X_test_scaled_df)  # 计算测试集所有样本的SHAP值

# logger.info("\n绘制 SHAP Summary Plot (条形图)...")
# # 计算每个特征的平均绝对SHAP值，作为其重要性
# shap_importance_vals = np.abs(shap_values.values).mean(axis=0)
# # 创建包含SHAP重要性的DataFrame
# shap_importance_df = pd.DataFrame(
#     {"Feature": X_test_scaled_df.columns, "Importance": shap_importance_vals}
# )
# save_path_shap = os.path.join(
#     shap_save_dir, "DecisionTree_SHAP_特征重要性组合图_final.png"
# )  # 定义保存路径
# # 调用组合重要性绘图函数，绘制SHAP重要性条形图
# plot_importance_combined(
#     shap_importance_df,
#     "SHAP 特征重要性 (平均绝对SHAP值)",
#     save_path_shap,
#     bar_color="#007bff",
# )

# logger.info("绘制 SHAP Summary Plot (散点分布图)...")
# shap.summary_plot(
#     shap_values, X_test_scaled_df, show=False
# )  # 生成SHAP摘要图（散点形式）
# plt.title("SHAP 特征影响概览 (散点分布)", fontsize=16)  # 添加标题
# plt.tight_layout()  # 调整布局
# plt.savefig(
#     os.path.join(shap_save_dir, "DecisionTree_SHAP_summary_scatter.png"),
#     dpi=300,
#     bbox_inches="tight",
# )  # 保存
# plt.close()  # 关闭图表

# logger.info("绘制 SHAP Dependence Plots...")
# shap_dependence_save_dir = os.path.join(
#     shap_save_dir, "Dependence_Plots"
# )  # 定义SHAP依赖图的保存目录
# os.makedirs(shap_dependence_save_dir, exist_ok=True)  # 创建目录
# # 为最重要的N个特征绘制SHAP依赖图
# for feature_name in top_features_pdp_names:
#     # 绘制单个特征的依赖图，图中颜色表示交互效应最强的另一个特征
#     shap.dependence_plot(
#         feature_name,
#         shap_values.values,
#         X_test_scaled_df,
#         interaction_index="auto",
#         show=False,
#     )
#     plt.gcf().suptitle(f"SHAP 依赖图: {feature_name}", fontsize=16)  # 添加总标题
#     plt.tight_layout()  # 调整布局
#     plt.savefig(
#         os.path.join(
#             shap_dependence_save_dir, f"DecisionTree_SHAP_dependence_{feature_name}.png"
#         ),
#         dpi=300,
#         bbox_inches="tight",
#     )  # 保存图表
#     plt.close()  # 关闭图表

# logger.info("绘制 SHAP Waterfall Plot (针对测试集第一个样本)...")
# plt.figure()  # 创建一个新的画布
# # 绘制瀑布图，展示单个预测（这里是测试集第一个样本）的SHAP值构成
# shap.plots.waterfall(shap_values[0], max_display=15, show=False)
# plt.title("SHAP Waterfall Plot (测试集样本 0)", fontsize=16)  # 添加标题
# plt.tight_layout()  # 调整布局
# plt.savefig(
#     os.path.join(shap_save_dir, "DecisionTree_SHAP_waterfall_sample_0.png"),
#     dpi=300,
#     bbox_inches="tight",
# )  # 保存
# plt.close()  # 关闭图表
# logger.info(
#     "----------------------------------------SHAP 分析完成-----------------------------------------"
# )

# # --- 【DecisionTree 修改】 ---
# logger.info(
#     "----------------------------------------开始 ALE 分析 (DecisionTree)-----------------------------------------"
# )
# ale_save_dir = os.path.join(
#     results_plot_save_dir, "DecisionTree_ALE_Plots_final"
# )  # 定义ALE图的保存目录
# os.makedirs(ale_save_dir, exist_ok=True)  # 创建目录
# logger.info(f"ALE 相关图将保存到: {ale_save_dir}")
# top_features_ale_names = top_features_pdp_names  # 使用与PDP相同的最重要特征列表

# logger.info(f"\n开始为最重要的 {len(top_features_ale_names)} 个特征绘制一维 ALE 图...")
# # PyALE 同样支持 DecisionTree 模型
# colors = plt.cm.viridis(np.linspace(0, 0.85, len(top_features_ale_names)))

# # 遍历最重要的特征
# for i, feature_name in enumerate(top_features_ale_names):
#     try:
#         # 使用PyALE库计算并绘制一维ALE图
#         ale_eff = ale(
#             X=X_train_scaled_df,
#             model=best_model,
#             feature=[feature_name],
#             feature_type="continuous",
#             grid_size=50,
#             include_CI=True,
#             C=0.95,
#         )
#         fig, ax = plt.gcf(), plt.gca()  # 获取当前的图和坐标轴
#         current_color = colors[i]  # 为当前特征选择一个颜色
#         if ax.lines:  # 如果图中有线（ALE主线）
#             ax.lines[0].set_color(current_color)  # 设置线的颜色
#             ax.lines[0].set_linewidth(2.5)  # 设置线的宽度
#         if ax.collections:  # 如果图中有集合（置信区间）
#             ax.collections[0].set_facecolor(current_color)  # 设置填充颜色
#             ax.collections[0].set_alpha(0.2)  # 设置透明度
#         ax.set_title(
#             f"累积局部效应 (ALE) - 特征: {feature_name}", fontsize=16
#         )  # 设置标题
#         ax.set_xlabel(f"{feature_name} (标准化值)", fontsize=12)  # 设置x轴标签
#         ax.set_ylabel("ALE (对预测值的影响)", fontsize=12)  # 设置y轴标签
#         plt.tight_layout()  # 调整布局
#         plt.savefig(
#             os.path.join(ale_save_dir, f"DecisionTree_ALE_1D_{feature_name}.png"),
#             dpi=300,
#             bbox_inches="tight",
#         )  # 保存
#         plt.close(fig)  # 关闭图表
#     except Exception as e:
#         logger.info(f"绘制 1D ALE for {feature_name} 出错: {e}")  # 打印错误信息
#         if plt.get_fignums():  # 如果有未关闭的图表
#             plt.close("all")  # 全部关闭

# logger.info("\n开始为最重要的特征对绘制二维 ALE 图...")
# if len(top_features_ale_names) >= 2:  # 确保至少有两个特征
#     # 遍历所有两两特征组合
#     for feat1_name, feat2_name in combinations(top_features_ale_names, 2):
#         try:
#             # 计算二维ALE效应，但不立即绘图 (plot=False)
#             ale_eff_2d = ale(
#                 X=X_train_scaled_df,
#                 model=best_model,
#                 feature=[feat1_name, feat2_name],
#                 feature_type="continuous",
#                 grid_size=30,
#                 plot=False,
#             )
#             fig, ax = plt.subplots(figsize=(8, 7))  # 创建画布
#             # 使用pcolormesh绘制二维ALE热力图
#             im = ax.pcolormesh(
#                 ale_eff_2d.index,
#                 ale_eff_2d.columns,
#                 ale_eff_2d.values.T,
#                 cmap="viridis",
#                 shading="auto",
#             )
#             fig.colorbar(im, ax=ax, label="ALE (对预测值的影响)")  # 添加颜色条
#             ax.set_title(
#                 f"二维 ALE: {feat1_name} vs {feat2_name}", fontsize=16
#             )  # 设置标题
#             ax.set_xlabel(f"{feat1_name} (标准化值)", fontsize=12)  # 设置x轴标签
#             ax.set_ylabel(f"{feat2_name} (标准化值)", fontsize=12)  # 设置y轴标签
#             plt.tight_layout()  # 调整布局
#             plt.savefig(
#                 os.path.join(
#                     ale_save_dir, f"DecisionTree_ALE_2D_{feat1_name}_vs_{feat2_name}.png"
#                 ),
#                 dpi=300,
#                 bbox_inches="tight",
#             )  # 保存图表
#             plt.close(fig)  # 关闭图表
#         except Exception as e:
#             logger.info(
#                 f"绘制 2D ALE for {feat1_name} & {feat2_name} 出错: {e}"
#             )  # 打印错误信息
#             if plt.get_fignums():  # 如果有未关闭的图表
#                 plt.close("all")  # 全部关闭

# logger.info(
#     "----------------------------------------ALE 分析完成-----------------------------------------"
# )
# logger.info(
#     "----------------------------------------脚本执行完毕-----------------------------------------"
# )
