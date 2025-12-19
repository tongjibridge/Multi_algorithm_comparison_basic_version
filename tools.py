# 导入所需的库

import pandas as pd  # 导入pandas库，用于数据处理和分析，特别是DataFrame操作
import numpy as np  # 导入numpy库，用于进行数值计算，特别是数组操作
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘制图表
import matplotlib  # 导入matplotlib主库，用于更底层的绘图设置
import os
from sklearn.preprocessing import StandardScaler  # 从sklearn导入数据标准化工具
import warnings  # 导入warnings库，用于控制警告信息的显示
import logging  # 导入logging库，用于记录日志信息
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# from scipy.interpolate import griddata  # 从scipy.interpolate导入griddata，用于插值（此脚本中未直接使用）
from matplotlib import font_manager
import pickle
import tqdm
import shap
import matplotlib.gridspec as gridspec
from pygam import LinearGAM, s

# --- 全局设置 ---
# 忽略特定类型的警告，避免在输出中显示不必要的警告信息
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils._bunch")
warnings.filterwarnings("ignore", category=UserWarning)

matplotlib.use(
    "TkAgg"
)  # 设置matplotlib的后端，'TkAgg'是一个图形界面后端，确保在某些环境下可以正常显示绘图窗口
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
font_path = "times+simsun.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
logger_tool = logging.getLogger("logger_tool")
logger_tool.setLevel(logging.INFO)
# --- （函数定义区） ---


# 定义一个函数，用于绘制回归模型的拟合效果图
def plot_regression_fit(
    y_true, y_pred, r2, rmse, mae, data_label_en, title_en, save_path
):
    """
    绘制真实值与预测值的散点图，并显示模型评估指标。
    y_true: 真实值
    y_pred: 预测值
    r2: R-squared值
    rmse: 均方根误差
    mae: 平均绝对误差
    data_label_en: 数据集标签 (如 'Train Set')
    title_en: 图表标题
    save_path: 图片保存路径
    """
    plt.style.use("seaborn-v0_8-whitegrid")  # 使用预设的绘图风格
    font_options = {
        "family": "sans-serif",  # 设置字体家族
        "sans-serif": prop.get_name(),  # 设置字体
        "size": 12,  # 设置字体大小
    }
    plt.rc("font", **font_options)  # 设置字体为无衬线字体，以获得更好的显示效果
    fig, ax = plt.subplots(figsize=(7, 7))  # 创建一个7x7英寸的画布和子图
    # 绘制真实值 vs 预测值的散点图
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.6,
        edgecolors="k",
        label=f"{data_label_en} (n={len(y_true)})",
        color="#4D4DFF",
    )
    # 计算并设置坐标轴的范围，确保1:1线能完整显示
    # lims = [
    #     np.min([y_true.min(), y_pred.min()]) - 5,
    #     np.max([y_true.max(), y_pred.max()]) + 5,
    # ]
    lims = [
        0,
        np.max([y_true.max(), y_pred.max()]) + 5,
    ]
    # 绘制1:1参考线 (y=x)，代表完美预测
    ax.plot(lims, lims, "r-", alpha=0.75, zorder=0)
    limsy = [lim * 1.2 for lim in lims]
    ax.plot(lims, limsy, "k-.", alpha=0.75, zorder=0)
    limsy = [lim * 0.8 for lim in lims]
    ax.plot(lims, limsy, "k-.", alpha=0.75, zorder=0)
    limsy = [lim * 1.1 for lim in lims]
    ax.plot(lims, limsy, "k--", alpha=0.75, zorder=0)
    limsy = [lim * 0.9 for lim in lims]
    ax.plot(lims, limsy, "k--", alpha=0.75, zorder=0)

    ax.set_aspect("equal")  # 设置x和y轴的比例相等
    ax.set_xlim(lims)  # 设置x轴范围
    ax.set_ylim(lims)  # 设置y轴范围
    # y_true_np = np.array(y_true)  # 将真实值转换为numpy数组
    # y_pred_np = np.array(y_pred)  # 将预测值转换为numpy数组
    # m, b = np.polyfit(y_true_np, y_pred_np, 1)  # 对散点进行线性拟合，得到斜率m和截距b
    # ax.plot(y_true_np, m * y_true_np + b, "r-", label="Linear Fit")  # 绘制线性拟合线
    ax.set_xlabel("True Values (%)", fontsize=12)  # 设置x轴标签
    ax.set_ylabel("Predicted Values (%)", fontsize=12)  # 设置y轴标签
    ax.set_title(title_en, fontsize=14, weight="bold")  # 设置图表标题
    # 准备要在图上显示的评估指标文本
    metrics_text = f"$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$\n$MAE = {mae:.3f}$"
    # 在图的左上角添加文本框，显示评估指标
    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    ax.legend(loc="lower right")  # 在右下角显示图例
    fig.savefig(save_path, dpi=200, bbox_inches="tight")  # 保存图表到指定路径
    plt.close(fig)  # 关闭图表，释放内存


# 定义一个函数，用于绘制组合特征重要性图（条形图+甜甜圈图）
def plot_importance_combined(df_importance, title, save_path, bar_color="dodgerblue"):
    """
    绘制特征重要性条形图，并在右下角嵌入一个显示Top-N特征占比的甜甜圈图。
    df_importance: 包含'Feature'和'Importance'两列的DataFrame
    title: 图表标题
    save_path: 图片保存路径
    bar_color: 条形图的颜色
    """
    df_importance_sorted = df_importance.sort_values(
        by="Importance", ascending=True
    )  # 按重要性升序排序
    font_options = {
        "family": "sans-serif",  # 设置字体家族
        "sans-serif": prop.get_name(),  # 设置字体
        "size": 18,  # 设置字体大小
    }
    plt.rc("font", **font_options)  # 设置中文字体为微软雅黑，以正确显示中文
    fig, ax = plt.subplots(figsize=(14, 10))  # 创建一个14x10英寸的画布和子图
    # 绘制水平条形图
    bars = ax.barh(
        df_importance_sorted["Feature"],
        df_importance_sorted["Importance"],
        color=bar_color,
        alpha=0.8,
    )
    ax.set_title(title, fontsize=18, pad=20)  # 设置标题
    ax.set_xlabel("Importance Score", fontsize=18)  # 设置x轴标签
    ax.set_ylabel("Feature", fontsize=18)  # 设置y轴标签
    ax.tick_params(axis="both", which="major", labelsize=18)  # 设置刻度标签的大小
    ax.grid(axis="x", linestyle="--", alpha=0.6)  # 显示x轴方向的网格线
    # 在每个条形图旁边显示具体的重要性数值
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width:.4f}",
            va="center",
            ha="left",
            fontsize=18,
        )
    ax.set_xlim(right=ax.get_xlim()[1] * 1.2)  # 调整x轴范围，为数值标签留出空间
    N_DONUT_FEATURES = 5  # 设置甜甜圈图中要显示的特征数量
    if len(df_importance) < N_DONUT_FEATURES:  # 如果总特征数小于5，则取全部特征
        N_DONUT_FEATURES = len(df_importance)
    df_desc = df_importance.sort_values(
        by="Importance", ascending=False
    )  # 按重要性降序排序
    top_n_features = df_desc.head(N_DONUT_FEATURES)  # 选取最重要的N个特征
    donut_feature_names = top_n_features["Feature"].tolist()  # 获取这N个特征的名称
    # 如果有特征且总重要性大于0，则绘制甜甜圈图
    if not top_n_features.empty and top_n_features["Importance"].sum() > 0:
        total_donut_importance = top_n_features[
            "Importance"
        ].sum()  # 计算Top-N特征的重要性总和
        donut_percentages = (
            top_n_features["Importance"] / total_donut_importance * 100
        )  # 计算每个特征在Top-N中的百分比
        ax_inset = fig.add_axes(
            [0.45, 0.15, 0.45, 0.45]
        )  # 在主图上创建一个嵌入的子图（甜甜圈图的位置）
        colors = matplotlib.colormaps["tab10"].colors  # 获取一组颜色
        # 绘制饼图（通过设置width属性使其变为甜甜圈图）
        wedges, _ = ax_inset.pie(
            donut_percentages,
            colors=colors[: len(top_n_features)],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.45, edgecolor="w"),
        )
        # 计算Top-N特征占总特征重要性的比例
        subset_importance_ratio = (
            top_n_features["Importance"].sum() / df_importance["Importance"].sum()
        )
        # 在甜甜圈中心添加文本
        ax_inset.text(
            0,
            0,
            f"Top {N_DONUT_FEATURES} Features\nAccount for\n{subset_importance_ratio:.2%}",
            ha="center",
            va="center",
            fontsize=16,
            linespacing=1.4,
        )
        label_threshold = 3.0  # 设置标签显示的阈值，小于此值的百分比会用引导线引出
        y_text_offsets = {"left": 1.4, "right": 1.4}  # 初始化引导线标签的垂直偏移量
        # 为每个扇区添加百分比标签
        for i, p in enumerate(wedges):
            percent = donut_percentages.iloc[i]
            ang = (p.theta2 - p.theta1) / 2.0 + p.theta1  # 计算扇区中心角度
            y = np.sin(np.deg2rad(ang))  # 计算标签的y坐标
            x = np.cos(np.deg2rad(ang))  # 计算标签的x坐标
            # 如果百分比小于阈值，使用引导线
            if percent < label_threshold and percent > 0:
                side = "right" if x > 0 else "left"  # 判断标签在左侧还是右侧
                y_pos = y_text_offsets[side]  # 获取当前侧的y偏移
                y_text_offsets[side] += (
                    -0.2 if y > 0 else 0.2
                )  # 更新偏移量，避免标签重叠
                connectionstyle = f"angle,angleA=0,angleB={ang}"  # 设置引导线样式
                # 添加带引导线的注释
                ax_inset.annotate(
                    f"{percent:.1f}%",
                    xy=(x, y),
                    xytext=(0.2 * np.sign(x), y_pos),
                    fontsize=16,
                    ha="center",
                    arrowprops=dict(
                        arrowstyle="-",
                        connectionstyle=connectionstyle,
                        relpos=(0.5, 0.5),
                    ),
                )
            # 如果百分比大于阈值，直接在扇区内显示
            elif percent > 0:
                ax_inset.text(
                    x * 0.8,
                    y * 0.8,
                    f"{percent:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
        # 在甜甜圈图右侧添加图例
        ax_inset.legend(
            wedges,
            donut_feature_names,
            loc="center left",
            bbox_to_anchor=(1.2, 0.5),
            frameon=False,
            fontsize=16,
        )
    plt.savefig(save_path, dpi=720, bbox_inches="tight")  # 保存高分辨率图表
    plt.close(fig)  # 关闭图表，释放内存


# 定义一个函数，用于绘制残差图
def plot_residuals_styled(residuals, y_pred, save_path, title):
    """
    绘制残差与预测值的关系图，并高亮显示异常值。
    residuals: 残差 (真实值 - 预测值)
    y_pred: 预测值
    save_path: 图片保存路径
    title: 图表标题
    """
    plt.style.use("seaborn-v0_8-whitegrid")  # 使用预设绘图风格
    font_options = {
        "family": "sans-serif",  # 设置字体家族
        "sans-serif": prop.get_name(),  # 设置字体
        "size": 12,  # 设置字体大小
    }
    plt.rc("font", **font_options)  # 设置中文字体
    fig, ax = plt.subplots(figsize=(10, 8))  # 创建一个10x8英寸的画布
    sd_residuals = np.std(residuals)  # 计算残差的标准差
    is_outlier = (
        np.abs(residuals) > 2 * sd_residuals
    )  # 定义异常值：绝对残差大于2倍标准差
    num_outliers = np.sum(is_outlier)  # 计算异常值的数量
    logger_tool.debug(
        f"In the dataset '{title}', {num_outliers} outliers (residuals > 2S.D.) were found."
    )  # 打印异常值信息
    sd_label = f"S.D. (±{sd_residuals:.2f})"  # 准备标准差区间的图例标签
    ax.axhspan(
        -sd_residuals, sd_residuals, color="yellow", alpha=0.3, label=sd_label
    )  # 绘制一个表示一个标准差范围的水平区域
    # 绘制正常值的散点图
    ax.scatter(
        y_pred[~is_outlier],
        residuals[~is_outlier],
        alpha=0.6,
        c="green",
        edgecolors="k",
        linewidth=0.5,
        s=50,
        label="Normal Values",
    )
    # 绘制异常值的散点图
    ax.scatter(
        y_pred[is_outlier],
        residuals[is_outlier],
        alpha=0.8,
        c="red",
        edgecolors="k",
        linewidth=0.5,
        s=70,
        label="Outliers (> 2S.D.)",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)  # 绘制残差为0的参考线
    ax.set_title(title, fontsize=16, weight="bold")  # 设置标题
    ax.set_xlabel("Predicted Values", fontsize=14)  # 设置x轴标签
    ax.set_ylabel(
        "Residuals (True Values - Predicted Values)", fontsize=14
    )  # 设置y轴标签
    # 设置图表边框样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)
    ax.legend(loc="upper right", fontsize=12)  # 显示图例
    y_max = np.max(np.abs(residuals)) * 1.2  # 计算y轴的范围
    ax.set_ylim(-y_max, y_max)  # 设置y轴范围
    plt.tight_layout()  # 调整布局，防止标签重叠
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # 保存图表
    plt.close()  # 关闭图表


# =========================================================================
# ======================= 新增：手动PDP计算与绘图函数 =======================
# =========================================================================
def manual_pdp_1d(model, X_data, feature_name, grid_resolution=50):
    """
    手动计算一维部分依赖(PDP)和个体条件期望(ICE)数据。
    返回:
    - grid_values: 特征的网格点。
    - pdp_values: 对应的PDP值 (ICE的均值)。
    - ice_lines: 所有样本的ICE线数据。
    """
    # 在特征的最小值和最大值之间生成一系列网格点
    grid_values = np.linspace(
        X_data[feature_name].min(), X_data[feature_name].max(), grid_resolution
    )
    # 初始化一个数组来存储每个样本的ICE线数据
    ice_lines = np.zeros((len(X_data), grid_resolution))
    pbar = tqdm.tqdm(total=len(X_data), desc="计算ICE线")

    # 遍历数据集中每一个样本
    for i, (_, sample) in enumerate(X_data.iterrows()):
        # 创建一个临时DataFrame，行数为网格点数，内容为当前样本的重复
        temp_df = pd.DataFrame([sample] * grid_resolution)
        # 将要分析的特征列替换为网格值
        temp_df[feature_name] = grid_values
        # 使用模型进行预测，得到这个样本在不同特征值下的预测结果，即ICE线
        ice_lines[i, :] = model.predict(temp_df)
        pbar.update(1)
    pbar.close()
    # PDP是所有ICE线的平均值，在每个网格点上求均值
    pdp_values = np.mean(ice_lines, axis=0)

    # 返回计算结果
    return grid_values, pdp_values, ice_lines


def manual_pdp_2d(model, X_data, features_tuple, grid_resolution=20):
    """
    手动计算二维部分依赖(PDP)数据。
    返回:
    - grid_1: 第一个特征的网格点。
    - grid_2: 第二个特征的网格点。
    - pdp_values: 二维网格上对应的PDP值。
    """
    feat1_name, feat2_name = features_tuple  # 获取两个特征的名称
    # 为第一个特征生成网格点
    grid_1 = np.linspace(
        X_data[feat1_name].min(), X_data[feat1_name].max(), grid_resolution
    )
    # 为第二个特征生成网格点
    grid_2 = np.linspace(
        X_data[feat2_name].min(), X_data[feat2_name].max(), grid_resolution
    )

    # 初始化一个二维数组来存储PDP值
    pdp_values = np.zeros((grid_resolution, grid_resolution))

    # 遍历二维网格的每一个点
    for i, val1 in enumerate(grid_1):
        for j, val2 in enumerate(grid_2):
            # 创建一个原始数据的临时副本
            X_temp = X_data.copy()
            # 将第一个特征的所有值都设为当前的网格点值
            X_temp[feat1_name] = val1
            # 将第二个特征的所有值都设为当前的网格点值
            X_temp[feat2_name] = val2

            # 对修改后的整个数据集进行预测
            preds = model.predict(X_temp)
            # 计算预测结果的平均值，作为该网格点的PDP值
            pdp_values[j, i] = np.mean(preds)

    # 返回计算结果
    return grid_1, grid_2, pdp_values


def plot_3d_scatter_three_features(X_data, y_pred, features_tuple, save_path):
    """
    绘制三个特征的3D散点图，并用预测值对散点进行着色。
    """
    font_options = {
        "family": "sans-serif",  # 设置字体家族
        "sans-serif": prop.get_name(),  # 设置字体
        "size": 18,  # 设置字体大小
    }
    plt.rc("font", **font_options)
    feat1_name, feat2_name, feat3_name = features_tuple  # 获取三个特征的名称
    fig = plt.figure(figsize=(12, 9))  # 创建一个12x9英寸的画布
    ax = fig.add_subplot(111, projection="3d")  # 添加一个3D子图
    # 绘制3D散点图，x,y,z轴分别是三个特征的值，颜色c由模型预测值决定
    sc = ax.scatter(
        X_data[feat1_name],
        X_data[feat2_name],
        X_data[feat3_name],
        c=y_pred,
        cmap="viridis",
        s=30,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.set_xlabel(
        f"{feat1_name} (Standardized Value)", fontsize=10, labelpad=10
    )  # 设置x轴标签
    ax.set_ylabel(
        f"{feat2_name} (Standardized Value)", fontsize=10, labelpad=10
    )  # 设置y轴标签
    ax.set_zlabel(
        f"{feat3_name} (Standardized Value)", fontsize=10, labelpad=10, rotation=180
    )  # 设置z轴标签
    ax.set_title(
        f"3D Scatter Plot of {feat1_name}, {feat2_name}, {feat3_name}", fontsize=14
    )  # 设置标题
    # 添加颜色条，并设置标签
    cbar = fig.colorbar(
        sc, shrink=0.5, aspect=20, label="Model Predicted Values", pad=0.1
    )
    ax.view_init(elev=20, azim=45)  # 设置3D视图的角度
    plt.savefig(save_path, dpi=300)  # 保存图表
    plt.close(fig)  # 关闭图表
    logger_tool.debug(f"成功绘制 3D 散点图 for {features_tuple}")  # 打印成功信息


def plot_3d_pdp_fixed_value(
    model,
    X_data,
    features,
    save_path,
    fixed_feature=None,
    fixed_value=None,
    grid_resolution=50,
):
    """
    绘制三个特征的3D PDP图，其中一个特征被固定在特定值。
    """
    feature_1, feature_2, feature_3 = features  # 获取三个特征名称
    if fixed_feature is None:  # 如果没有指定要固定的特征
        fixed_feature = feature_3  # 默认固定第三个特征
    # 找出需要变化的两个特征
    varying_features = [f for f in features if f != fixed_feature]
    varying_feat_1, varying_feat_2 = varying_features[0], varying_features[1]
    if fixed_value is None:  # 如果没有指定固定的值
        fixed_value = X_data[fixed_feature].mean()  # 默认使用该特征的平均值
    # 为两个变化的特征生成网格点
    feat1_vals = np.linspace(
        X_data[varying_feat_1].min(), X_data[varying_feat_1].max(), grid_resolution
    )
    feat2_vals = np.linspace(
        X_data[varying_feat_2].min(), X_data[varying_feat_2].max(), grid_resolution
    )
    XX, YY = np.meshgrid(feat1_vals, feat2_vals)  # 创建二维网格
    # 使用数据集中所有特征的平均值作为背景行，以减少其他特征的影响
    background_row = X_data.mean().to_dict()
    # 创建一个包含网格点组合的DataFrame
    grid_data = pd.DataFrame(np.c_[XX.ravel(), YY.ravel()], columns=varying_features)
    # 创建一个用于预测的网格DataFrame，以背景行作为基础
    X_grid = pd.DataFrame([background_row] * len(grid_data))
    # 将变化的特征列替换为网格值
    X_grid[varying_feat_1] = grid_data[varying_feat_1].values
    X_grid[varying_feat_2] = grid_data[varying_feat_2].values
    # 将固定的特征列设置为指定的值
    X_grid[fixed_feature] = fixed_value
    X_grid = X_grid[X_data.columns]  # 确保列顺序与训练时一致
    preds = model.predict(X_grid).reshape(XX.shape)  # 进行预测，并重塑为网格形状
    font_options = {
        "family": "sans-serif",  # 设置字体家族
        "sans-serif": prop.get_name(),  # 设置字体
        "size": 18,  # 设置字体大小
    }
    plt.rc("font", **font_options)  # 设置中文字体
    fig = plt.figure(figsize=(12, 9))  # 创建画布
    ax = fig.add_subplot(111, projection="3d")  # 创建3D子图
    # 绘制3D曲面图
    surface = ax.plot_surface(
        XX, YY, preds, cmap="viridis", alpha=0.9, edgecolor="k", linewidth=0.2
    )
    fig.colorbar(
        surface, ax=ax, shrink=0.6, aspect=20, label="Predicted Values"
    )  # 添加颜色条
    ax.set_xlabel(
        f"{varying_feat_1} (Standardized Value)", fontsize=16, labelpad=10
    )  # 设置x轴标签
    ax.set_ylabel(
        f"{varying_feat_2} (Standardized Value)", fontsize=16, labelpad=10
    )  # 设置y轴标签
    ax.set_zlabel(
        "Predicted Values", fontsize=16, labelpad=10, rotation=90
    )  # 设置z轴标签
    # 设置标题
    title_text = f"3D Dependency Plot: {varying_feat_1} vs {varying_feat_2}\nFixed {fixed_feature} = {fixed_value:.3f}"
    ax.set_title(title_text, fontsize=18)
    ax.view_init(elev=25, azim=-120)  # 设置视角
    plt.savefig(save_path, dpi=300)  # 保存图片
    plt.close(fig)  # 关闭图表
    logger_tool.debug(
        f"成功绘制 固定值3D PDP for {features}，固定 {fixed_feature}"
    )  # 打印成功信息


def data_norm_get(
    x_train, x_test, y_train, y_test, save_path=None, non_standardize_features=[]
):
    # 确保这些特征确实存在于数据中
    existing_non_standardize_features = [
        col for col in non_standardize_features if col in x_train.columns
    ]

    # 分离需要标准化和不需要标准化的特征
    standardize_features = [
        col for col in x_train.columns if col not in existing_non_standardize_features
    ]

    # 只对需要标准化的特征进行处理
    scaler = StandardScaler()  # 实例化一个StandardScaler对象

    # 对训练集需要标准化的特征进行拟合和转换
    X_train_standardize_scaled = scaler.fit_transform(x_train[standardize_features])
    X_train_standardize_scaled_df = pd.DataFrame(
        X_train_standardize_scaled, columns=standardize_features, index=x_train.index
    )
    if save_path is not None:
        pickle.dump(scaler, open(save_path, "wb"))
    # 对测试集需要标准化的特征进行转换
    X_test_standardize_scaled = scaler.transform(x_test[standardize_features])
    X_test_standardize_scaled_df = pd.DataFrame(
        X_test_standardize_scaled, columns=standardize_features, index=x_test.index
    )

    # 合并标准化后的特征和不需要标准化的原始特征
    X_train_scaled_df = pd.concat(
        [X_train_standardize_scaled_df, x_train[existing_non_standardize_features]],
        axis=1,
    )
    X_test_scaled_df = pd.concat(
        [X_test_standardize_scaled_df, x_test[existing_non_standardize_features]],
        axis=1,
    )

    # 确保列的顺序与原始数据一致
    X_train_scaled_df = X_train_scaled_df[x_train.columns]
    X_test_scaled_df = X_test_scaled_df[x_test.columns]

    return X_train_scaled_df, X_test_scaled_df


def plot_custom_shap_summary(shap_values, X_test_data, title, ax, cmap_name):
    fig = ax.get_figure()  # 获取当前坐标轴所属的图形对象
    plt.sca(ax)  # 将当前绘图的坐标轴设置为传入的ax对象
    # 绘制shap摘要图
    shap.summary_plot(
        shap_values,
        X_test_data,
        plot_type="dot",
        show=False,
        cmap=plt.get_cmap(cmap_name),
        color_bar=True,
    )
    mappable = (
        ax.collections[0] if ax.collections else None
    )  # 获取图中的散点图对象，用于颜色条映射
    ax.set_title(title, fontsize=16)  # 设置标题
    ax.tick_params(axis="both", labelsize=14)
    ax.set_facecolor("#f0f0f0")  # 设置背景颜色
    ax.grid(
        axis="both", color="white", linestyle="-", linewidth=1.3, zorder=0
    )  # 设置背景网格线
    ax.tick_params(axis="both", length=0, labelsize=16)  # 隐藏坐标轴刻度线，但保留标签
    for spine in ax.spines.values():  # 遍历所有图框
        spine.set_visible(False)  # 去掉


def plot_interaction_heatmap(
    shap_values, shap_interaction_values, X_data, title, ax, cmap_name
):
    feature_names = X_data.columns  # 获取所有特征的名称
    N_FEATURES = len(feature_names)  # 计算特征的总数
    mean_abs_shap = np.abs(shap_values).mean(0)  # 计算每个特征的平均绝对SHAP值
    sorted_indices = np.argsort(mean_abs_shap)[::-1]  # 根据特征重要性降序排序，获取索引
    sorted_names = feature_names[
        sorted_indices
    ]  # 根据排序后的索引，获取排序后的特征名称
    mean_abs_interactions = np.abs(shap_interaction_values).mean(
        0
    )  # 计算所有特征对的平均绝对SHAP交互值
    plot_matrix = mean_abs_interactions[sorted_indices][
        :, sorted_indices
    ]  # 根据特征重要性对交互矩阵进行重排
    lower_triangle_indices = np.tril_indices(
        N_FEATURES, k=-1
    )  # 获取矩阵下三角部分的索引（不包括对角线）
    vmax = np.nanmax(
        plot_matrix[lower_triangle_indices]
    )  # 找到下三角部分中的最大交互值，用于归一化
    if np.isnan(vmax) or vmax == 0:
        vmax = 1.0  # 如果最大值为NaN或0，则设为1以避免错误
    vmin = 0  # 设置交互值的最小值为0
    cmap = plt.get_cmap(cmap_name)  # 从指定的名称获取颜色映射对象
    scaling_factor = 300  # 定义一个缩放因子，用于控制图中气泡的大小
    ax.set_xticks(range(len(sorted_names)))  # 设置x轴的刻度位置
    ax.set_xticklabels(sorted_names, rotation=0)  # 设置x轴的特征名
    ax.set_yticks(range(len(sorted_names)))  # 设置y轴的刻度位置
    ax.set_yticklabels(sorted_names)  # 设置y轴的特征名
    ax.tick_params(axis="both", which="major", labelsize=16)  # 设置刻度标注的大小
    ax.set_xlabel("Driving factors", fontsize=16)  # 设置x轴的标题

    ax.set_ylabel("Driving factors", fontsize=16)  # 设置y轴的标题
    ax.set_xlim(-0.5, len(sorted_names) - 0.5)  # 设置x轴的显示范围
    ax.set_ylim(-0.5, len(sorted_names) - 0.5)  # 设置y轴的显示范围
    ax.set_facecolor("#f0f0f0")  # 设置背景颜色
    ax.grid(
        axis="both", color="white", linestyle="-", linewidth=1.3, zorder=0
    )  # 背景网格线

    for spine in ax.spines.values():  # 遍历所有图框
        spine.set_visible(False)  # 隐藏所有图框
    base_x, base_y = 0.58, 0.04  # 定义图例区域的基准坐标
    legend_height, legend_width = 0.45, 0.45  # 定义图例区域的整体高和宽
    title_height = 0.05  # 定义图例标题区域的高度
    # ax_title.axis("off")  # 关闭该子坐标轴的边框和刻度
    # ax.set_axis_off()
    cbar_width = 0.04  # 定义颜色条部分的宽度
    cbar_height = legend_height - title_height  # 计算颜色条部分的高度
    # 在现有的ax坐标轴中创建一个嵌入式的新坐标轴ax_cbar，用于放置颜色条
    ax_cbar = ax.inset_axes(
        [
            base_x,
            base_y,
            cbar_width,
            cbar_height,
        ],  # 定义新坐标轴的位置和大小。这是一个包含四个值的列表：[x坐标, y坐标, 宽度, 高度]。
        transform=ax.transAxes,  # 指定坐标系，相对于父坐标轴ax的边界框来计算的
        zorder=5,
    )
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)  # 创建一个归一化对象

    for i in range(len(sorted_names)):
        j = 0
        while j <= i:
            ax.scatter(
                j,
                i,
                s=scaling_factor * plot_matrix[i, j] / vmax,
                c=plot_matrix[i, j],
                cmap=cmap,
                norm=norm,
            )
            j += 1

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # 创建一个可映射对象
    sm.set_array([])  # 设置一个空数组给可映射对象    # 创建并配置颜色条
    cbar = plt.colorbar(
        sm,
        cax=ax_cbar,  # 指定用于绘制颜色条的坐标轴
        orientation="vertical",  # 设置颜色条的方向为垂直方向
        ticks=np.linspace(vmin, vmax, 5),
    )  # 设置颜色条上的刻度
    cbar.ax.set_yticklabels(
        [f"{val:.4f}" for val in cbar.get_ticks()]
    )  # 自定义颜色条的刻度标签格式
    cbar.ax.tick_params(labelsize=16, length=0)  # 设置颜色条刻度的样式
    cbar.outline.set_visible(True)  # 显示颜色条的轮廓线
    circ_width = legend_width - cbar_width  # 计算尺寸图例部分的宽度
    circ_height = cbar_height  # 计算尺寸图例部分的高度
    # 创建一个用于气泡图例的嵌入式坐标轴
    ax_circ = ax.inset_axes(
        [
            base_x + cbar_width,  # x
            base_y,  # y
            circ_width,  # 坐标轴的宽度
            circ_height,
        ],  # 坐标轴的高度
        transform=ax.transAxes,  # 指定坐标系
        zorder=5,
    )
    ax_circ.axis("off")  # 关闭该子坐标轴的边框和刻度
    num_circles = 5  # 定义要在图例中显示的圆圈数量
    size_values = np.linspace(
        vmax / num_circles, vmax, num_circles
    )  # 生成一系列等间距的数值，用于图例圆圈
    legend_dot_sizes = (
        size_values / vmax * scaling_factor
    )  # 计算图例中每个圆圈的实际大小
    y_positions = np.linspace(0.1, 0.9, num_circles)  # 生成图例中每个圆圈的垂直位置
    x_pos_circle, x_pos_text = 0.35, 0.4  # 定义图例中圆圈和文本的水平位置
    # 循环绘制一个气泡图例及其对应的文本标签
    for i in range(num_circles):
        # 在ax_circ坐标轴上绘制一个圆
        ax_circ.scatter(
            x_pos_circle,  # 圆心的x坐标
            y_positions[i],  # 圆心的y坐标，每次循环使用不同的垂直位置
            s=legend_dot_sizes[i],  # 圆的大小，每次循环使用不同的大小
            facecolors="none",  # 填充颜色设置为none，即空心圆
            edgecolors="black",  # 边框颜色设置为黑色
            linewidth=1,
        )  # 边框线宽为1
        # 添加文本标签
        ax_circ.text(
            x_pos_text,  # x坐标
            y_positions[i],  # y坐标
            f"{size_values[i]:.4f}",  # 要显示的文本内容
            va="center",  # 垂直对齐方式
            ha="left",  # 水平对齐方式
            fontsize=18,
        )  # 字体大小
    ax_circ.set_xlim(0, 1)  # 子坐标轴的x范围
    ax_circ.set_ylim(0, 1)  # 子坐标轴的y范围
    ax.set_title(title, fontsize=18)  # 设置主图的标题


def create_and_save_summary_plot(
    shap_values,
    X_test,
    title,
    cmap_name,
    output_folder,
    filename_base,
    selected_theme_id=20,
):
    print(f"\n{'=' * 20} 开始绘制SHAP摘要图 {'=' * 20}")
    fig, ax = plt.subplots(figsize=(8, 10))
    # 调用函数绘制摘要图
    plot_custom_shap_summary(shap_values, X_test, title, ax, cmap_name)
    # 保存
    png_path = os.path.join(output_folder, f"{filename_base}_{selected_theme_id}.png")
    pdf_path = os.path.join(output_folder, f"{filename_base}_{selected_theme_id}.pdf")
    # 保存
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("SHAP 摘要图已保存为 PNG 和 PDF。")


def create_and_save_top_dependence_plots(
    shap_values,
    X_test,
    cmap_name,
    output_folder,
    selected_theme_id=20,
):
    nature_colors = {
        "blue": "#1f77b4",
        "orange": "#ff7f0e",
        "green": "#2ca02c",
        "red": "#d62728",
        "purple": "#9467bd",
        "gray": "#7f7f7f",
    }
    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # 计算每个特征的平均绝对SHAP值
    feature_importances = pd.Series(
        mean_abs_shap, index=X_test.columns
    )  # 创建一个包含特征重要性的Series
    top_features = (
        feature_importances.sort_values(ascending=False).head(6).index.tolist()
    )  # 获取最重要的特征
    print(f"重要性最高的特征是: {top_features}")
    plot_paths = []  # 创建一个空列表，用于存储每个独立依赖图的保存路径
    feature_names = X_test.columns.tolist()  # 获取特征名称列表
    for i, feature in enumerate(top_features):  # 循环遍历
        print(f"正在为特征 '{feature}' 绘制并保存独立的依赖图...")
        fig_dep, ax_dep = plt.subplots(figsize=(7, 5))
        # 绘制依赖图
        x_vals = X_test[feature].values
        y_vals = shap_values[:, feature_names.index(feature)]
        gam = LinearGAM(s(0), n_splines=10).fit(x_vals.reshape(-1, 1), y_vals)
        x_pred = np.linspace(x_vals.min(), x_vals.max(), 300)
        y_pred = gam.predict(x_pred)

        XX = gam.generate_X_grid(term=0, n=300)
        predictions = gam.prediction_intervals(XX, width=0.95)

        ax_dep.scatter(
            x_vals, y_vals, color=nature_colors["blue"], alpha=0.3, zorder=1, s=10
        )

        ax_dep.plot(
            x_pred,
            y_pred,
            color=nature_colors["blue"],
            linewidth=2,
            zorder=3,
        )

        ax_dep.fill_between(
            XX.flatten(),
            predictions[:, 0],
            predictions[:, 1],
            color=nature_colors["gray"],
            alpha=0.3,
            zorder=0,
        )

        ax_dep.axhline(
            y=0, color="black", linestyle="--", linewidth=1, alpha=0.5
        )  # 在y=0处画一条水平虚线

        zero_crossings = np.where(np.diff(np.sign(y_pred)))[0]
        critical_points = []
        for crossing in zero_crossings:
            x1, x2 = x_pred[crossing], x_pred[crossing + 1]
            y1, y2 = y_pred[crossing], y_pred[crossing + 1]
            if y1 * y2 < 0:
                x_critical = x1 - y1 * (x2 - x1) / (y2 - y1)
                critical_points.append(x_critical)
                ax_dep.axvline(
                    x_critical,
                    color=nature_colors["red"],
                    linestyle="--",
                    alpha=0.7,
                )
                ax_dep.text(
                    x_critical,
                    ax_dep.get_ylim()[1] * 0.7,
                    f"{x_critical:.1f}",
                    ha="center",
                    va="top",
                    fontsize=12,
                    color="black",
                    fontweight="bold",
                )
        ax_dep.fill_between(
            x_pred,
            y_pred,
            0,
            where=(y_pred >= 0),
            color=nature_colors["blue"],
            alpha=0.4,
            interpolate=True,
            label="Positive shap",
        )
        ax_dep.fill_between(
            x_pred,
            y_pred,
            0,
            where=(y_pred < 0),
            color=nature_colors["red"],
            alpha=0.4,
            interpolate=True,
            label="Negative shap",
        )
        ax_dep.set_ylabel("SHAP value", fontsize=16)
        ax_dep.set_xlabel(feature, fontsize=16)
        ax_dep.legend(fontsize=16)
        ax_dep.minorticks_on()
        ax_dep.grid(True, which="both")

        filename = f"shap_dependence_{feature}"  # 文件名
        # 保存
        png_path = os.path.join(output_folder, f"{filename}_{selected_theme_id}.png")
        pdf_path = os.path.join(output_folder, f"{filename}_{selected_theme_id}.pdf")
        fig_dep.savefig(png_path, dpi=300, bbox_inches="tight")
        fig_dep.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig_dep)
        plot_paths.append(png_path)  # 将保存的路径添加到列表中用于后续拼接
        print(f"特征 '{feature}' 的依赖图已保存。")  # 拼接结果图
        images = [Image.open(path) for path in plot_paths]  # 打开所有已保存的依赖图
        width, height = images[0].size  # 获取单张图片的尺寸
        composite_image = Image.new(
            "RGB", (width * 3, height * 2), "white"
        )  # 创建一个空白大图
        coords = [
            (c * width, r * height) for r in range(2) for c in range(3)
        ]  # 生成每张小图要粘贴的坐标
        for img, coord in zip(images, coords):  # 遍历图片和坐标
            composite_image.paste(img, coord)  # 将小图粘贴到大图的指定位置
        for img in images:
            img.close()
        composite_filename = "shap_dependence_composite_stitched"  # 组合图文件名
        composite_image.save(
            os.path.join(output_folder, f"{composite_filename}_{selected_theme_id}.png")
        )
        composite_image.save(
            os.path.join(
                output_folder, f"{composite_filename}_{selected_theme_id}.pdf"
            ),
            "PDF",
            resolution=300.0,
        )


def create_and_save_interaction_heatmap(
    shap_values,
    shap_interaction_values,
    X_test,
    title,
    cmap_name,
    output_folder,
    filename_base,
    selected_theme_id=20,
):
    print(f"\n{'=' * 20} 开始处理 SHAP 交互效应摘要图 {'=' * 20}")
    fig, ax = plt.subplots(figsize=(12, 11))  # 调用函数绘图
    plot_interaction_heatmap(
        shap_values, shap_interaction_values, X_test, title, ax, cmap_name
    )
    # 保存
    png_path = os.path.join(output_folder, f"{filename_base}_{selected_theme_id}.png")
    pdf_path = os.path.join(output_folder, f"{filename_base}_{selected_theme_id}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def create_and_save_top_interaction_dependence_plots(
    shap_interaction_values,
    X_test,
    title,
    cmap_name,
    output_folder,
    filename_base,
    selected_theme_id=20,
):
    print(f"\n{'=' * 20} 开始处理SHAP交互效应依赖图 {'=' * 20}")
    feature_names = X_test.columns  # 获取特征名称
    mean_abs_interactions = np.abs(shap_interaction_values).mean(
        0
    )  # 计算平均绝对交互值
    upper_tri_indices = np.triu_indices(
        len(feature_names), k=1
    )  # 获取交互矩阵的上三角索引
    # 创建一个列表，包含所有特征对及其交互强度
    interaction_list = [
        {
            "pair": (feature_names[i], feature_names[j]),
            "strength": mean_abs_interactions[i, j],
        }
        for i, j in zip(upper_tri_indices[0], upper_tri_indices[1])
    ]
    # 根据交互强度对列表进行降序排序，并提取前三名的特征对
    top_pairs = [
        item["pair"]
        for item in sorted(interaction_list, key=lambda x: x["strength"], reverse=True)[
            :9
        ]
    ]
    print(f"已识别出前9大交互组合: {top_pairs}")  # 打印识别出的前9交互组合
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))  # 创建一个3行3列的子图布局
    fig.suptitle(title, fontsize=22, y=0.90)  # 为整个图形设置一个总标题
    for i, pair in enumerate(top_pairs):  # 循环遍历前三个交互对
        ax = axes[i // 3, i % 3]  # 获取当前要绘图的子图坐标轴
        # 绘制交互依赖图
        shap.dependence_plot(
            pair, shap_interaction_values, X_test, ax=ax, show=False, dot_size=22
        )
        scatter_object = ax.collections[0]  # 获取刚刚绘制的散点图对象
        scatter_object.set_cmap(plt.get_cmap(cmap_name))  # 手动设置其颜色映射
        scatter_object.set_zorder(3)  # 确保散点图在最上层
        ax.set_title(
            f"{pair[0]} and {pair[1]} ({i + 1})", loc="right", fontsize=22
        )  # 在子图右上角设置编号

        ax.set_facecolor("#f0f0f0")  # 设置背景颜色
        ax.grid(
            axis="both", color="white", linestyle="-", linewidth=2, zorder=0
        )  # 添加网格线

        ax.axhline(
            0, color="gray", linestyle="--", linewidth=1, zorder=2
        )  # 在y=0处画一条虚线

        ax.tick_params(axis="both", length=0, labelsize=22)  # 隐藏刻度线
        for spine in ax.spines.values():
            spine.set_visible(False)  # 隐藏图框
        ax.set_ylabel("SHAP interaction value", fontsize=22)  # 设置y轴标签
        ax.set_xlabel(ax.get_xlabel(), fontsize=22)  # 设置x轴标签
        if len(fig.axes) > i + 2:
            fig.axes[-1].remove()  # 移除自动生成的颜色条
        cb = fig.colorbar(
            scatter_object, orientation="horizontal", label=f"{pair[1]}"
        )  # 创建一个水平方向的颜色条
        main_feature, interaction_feature = pair  # 解包特征对名称
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 自动调整子图布局以防止重叠
    # 保存
    png_path = os.path.join(output_folder, f"{filename_base}_{selected_theme_id}.png")
    pdf_path = os.path.join(output_folder, f"{filename_base}_{selected_theme_id}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_regression_fit2(
    y_train, y_train_pred, r2, rmse, mae, data_label_en, title_en, save_path
):
    color_schemes = {
        "2_Nature": {
            "train": "#C15060",
            "test": "#34768C",
            "line": "#3C5488",
            "band": "#C15060",
            "grid": "#B0B0B0",
            "bg_fill": "#FAFAFA",
        },
    }
    selected_theme_id = 20
    c_train = color_schemes["2_Nature"]["train"]
    c_test = color_schemes["2_Nature"]["test"]
    c_line = color_schemes["2_Nature"]["line"]
    c_band = color_schemes["2_Nature"]["band"]
    c_grid = color_schemes["2_Nature"]["grid"]
    c_bg_fill = color_schemes["2_Nature"]["bg_fill"]
    min_val = 0
    max_val = max(y_train.max(), y_train_pred.max()) + 5
    fig = plt.figure(figsize=(12, 8), dpi=600)
    gs = gridspec.GridSpec(
        2,
        3,
        width_ratios=[0.6, 3, 0.5],
        height_ratios=[0.5, 3],
        wspace=0.25,
        hspace=0.1,
    )
    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    ax_left = fig.add_subplot(gs[1, 0], sharey=ax_main)
    ax_top_left = fig.add_subplot(gs[0, 0], sharex=ax_left)
    ax_main.scatter(
        y_train,
        y_train_pred,
        c=c_train,
        s=60,
        alpha=0.9,
        edgecolors="grey",
        linewidth=0.5,
        label="Training set",
    )

    ax_main.plot(
        [min_val, max_val], [min_val, max_val], c=c_line, linestyle="--", linewidth=1.5
    )
    x_line = np.linspace(min_val, max_val, 100)
    ax_main.fill_between(
        x_line, x_line * 0.8, x_line * 1.2, color=c_band, alpha=0.25, label="±20%"
    )
    metrics_text = f"$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$\n$MAE = {mae:.3f}$"
    # 在图的左上角添加文本框，显示评估指标
    ax_main.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax_main.transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    ax_main.set_xlim([min_val, max_val])  # 设置x轴范围
    ax_main.set_ylim([min_val, max_val])
    ax_main.tick_params(axis="both", labelsize=14)
    ax_main.set_xlabel("True Values", fontsize=16)  # 设置x轴标签
    ax_main.set_ylabel("Predicted Values", fontsize=16)  # 设置y轴标签
    # 显示图例在右下角
    ax_main.legend(fontsize=14, loc="lower right")

    bins = 15
    ax_top.hist(
        [y_train], bins=bins, stacked=True, color=[c_train], edgecolor="grey", alpha=0.8
    )
    ax_top.tick_params(axis="both", labelsize=14)
    ax_top.set_ylabel("Count", fontsize=16)  # 设置y轴标签
    ax_right.hist(
        [y_train_pred],
        bins=bins,
        stacked=True,
        orientation="horizontal",
        color=[c_train],
        edgecolor="grey",
        alpha=0.8,
    )
    ax_right.tick_params(axis="both", labelsize=14)
    ax_right.set_xlabel("Count", fontsize=16)  # 设置y轴标签
    resid_train = y_train - y_train_pred
    ax_left.scatter(
        resid_train,
        y_train_pred,
        c=c_train,
        s=40,
        alpha=0.8,
        edgecolors="grey",
        linewidth=0.5,
    )
    ax_left.tick_params(axis="both", labelsize=14)
    ax_left.axvline(0, color=c_line, linestyle="--", linewidth=1.2)
    ax_left.axvspan(-3, 3, color=c_band, alpha=0.2)
    ax_left.set_xlabel("Residuals", fontsize=16)  # 设置y轴标签
    ax_top_left.hist(
        [resid_train],
        bins=10,
        stacked=True,
        color=[c_train],
        edgecolor="grey",
        alpha=0.8,
    )
    ax_top_left.tick_params(axis="both", labelsize=14)
    ax_top_left.set_ylabel("Count", fontsize=16)  # 设置y轴标签
    for ax in [ax_main, ax_top, ax_right, ax_left, ax_top_left]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

    plt.savefig(save_path, dpi=600, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)
