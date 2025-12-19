import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import font_manager
import scipy.stats


font_path = "times+simsun.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
font_options = {
    "family": "sans-serif",  # 设置字体家族
    "sans-serif": prop.get_name(),  # 设置字体
    "size": 12,  # 设置字体大小
}
plt.rc("font", **font_options)
# 读入随机数据
df = pd.read_excel("./fpr筋机器学习预处理cn.xlsx")
fig = plt.figure(figsize=(8, 8))
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.5)

# 创建子图
ax_joint = plt.subplot(grid[1:4, 0:3])
ax_histx = plt.subplot(grid[0, 0:3])
ax_histy = plt.subplot(grid[1:4, 3])

# 创建中间的联合密度图
sns.kdeplot(
    data=df,
    x="混凝土强度fcu",
    y="粘结强度",
    cmap="Blues",
    fill=True,
    ax=ax_joint,
    alpha=0.6,
)
# sns.kdeplot(data=df, x="x2", y="y2", cmap="Reds", fill=True, ax=ax_joint, alpha=0.6)

# 创建顶部直方图
# sns.kdeplot(
#     data=df,
#     x="混凝土强度fcu",
#     palette=["#C8E9EF"],
#     fill=True,
#     ax=ax_histx,
#     alpha=0.6,
# )
# ax_histx.hist(
#     df["混凝土强度fcu"],
#     bins=15,
#     color="#C8E9EF",
#     edgecolor="black",
#     alpha=0.6,
#     density=True,
# )
x_min, x_max = 90, 210
y_min, y_max = 0, 70
density = scipy.stats.gaussian_kde(df["混凝土强度fcu"])
sorted_data = df["混凝土强度fcu"].sort_values()
x_smooth = np.linspace(x_min, x_max, 1000)
y_smooth = density(x_smooth)

ax_histx.fill_between(x_smooth, y_smooth, color="#68b88e", alpha=0.6)
# ax_histx.hist(df["x2"], bins=25, color='#E28D8D', edgecolor='black', alpha=0.6, density=True)

# 创建右侧直方图
density = scipy.stats.gaussian_kde(df["粘结强度"])
sorted_data = df["粘结强度"].sort_values()
x_smooth = np.linspace(y_min, y_max, 1000)
y_smooth = density(x_smooth)

ax_histy.fill_betweenx(x_smooth, y_smooth, color="#ee4863", alpha=0.6)
# ax_histy.hist(
#     df["粘结强度"],
#     bins=20,
#     color="#C8E9EF",
#     edgecolor="black",
#     alpha=0.6,
#     orientation="horizontal",
#     density=True,
# )

# ax_histy.hist(df["y2"], bins=40, color='#E28D8D', edgecolor='black', alpha=0.6, orientation='horizontal', density=True)

# 设置坐标轴范围
ax_histx.set_xlim(x_min, x_max)

ax_histy.set_ylim(y_min, y_max)

# 设置坐标限制以对齐
ax_joint.set_xlim(ax_histx.get_xlim())
ax_joint.set_ylim(ax_histy.get_ylim())

# 设置 colorbar 的位置
axins1 = inset_axes(
    ax_joint,
    width="60%",
    height="15%",
    bbox_to_anchor=(0.98, 1.1, 0.4, 0.1),
    bbox_transform=ax_joint.transAxes,
)
# axins2 = inset_axes(
#     ax_joint,
#     width="60%",
#     height="15%",
#     bbox_to_anchor=(0.98, 1.06, 0.4, 0.1),
#     bbox_transform=ax_joint.transAxes,
# )

# 创建 ScalarMappable 对象
norm1 = plt.Normalize(vmin=0, vmax=1)
sm1 = plt.cm.ScalarMappable(cmap="Blues", norm=norm1)
sm1.set_array([])

norm2 = plt.Normalize(vmin=0, vmax=1)
sm2 = plt.cm.ScalarMappable(cmap="Reds", norm=norm2)
sm2.set_array([])

# 添加 colorbar
cbar1 = plt.colorbar(sm1, cax=axins1, orientation="horizontal")
# cbar2 = plt.colorbar(sm2, cax=axins2, orientation="horizontal")
# 隐藏 colorbar 的刻度和标签
cbar1.ax.tick_params(labelbottom=False, labeltop=False, bottom=False, top=False)

# 设置刻度线
ax_joint.xaxis.set_ticks_position("bottom")
ax_joint.yaxis.set_ticks_position("left")
ax_histx.xaxis.set_ticks_position("bottom")
ax_histx.yaxis.set_ticks_position("left")
ax_histy.xaxis.set_ticks_position("bottom")
ax_histy.yaxis.set_ticks_position("left")
# 设置刻度线的粗细
ax_joint.tick_params(axis="both", which="major", width=2, size=6, labelsize=14)
ax_histx.tick_params(axis="both", which="major", width=2, size=6, labelsize=14)
ax_histy.tick_params(axis="both", which="major", width=2, size=6, labelsize=14)

# 移除边际图上侧和右侧的坐标轴
ax_histx.spines["top"].set_visible(False)
ax_histx.spines["right"].set_visible(False)
ax_histy.spines["top"].set_visible(False)
ax_histy.spines["right"].set_visible(False)
# 设置坐标轴线的粗细
ax_joint.spines["top"].set_linewidth(2)
ax_joint.spines["bottom"].set_linewidth(2)
ax_joint.spines["left"].set_linewidth(2)
ax_joint.spines["right"].set_linewidth(2)
ax_histx.spines["bottom"].set_linewidth(2)
ax_histx.spines["left"].set_linewidth(2)
ax_histy.spines["bottom"].set_linewidth(2)
ax_histy.spines["left"].set_linewidth(2)

# 设置标签和图例
ax_joint.set_xlabel("X_lab", fontsize=18)
ax_joint.set_ylabel(
    "Y_lab",
    fontsize=18,
)
ax_histx.set_ylabel(
    "Density",
    fontsize=18,
)
ax_histy.set_xlabel(
    "Density",
    fontsize=18,
)
# ax_histx.legend(
#     ["混凝土强度fcu"],  # , "GroupB"],
#     loc="upper right",
#     frameon=False,
#     bbox_to_anchor=(1.4, 0.95),
#     fontsize=12,
# )

plt.show()
