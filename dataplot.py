# import pandas as pd
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib import font_manager
# import scipy.stats


# font_path = "times+simsun.ttf"
# font_manager.fontManager.addfont(font_path)
# prop = font_manager.FontProperties(fname=font_path)
# font_options = {
#     "family": "sans-serif",  # 设置字体家族
#     "sans-serif": prop.get_name(),  # 设置字体
#     "size": 16,  # 设置字体大小
# }
# plt.rc("font", **font_options)


# # 从文档中提取筋类型数据
# # 根据分析报告，筋类型分布如下：
# steel_types = ["黏砂", "带肋", "光圆"]
# counts = [140, 323, 12]
# percentages = [29.4, 68.0, 2.5]

# # 创建图形
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# # 颜色配置
# colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

# # 1. 饼图 - 显示占比
# wedges, texts, autotexts = ax1.pie(
#     counts,
#     labels=steel_types,
#     colors=colors,
#     autopct="%1.1f%%",
#     startangle=90,
#     textprops={"fontsize": 11},
# )
# ax1.set_title("FRP筋表面类型分布占比", fontsize=16, fontweight="bold", pad=20)

# # 美化饼图文字
# for autotext in autotexts:
#     autotext.set_color("white")
#     autotext.set_fontweight("bold")

# # 2. 柱状图 - 显示具体数量
# bars = ax2.bar(
#     steel_types,
#     counts,
#     color=colors,
#     alpha=0.8,
#     edgecolor="black",
#     linewidth=1,
#     width=0.6,
# )
# ax2.set_title("FRP筋表面类型试验数量分布", fontsize=16, fontweight="bold", pad=20)
# ax2.set_ylabel("试验组数", fontsize=16)
# ax2.set_xlabel("筋类型", fontsize=16)

# # 在柱状图上添加数值标签
# for bar, count, pct in zip(bars, counts, percentages):
#     height = bar.get_height()
#     ax2.text(
#         bar.get_x() + bar.get_width() / 2.0,
#         height + 5,
#         f"{count}组\n({pct}%)",
#         ha="center",
#         va="bottom",
#         fontsize=11,
#         fontweight="bold",
#     )

# # 设置y轴范围
# ax2.set_ylim(0, max(counts) * 1.15)

# # 添加网格线
# ax2.grid(axis="y", alpha=0.3, linestyle="--")

# # 调整布局
# plt.tight_layout()

# # 保存图片
# plt.savefig(
#     "表面类型distribution2.png",
#     dpi=300,
#     bbox_inches="tight",
#     facecolor="white",
#     edgecolor="none",
# )
# plt.close()

# # 输出数据汇总表
# summary_data = pd.DataFrame(
#     {
#         "筋类型": steel_types,
#         "试验组数": counts,
#         "占比(%)": percentages,
#         "占比说明": [
#             "CFRP筋，强度高、耐腐蚀",
#             "GFRP筋，性价比高、应用广",
#             "玄武岩FRP筋，新型材料",
#         ],
#     }
# )

# print("FRP筋类型分布汇总表：")
# print("=" * 60)
# print(summary_data.to_string(index=False))
# print("=" * 60)
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
ax_histx = plt.subplot()

x_min, x_max = 90, 210
y_min, y_max = 0, 70
density = scipy.stats.gaussian_kde(df["混凝土强度fcu"])
sorted_data = df["混凝土强度fcu"].sort_values()
x_smooth = np.linspace(x_min, x_max, 1000)
y_smooth = density(x_smooth)

n, bins, patches = ax_histx.hist(
    df["混凝土强度fcu"],
    bins=10,
    color="#E28D8D",
    edgecolor="black",
    alpha=0.6,
    density=True,
)
plt.plot(bins[:10] + (bins[1] - bins[0]) / 2, n, "--", color="#2ca02c")
plt.show()
