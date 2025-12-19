# 读取test.xslx，以第一列为横坐标，第二列为纵坐标
import pandas as pd

df = pd.read_excel("test.xlsx")
x = df.iloc[:, 0]
y = df.iloc[:, 1]

# 用plt绘制散点图，坐标轴标签为每列的特征名
import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("Scatter Plot of x vs y")
plt.show()
