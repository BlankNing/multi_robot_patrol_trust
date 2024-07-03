import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 假设我们有以下浮点数列表
data = [1.2, 2.3, 3.1, 2.8, 1.9, 3.5, 2.7, 2.2, 1.5, 3.8, 2.9, 3.2, 1.8, 2.5, 3.7]

# 创建图形和子图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制柱状图
n, bins, patches = ax1.hist(data, bins=10, edgecolor='black', alpha=0.7)

# 设置柱状图的 y 轴标签
ax1.set_ylabel('Frequency')

# 创建一个共享 x 轴的次坐标轴
ax2 = ax1.twinx()

# 计算核密度估计
kde = stats.gaussian_kde(data)
x_range = np.linspace(min(data), max(data), 100)
y_kde = kde(x_range)

# 绘制分布曲线
ax2.plot(x_range, y_kde, 'r-', linewidth=2)

# 设置分布曲线的 y 轴标签
ax2.set_ylabel('Density')

# 设置图表标题和 x 轴标签
plt.title('Histogram and Distribution Curve')
ax1.set_xlabel('Value')

# 显示图形
plt.show()