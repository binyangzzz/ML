import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# 数据点
x = np.linspace(1, 100, 100)  # 注意这里修改了x的范围，以符合您要求的1到100的坐标轴

# 计算各个函数的值
y_factorial = [factorial(i) for i in x]  # n!
y_exp = 2 ** x  # 2^n
y_nlog2n = x * np.log(x)  # nlog2n
y_n = x  # n
y_sqrt = np.sqrt(x)  # 根号n
y_log2 = np.log(x)  # log2n
y_one = np.ones_like(x)  # 1

# 转换为浮点数数组，以便绘图
y_factorial = np.array(y_factorial, dtype=float)

# 绘图
plt.figure(figsize=(10, 6))

plt.plot(x, y_factorial, label='n!')
plt.plot(x, y_exp, label='2^n')
plt.plot(x, y_nlog2n, label='nlog2n')
plt.plot(x, y_n, label='n')
plt.plot(x, y_sqrt, label='sqrt(n)')
plt.plot(x, y_log2, label='log2n')
plt.plot(x, y_one, label='1')

plt.yscale('log')  # 使用对数刻度
plt.legend()
plt.title('Comparison of Functions')
plt.xlabel('n')
plt.ylabel('Function Value')
plt.grid(True)

# 保留坐标轴线，去掉刻度标签
plt.xticks([], [])  # 去掉x轴刻度标签
plt.yticks([], [])  # 去掉y轴刻度标签
ax = plt.gca()
ax.spines['bottom'].set_position('zero')  # 将x轴移动到y=0的位置
ax.spines['left'].set_position('zero')  # 将y轴移动到x=0的位置
for key, spine in ax.spines.items():
    if key in ['top', 'right']:
        spine.set_visible(False)  # 隐藏上边和右边的坐标轴线

# 显示图表
plt.show()