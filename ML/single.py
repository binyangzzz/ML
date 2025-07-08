# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import stats
# import numpy as np
#
# # 读取Excel文件
# excel_file = 'train.xlsx'  # 替换为您的Excel文件名
# column_name = 'Srv_count'  # 替换为您要绘制分布的列名
# df = pd.read_excel(excel_file, engine='openpyxl')
#
# # 过滤掉大于1000的数据
# df_filtered = df[df[column_name] <= 200]
#
# # 绘制数据分布
# plt.figure(figsize=(10, 6))
# plt.hist(df_filtered[column_name], bins=30, edgecolor='black', range=(0, 200))  # 设置直方图的范围从0到1000
# plt.title(f'Distribution of {column_name}')
# plt.xlabel(column_name)
# plt.ylabel('Frequency')
# plt.xlim(0, 200)  # 设置x轴的范围从0到1000
# plt.grid(True)
# plt.show()
#
#
# # # 过滤掉大于1000和等于255的数据
# # df_filtered = df[df[column_name] != 255]
# #
# # # 绘制数据分布
# # plt.figure(figsize=(10, 6))
# # plt.hist(df_filtered[column_name], bins=30, edgecolor='black', density=True, alpha=0.7,
# #          label='Data Distribution')  # density=True 用于直方图归一化
# #
# # # 正态分布拟合
# # mu, std = stats.norm.fit(df_filtered[column_name])  # 计算均值和标准差
# # xmin, xmax = plt.xlim()  # 获取当前x轴的范围
# # x = np.linspace(xmin, xmax, 100)
# # p = stats.norm.pdf(x, mu, std)  # 生成正态分布的概率密度函数
# #
# # # 在直方图上绘制正态分布曲线
# # plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')
# #
# # # 设置图表标题和坐标轴标签
# # plt.title(f'Distribution of {column_name} and Fitted Normal Distribution')
# # plt.xlabel(column_name)
# # plt.ylabel('Frequency')
# #
# # # 添加图例
# # plt.legend()
# #
# # # 显示网格
# # plt.grid(True)
# #
# # # 显示图表
# # plt.show()
# #
# # # 进行正态性检验，例如使用Shapiro-Wilk测试
# # shapiro_stat, shapiro_p = stats.shapiro(df_filtered[column_name])
# # print(f"Shapiro-Wilk Test Statistic: {shapiro_stat}, P-Value: {shapiro_p}")
# #
# # # 根据P值判断数据是否符合正态分布
# # if shapiro_p > 0.05:
# #     print("数据近似服从正态分布")
# # else:
# #     print("数据不服从正态分布")

import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
excel_file = 'train.xlsx'  # 替换为你的Excel文件名
sheet_name = 'Sheet1'  # 如果有多个工作表，请指定正确的工作表名
column_name = 'Flag'  # 替换为你的列名

df = pd.read_excel(excel_file, sheet_name=sheet_name)

# 计算每个唯一值的出现次数
value_counts = df[column_name].value_counts()

# 计算每个类别的占比
total_count = value_counts.sum()
percentage_counts = (value_counts / total_count) * 100

# 将2%以下的数据归为一类
small_percentages = percentage_counts[percentage_counts < 2]
other_count = small_percentages.sum()
percentage_counts = percentage_counts[percentage_counts >= 2]
percentage_counts['Other'] = other_count

# 重新计算占比，以考虑“其他”类别的加入
total_percentage = percentage_counts.sum()
percentage_counts = (percentage_counts / total_percentage) * 100

plt.rcParams.update({'font.size': 14})  # 调整这里的数字来改变字体大小

# 绘制饼图
plt.figure(figsize=(10, 6))
plt.pie(percentage_counts, labels=percentage_counts.index, autopct='%1.1f%%', startangle=140)

# 设置标题
plt.title('Distribution of Values in Column (Including "Other")')

# 显示图表
plt.show()