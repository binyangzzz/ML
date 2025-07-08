import pandas as pd
import numpy as np
import random
import missingno as msno
import matplotlib.pyplot as plt

# # 读取Excel文件
# df = pd.read_excel('train.xlsx')  # 请将'your_file.xlsx'替换为您的Excel文件名
#
# # 计算需要制造缺失值的数量
# total_rows = len(df)
# missing_values_col56 = int(0.08 * total_rows)  # 第五列和第六列各需要制造的缺失值数量
# missing_values_col1 = int(0.04 * total_rows)  # 第一列需要制造的缺失值数量
#
# # 随机选择行来制造缺失值
# rows_to_nan_col56 = random.sample(range(total_rows), missing_values_col56 * 2)  # 选择行号
# rows_to_nan_col1 = random.sample(range(total_rows), missing_values_col1)  # 选择行号
#
# # 为第五列和第六列制造缺失值
# for row in rows_to_nan_col56[:missing_values_col56]:  # 为第五列制造缺失值
#     df.at[row, df.columns[4]] = np.nan
# for row in rows_to_nan_col56[missing_values_col56:]:  # 为第六列制造缺失值
#     df.at[row, df.columns[5]] = np.nan
#
# # 为第一列制造缺失值
# for row in rows_to_nan_col1:
#     df.at[row, df.columns[0]] = np.nan
#
# # 使用missingno进行缺失值可视化
# msno.matrix(df[df.columns[0:6]])  # 可视化前六列数据的缺失值矩阵图
# plt.figure(figsize=(10, 8), dpi=100)
# plt.savefig('missing_matrix.png')  # 可选：将矩阵图保存到文件
# plt.show()  # 显示矩阵图
#
# msno.bar(df[df.columns[0:6]])  # 可视化前六列数据的缺失值条形图
# plt.savefig('missing_bar.png')  # 可选：将条形图保存到文件
# plt.show()  # 显示条形图
#
# # 将前六列数据保存到新的Excel文件中
# df[df.columns[0:6]].to_excel('modified_data.xlsx', index=False)

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

# 读取Excel文件的前1000行
df = pd.read_excel('modified_data.xlsx', nrows=1000)  # 替换'your_file.xlsx'为你的Excel文件名

# 使用missingno进行缺失值可视化
# 设置图形大小
plt.figure(figsize=(10, 4))

# 绘制缺失值矩阵图
msno.matrix(df)
plt.title('Missing Values Matrix for First 1000 Rows')
plt.show()

# 绘制缺失值条形图
plt.figure(figsize=(12, 6))
msno.bar(df)
plt.title('Missing Values Bar Chart for First 1000 Rows')
plt.show()