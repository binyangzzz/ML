import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('train.xlsx')

# 假设'Index'列是我们分组的依据
grouped = df.groupby('Index')

# 找到最大的组
largest_group = max(grouped, key=lambda x: len(x[1]))
largest_group_data = largest_group[1]

# 将最大的组数据保存到新的Excel文件中
largest_group_data.to_excel('largest_group.xlsx', index=False)

# 对最大的组进行离群点检测，这里以'Count'列为例
values = largest_group_data['Count']
z_scores = stats.zscore(values)
outliers = largest_group_data[z_scores.abs() > 3]  # 假设Z-score的绝对值大于3为离群点
normal_points = largest_group_data[z_scores.abs() <= 3]  # 非离群点

# 打印离群点
print(outliers)

# 如果需要将离群点也保存到Excel文件中
outliers.to_excel('outliers.xlsx', index=False)

# 可视化离群点
plt.figure(figsize=(10, 6))
plt.scatter(normal_points.index, normal_points['Count'], color='blue', label='Normal Points', alpha=0.5)
plt.scatter(outliers.index, outliers['Count'], color='red', label='Outliers', marker='^')
plt.xlabel('Index')
plt.ylabel('Count')
plt.title('Outlier Detection in the Largest Group')
plt.legend()
plt.show()
print("Number of outliers:", len(outliers))
