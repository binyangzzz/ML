import pandas as pd
import numpy as np

# 读取 Excel 数据集，假设数据集中第一列为标签，后十二列为特征
data = pd.read_excel("your_data.xlsx")

# 选择特征列进行 Z-score 标准化处理
features = data.iloc[:, 1:]  # 选择除了第一列标签外的所有列作为特征
z_scores = (features - features.mean()) / features.std()  # 计算 Z-score

# 检测异常值，假设阈值为3
threshold = 3
outliers = (np.abs(z_scores) > threshold).any(axis=1)  # 检测是否有任何一列的 Z-score 超过阈值

# 打印异常值
print("异常值数量：", outliers.sum())
print("异常值索引：", data[outliers].index.tolist())
