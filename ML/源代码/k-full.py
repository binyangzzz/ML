import pandas as pd
from sklearn.impute import KNNImputer

# 读取Excel文件
data = pd.read_excel('your_dataset.xlsx')

# 提取特征列
features = data.iloc[:, 1:]  # 数据特征从第二列开始

# 初始化KNN填充器
imputer = KNNImputer(n_neighbors=5)

# 使用KNN填充缺失值
filled_features = imputer.fit_transform(features)

# 将填充后的特征重新添加到原始数据中
filled_data = data.copy()
filled_data.iloc[:, 1:] = filled_features

# 保存填充后的数据到新的Excel文件
filled_data.to_excel('filled_dataset.xlsx', index=False)
