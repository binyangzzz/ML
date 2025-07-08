# import pandas as pd
#
# # 读取Excel文件
# df = pd.read_excel('train2.xlsx')  # 替换为你的Excel文件名
#
# # 计算每个特征与Index的相关性
# correlations = df.corr()['Index'].sort_values(ascending=False)
#
# # 选择相关性最高的8个特征（不包括Index本身）
# top_correlated_features = correlations[1:9].index.tolist()  # Index 0是Index自身，所以跳过
#
# # 将选定的特征和Index一起提取出来
# selected_features_df = df[['Index'] + top_correlated_features]
#
# # 将提取的数据存储到新的Excel文件中
# selected_features_df.to_excel('top_correlated_features.xlsx', index=False)

import pandas as pd

# 读取测试集Excel文件
test_df = pd.read_excel('train2.xlsx')  # 替换为你的测试集Excel文件名

# 假设你之前已经得到了相关性最高的8个特征列表 top_correlated_features
# 如果没有，请确保你有这个列表，例如：
top_correlated_features = ['Dst_host_srv_count', 'Logged_in', 'Dst_host_same_srv_rate', 'Dst_host_count', 'Dst_host_srv_serror_rate', 'Srv_serror_rate', 'Serror_rate', 'Dst_host_serror_rate']

# 提取选定的特征和Index
selected_columns = ['Index'] + top_correlated_features
test_selected_features_df = test_df[selected_columns]

# 将提取的数据存储到新的Excel文件中
test_selected_features_df.to_excel('test_top_correlated_features.xlsx', index=False)
