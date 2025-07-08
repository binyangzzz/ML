import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

# 读取数据
df = pd.read_excel("train.xlsx")

# 计算特征与标签之间的相关系数
correlations = df.corr().iloc[1:, 0]  # 计算每个特征与标签的相关系数
correlation_abs_max_feature = correlations.abs().idxmax()  # 找到相关性绝对值最大的特征
print("Correlation absolute max feature:", correlation_abs_max_feature)

# 提取所有特征和标签
X = df.iloc[:, 1:]  # 所有特征
y = df.iloc[:, 0]      # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型并使用所有特征进行训练
model = LinearRegression()

start_time = time.time()  # 记录训练开始时间
model.fit(X_train, y_train)
train_time = time.time() - start_time  # 记录训练时间

# 在测试集上进行预测
start_time = time.time()  # 记录预测开始时间
y_pred = model.predict(X_test)
predict_time = time.time() - start_time  # 记录预测时间

# 提取与标签相关性最高的特征
X_selected = X_test[correlation_abs_max_feature].values.reshape(-1, 1)

# 计算均方根误差（RMSE）
rmse_selected = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE) with selected feature:", rmse_selected)

# 加上计算时间的代码
print("Training time:", train_time, "seconds")
print("Prediction time:", predict_time, "seconds")

# 预测结果加上阈值
y_pred_threshold = np.where(y_pred > 0.5, 1, 0)

# 计算准确率
accuracy = np.mean(y_pred_threshold == y_test)
print("Accuracy:", accuracy)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X_selected, y_test, color='blue', label='True values', s=0.1)
plt.scatter(X_selected, y_pred, color='red', label='Predicted values', s=0.1)
plt.xlabel(correlation_abs_max_feature)
plt.ylabel('Label')
plt.title('True vs Predicted values using selected feature')
plt.legend()
plt.show()
