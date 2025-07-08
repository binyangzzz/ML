# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# def stochastic_gradient_descent(X_train, y_train, theta, alpha, num_iterations):
#     m = len(y_train)
#     cost_history = []
#
#     for i in range(num_iterations):
#         # 随机选择一个样本
#         rand_idx = np.random.randint(0, m)
#         X_rand = X_train[rand_idx, :].reshape(1, -1)
#         y_rand = y_train.iloc[rand_idx]
#
#         # 计算预测值
#         y_pred = np.dot(X_rand, theta)
#
#         # 计算误差
#         error = y_pred - y_rand
#
#         # 计算梯度
#         gradient = np.dot(X_rand.T, error)
#
#         # 更新参数
#         theta -= alpha * gradient
#
#         # 计算损失函数
#         cost = np.mean((error)**2)
#         cost_history.append(cost)
#
#     return theta, cost_history
#
# # 读取数据
# df = pd.read_excel("train.xlsx")
#
# # 提取特征和标签
# X = df.iloc[:, 1:]  # 特征
# y = df.iloc[:, 0]   # 标签
#
# # 数据标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 添加偏置项
# X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # 初始化参数
# theta = np.zeros(X_train.shape[1])  # 初始参数全部为0
# alpha = 0.001  # 学习率
# num_iterations = 4000  # 迭代次数
#
# # 执行随机梯度下降
# theta_final, cost_history = stochastic_gradient_descent(X_train, y_train, theta, alpha, num_iterations)
#
# # 在测试集上进行预测
# y_pred = np.dot(X_test, theta_final)
#
# # 计算均方根误差（RMSE）
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("Root Mean Squared Error (RMSE):", rmse)
#
# # 可视化损失函数的变化过程
# plt.figure(figsize=(10, 6))
# plt.plot(cost_history, color='blue')
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Cost vs Iteration')
# plt.show()
#
# # 可视化随机梯度下降过程中的预测标签值变化
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted Label')
# plt.xlabel('Sample Index')
# plt.ylabel('Predicted Label')
# plt.title('Predicted Label vs Iteration')
# plt.legend()
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

def stochastic_gradient_descent(X_train, y_train, theta, alpha, num_iterations):
    m = len(y_train)
    cost_history = []

    for i in range(num_iterations):
        # 随机选择一个样本
        rand_idx = np.random.randint(0, m)
        X_rand = X_train[rand_idx, :].reshape(1, -1)
        y_rand = y_train.iloc[rand_idx]

        # 计算预测值
        y_pred = np.dot(X_rand, theta)

        # 计算误差
        error = y_pred - y_rand

        # 计算梯度
        gradient = np.dot(X_rand.T, error)

        # 更新参数
        theta -= alpha * gradient

        # 计算损失函数
        cost = np.mean((error)**2)
        cost_history.append(cost)

    return theta, cost_history

# 读取数据
df = pd.read_excel("train.xlsx")

# 提取特征和标签
X = df.iloc[:, 1:]  # 特征
y = df.iloc[:, 0]   # 标签

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 添加偏置项
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化参数
theta = np.zeros(X_train.shape[1])  # 初始参数全部为0
alpha = 0.001  # 学习率
num_iterations = 4000  # 迭代次数

# 计算时间
start_time = time.time()

# 执行随机梯度下降
theta_final, cost_history = stochastic_gradient_descent(X_train, y_train, theta, alpha, num_iterations)

end_time = time.time()
print("Time taken for training:", end_time - start_time, "seconds")

# 保留最小损失函数值
min_cost = min(cost_history)

# 可视化损失函数的变化过程（仅保留最小值）
plt.figure(figsize=(10, 6))
plt.plot(cost_history, color='blue')
plt.scatter(cost_history.index(min_cost), min_cost, color='red', label='Minimum Cost')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration')
plt.legend()
plt.show()

# 在测试集上进行预测
y_pred = np.dot(X_test, theta_final)

# 预测结果加上阈值
y_pred_threshold = np.where(y_pred > 0.5, 1, 0)

# 计算准确率
accuracy = np.mean(y_pred_threshold == y_test)
print("Accuracy:", accuracy)

# 计算均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# 可视化预测值和真实值对比
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='True Label', s=10)
plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Label', s=10)
plt.xlabel('Sample Index')
plt.ylabel('Label')
plt.title('True vs Predicted values (Last Iteration)')
plt.legend()
plt.show()