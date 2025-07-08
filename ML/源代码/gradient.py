import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# 读取数据
df = pd.read_excel("train.xlsx")

# 提取特征和标签
X = df.iloc[:, 1:]  # 特征
y = df.iloc[:, 0]   # 标签



# 添加偏置项
X['bias'] = 1

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 初始化参数
# theta = np.zeros(X_train.shape[1])  # 初始参数全部为0
# alpha = 0.000001  # 学习率
# num_iterations = 4000  # 迭代次数
#
# #定义梯度下降函数
# def gradient_descent(X, y, theta, alpha, num_iterations):
#     m = len(y)
#     for i in tqdm(range(num_iterations)):
#         # 计算预测值
#         y_pred = np.dot(X, theta)
#         # 计算误差
#         error = y_pred - y
#         # 计算梯度
#         gradient = (1/m) * np.dot(X.T, error)
#         # 更新参数
#         theta -= alpha * gradient
#     return theta
#
#
#
# # 使用梯度下降训练模型
# theta = gradient_descent(X_train, y_train, theta, alpha, num_iterations)
#
# # 在测试集上进行预测
# y_pred = np.dot(X_test, theta)
#
# # 计算均方根误差（RMSE）
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("Root Mean Squared Error (RMSE):", rmse)
#
# y_test_with_jitter = y_test + np.random.normal(0, 0.02, size=y_test.shape)
#
# # 可视化结果
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test_with_jitter, y_pred, color='blue',s=0.1)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.title('True vs Predicted values')
# plt.show()


def compute_rmse(X, y, theta):
    m = len(y)
    y_pred = np.dot(X, theta)
    error = y_pred - y
    mse = np.sum(error ** 2) / m
    rmse = np.sqrt(mse)
    return rmse

# 定义梯度下降函数
def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history = []

    for i in tqdm(range(num_iterations)):
        # 计算预测值
        y_pred = np.dot(X, theta)
        # 计算误差
        error = y_pred - y
        # 计算梯度
        gradient = (1/m) * np.dot(X.T, error)
        # 更新参数
        theta -= alpha * gradient
        # 计算损失函数
        cost = compute_rmse(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# 初始化参数
theta = np.zeros(X_train.shape[1])  # 初始参数全部为0
alpha = 0.000001  # 学习率
num_iterations = 4000  # 迭代次数

# 使用梯度下降训练模型
theta, cost_history = gradient_descent(X_train.values, y_train.values, theta, alpha, num_iterations)

final_cost = compute_rmse(X_train.values, y_train.values, theta)
print("Final Cost:", final_cost)

# 绘制损失函数变化图
plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), cost_history, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration')
plt.show()