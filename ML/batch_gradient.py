import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取数据
df = pd.read_excel("train.xlsx")

# 提取特征和标签
X = df.iloc[:, 1:]  # 特征
y = df.iloc[:, 0]   # 标签

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 添加偏置项
X_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=42)

# 初始化参数
theta = np.random.randn(X_train.shape[1])  # 初始参数随机初始化

# 定义批梯度下降函数
def batch_gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    loss_history = []  # 用于存储损失值的列表
    for _ in range(num_iterations):
        gradients = (2/m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradients
        # 计算损失值并存储
        loss = mean_squared_error(y, X.dot(theta))
        loss_history.append(loss)
    return theta, loss_history

# 设置超参数
alpha = 0.01  # 学习率
num_iterations = 1000  # 迭代次数

# 使用批梯度下降训练模型，并获取损失值的变化历史
theta, loss_history = batch_gradient_descent(X_train, y_train, theta, alpha, num_iterations)

# 绘制损失函数变化趋势图
plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), loss_history, color='blue')
plt.title('Loss Function Trend')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 在测试集上进行预测
y_pred = X_test.dot(theta)

y_test = y_test + np.random.normal(0, 0.02, size=y_test.shape)
y_pred = y_pred + np.random.normal(0, 0.02, size=y_pred.shape)
# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, label='True Labels', alpha=0.5, s=10)
plt.scatter(y_test, y_pred, label='Predicted Labels', alpha=0.5, s=10)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('True vs Predicted Labels')
plt.legend()
plt.show()

# 计算均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)
