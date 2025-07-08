import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import time
from sklearn.metrics import mean_squared_error

# 读取数据
df = pd.read_excel("train.xlsx")

# 提取特征和标签
X = df.iloc[:, 1:].values  # 特征
y = df.iloc[:, 0].values    # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 建立神经网络模型
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))  # 添加一个隐藏层
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 分类问题需要使用 sigmoid 激活函数

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型并计算时间
start_time = time.time()
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test))
end_time = time.time()
training_time = end_time - start_time
print("Training Time:", training_time, "seconds")

# 在测试集上进行预测并计算时间
start_time = time.time()
y_pred_prob = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)  # 将概率转换为类标签
end_time = time.time()
prediction_time = end_time - start_time
print("Prediction Time:", prediction_time, "seconds")

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

y_test = y_test + np.random.normal(0, 0.02, size=y_test.shape)
y_pred = y_pred + np.random.normal(0, 0.02, size=y_pred.shape)

# 绘制两个散点图在同一图中
plt.figure(figsize=(10, 6))

# 散点图1：横坐标为测试集标签，纵坐标为测试集标签
plt.scatter(y_test, y_test, label='True Labels', alpha=0.5, s=0.1, color='blue', )

# 散点图2：横坐标为测试集标签，纵坐标为其真实值
plt.scatter(y_test, y_pred, label='Predicted Labels', alpha=0.5, s=0.1, color='red')

plt.xlabel('True Labels')
plt.ylabel('Labels')
plt.title('True Labels vs Predicted Labels')
plt.legend()
plt.show()
