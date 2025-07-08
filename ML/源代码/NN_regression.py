import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

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
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test))

# 绘制损失函数变化趋势
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 在测试集上进行预测
y_pred = model.predict(X_test_scaled)

# 计算均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

y_test = y_test +np.random.normal(0, 0.02, size=y_test.shape)
y_pred = y_pred + np.random.normal(0, 0.02, size=y_pred.shape)

# 绘制散点图
plt.figure(figsize=(10, 6))

# 第二个散点图：测试集标签 vs. 真实值
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test, label='True Labels', alpha=0.5)
plt.scatter(y_test, y_pred, label='Predicted Labels', alpha=0.5)
plt.xlabel('Test Labels')
plt.ylabel('True and Predicted Labels')
plt.title('Test Labels vs. True and Predicted Labels')
plt.legend()

plt.tight_layout()
plt.show()
