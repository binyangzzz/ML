import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
train_df = pd.read_excel('train_.xlsx')
test_df = pd.read_excel('test_.xlsx')

# 分离特征和标签
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values  # 保存测试集的真实标签以便可视化对比

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建神经网络模型
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # 增加层数和神经元数量
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))  # 新增层，使用relu激活函数
model.add(Dense(1))  # 回归问题，输出层不使用激活函数

# 编译模型，手动设置学习率
optimizer = Adam(learning_rate=0.0005)  # 手动设置学习率为0.001
model.compile(loss='mean_squared_error', optimizer=optimizer)

# 训练模型并保存历史记录
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, verbose=1)

# 对测试集进行预测
predictions = model.predict(X_test_scaled)
noise_scale = 0.05  # 调整此值以改变扰动的幅度
perturbed_predictions = predictions + np.random.normal(0, noise_scale, size=predictions.shape)

# 评估模型
test_mse = mean_squared_error(y_test, predictions)
print(f'Test Mean Squared Error: {test_mse}')

# 可视化训练过程
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 可视化预测结果与实际目标的对比，使用扰动后的预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, perturbed_predictions, label='Predictions with Perturbation', s=0.5)  # 调整了散点大小
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label='Perfect Prediction Line')
plt.title('Prediction vs Actual with Perturbation')
plt.xlabel('Actual Target')
plt.ylabel('Predicted Target with Perturbation')
plt.legend()
plt.grid(True)  # 添加网格以便更好地查看数据分布
plt.show()