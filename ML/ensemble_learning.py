import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm

# 读取数据
df = pd.read_excel("train.xlsx")

# 提取特征和标签
X = df.iloc[:, 1:].values  # 特征
y = df.iloc[:, 0].values    # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林模型
start_time_rf = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 使用tqdm添加进度条
with tqdm(total=len(X_train)) as pbar:
    rf_model.fit(X_train, y_train)
    pbar.update(len(X_train))

rf_predictions = rf_model.predict(X_test)
end_time_rf = time.time()
rf_time = end_time_rf - start_time_rf

# 第一个神经网络模型
start_time_nn1 = time.time()
nn_model = Sequential()
nn_model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
nn_predictions = nn_model.predict(X_test).flatten()
end_time_nn1 = time.time()
nn_time1 = end_time_nn1 - start_time_nn1

# 计算第一个神经网络模型的RMSE
nn_rmse1 = np.sqrt(mean_squared_error(y_test, np.round(nn_predictions)))

# 构建新特征
X_combined = np.column_stack((rf_predictions, nn_predictions))

# 集成神经网络模型
start_time_ensemble = time.time()
ensemble_model = Sequential()
ensemble_model.add(Dense(64, activation='relu', input_shape=(X_combined.shape[1],)))
ensemble_model.add(Dense(32, activation='relu'))
ensemble_model.add(Dense(1, activation='sigmoid'))
ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ensemble_model.fit(X_combined, y_test, epochs=100, batch_size=32, verbose=0)
ensemble_predictions = ensemble_model.predict(X_combined).flatten()
end_time_ensemble = time.time()
ensemble_time = end_time_ensemble - start_time_ensemble

# 计算集成神经网络模型的准确率
ensemble_accuracy = accuracy_score(y_test, np.round(ensemble_predictions))

# 打印结果
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("First Neural Network Accuracy:", accuracy_score(y_test, np.round(nn_predictions)))
print("Ensemble Model Accuracy:", ensemble_accuracy)
print("Random Forest Time:", rf_time)
print("First Neural Network Time:", nn_time1)
print("Ensemble Model Time:", ensemble_time)
print("First Neural Network RMSE:", nn_rmse1)

ensemble_rmse = np.sqrt(mean_squared_error(y_test, np.round(ensemble_predictions)))
print("Ensemble Model RMSE:", ensemble_rmse)
