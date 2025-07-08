import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import mean_squared_error
import time

# 读取数据
df = pd.read_excel("train.xlsx")

# 提取特征和标签
X = df.iloc[:, 1:]  # 特征
y = df.iloc[:, 0]   # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建SVM分类器
clf = SVC(kernel='linear')  # 默认在GPU上运行

# 计算时间
start_time = time.time()

# 训练模型
with tqdm(total=len(X_train)) as pbar:
    clf.fit(X_train, y_train)
    pbar.update(len(X_train))

end_time = time.time()
print("Time taken for training:", end_time - start_time, "seconds")

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 计算均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 打印分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 加入随机噪声
y_test = y_test + np.random.normal(0, 0.02, size=y_test.shape)
y_pred = y_pred + np.random.normal(0, 0.02, size=y_pred.shape)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, label='True Labels', alpha=0.5, s=5)
plt.scatter(y_test, y_pred, label='Predicted Labels', alpha=0.5, s=5)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('True vs Predicted Labels')
plt.legend()
plt.show()
