import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 读取数据
df = pd.read_excel("filtered_train.xlsx")

# 提取特征和标签
X = df.iloc[:, 1:]  # 特征
y = df.iloc[:, 0]   # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型并计算时间
start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time
print("Training Time:", training_time, "seconds")

# 在测试集上进行预测并计算时间
start_time = time.time()
y_pred = clf.predict(X_test)
end_time = time.time()
prediction_time = end_time - start_time
print("Prediction Time:", prediction_time, "seconds")

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

y_test2 = y_test + np.random.normal(0, 0.02, size=y_test.shape)
y_pred2 = y_pred + np.random.normal(0, 0.02, size=y_pred.shape)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, label='True Labels', alpha=0.5, s=20)
plt.scatter(y_test, y_pred, label='Predicted Labels', alpha=0.5, s=20)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('True vs Predicted Labels')
plt.legend()
plt.show()

# 使用predict_proba获取预测概率
y_prob = clf.predict_proba(X_test)[:, 1]  # 取第二列，即类别1的概率

# 绘制ROC曲线（确保y_test是二分类标签）
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # 使用未修改的y_test
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc="lower right")
plt.show()

# 绘制PR曲线（确保y_test是二分类标签）
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)  # 使用未修改的y_test
plt.figure()
plt.plot(recall, precision, color='navy', lw=2, label='Precision-Recall curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Example')
plt.legend(loc="lower left")
plt.show()