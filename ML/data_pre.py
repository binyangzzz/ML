import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#plt.rcParams.update({'font.size': 16}) # 设置字体大小
def change_class(excel_file):
    """
    读取Excel文件并将列名为“Class”的一列的数据中的“normal”改为0，其余改为1

    参数：
    - excel_file: Excel文件路径

    返回值：
    - None
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 将列名为“Class”的一列的数据中的“normal”改为0，其余改为1
    df['Class'] = df['Class'].apply(lambda x: 0 if x == 'normal' else 1)

    # 保存修改后的数据到同一Excel文件
    df.to_excel(excel_file, index=False)

    print("数据已成功处理并保存到Excel文件中！")

def encode_columns(excel_file, columns):
    """
    对Excel文件中的指定列进行编码

    参数：
    - excel_file: Excel文件路径
    - columns: 需要进行编码的列名列表

    返回值：
    - None
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 创建LabelEncoder对象
    encoder = LabelEncoder()

    # 对指定列进行编码
    for column in columns:
        df[column] = encoder.fit_transform(df[column])

        # 遍历原始数据
    # for item in data:
    #     # 如果当前类别不在映射字典中，则将其添加进字典，并分配一个唯一的编码
    #     if item not in label_map:
    #         label_map[item] = label_count
    #         label_count += 1
    #
    #     # 将当前类别的编码添加到转换后的数据列表中
    #     transformed_data.append(label_map[item])
    #
    # return transformed_data, label_map

    # 保存修改后的数据到同一Excel文件
    df.to_excel(excel_file, index=False)

    print("数据已成功编码并保存到Excel文件中！")



def calculate_correlation_matrix_and_plot_heatmap(excel_file):
    """
    读取Excel文件并计算特征值的相关性矩阵，并绘制热力图

    参数：
    - excel_file: Excel文件路径

    返回值：
    - None
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 初始化进度条
    pbar = tqdm(total=len(df.columns), desc="Calculating Correlation Matrix")

    # 计算特征值的相关性矩阵
    correlation_matrix = df.corr()

    # 关闭进度条
    pbar.close()

    # 调整热力图尺寸
    plt.figure(figsize=(14, 12))

    # 绘制热力图
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, xticklabels=True, yticklabels=True)
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()



def visualize_class_correlation(excel_file):
    """
    读取Excel文件并计算特征值与 'Class' 的相关性，并绘制柱状图展示相关性排序

    参数：
    - excel_file: Excel文件路径

    返回值：
    - None
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 计算特征值与 'Class' 的相关性
    class_correlation = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_correlation.values, y=class_correlation.index, palette='coolwarm')
    plt.title('Feature Correlation with Class')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Features')
    plt.show()

def filter_low_correlation_features(excel_file, threshold=0.2):
    """
    读取Excel文件并去除与 'Class' 相关性小于指定阈值的特征列，将结果保存到新的Excel文件中

    参数：
    - excel_file: 原始Excel文件路径
    - threshold: 相关性阈值，默认为0.2

    返回值：
    - None
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 计算特征值与 'Class' 的相关性
    class_correlation = df.corr()['Class'].drop('Class').abs()

    # 找出与 'Class' 相关性大于等于阈值的特征列
    selected_features = class_correlation[class_correlation >= threshold].index.tolist()

    # 将 'Class' 列移动到DataFrame的第一列
    selected_features.insert(0, 'Class')

    # 重新排列DataFrame的列顺序
    filtered_df = df[selected_features]

    # 保存到新的Excel文件
    new_excel_file = 'filtered_' + excel_file  # 新的Excel文件名
    filtered_df.to_excel(new_excel_file, index=False)

    print("数据已成功筛选并保存到新的Excel文件中：", new_excel_file)


def draw_boxplots(excel_file):
    """
    读取Excel文件并为每一列数据绘制箱型图

    参数：
    - excel_file: Excel文件路径

    返回值：
    - None
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 提取特征列名
    feature_columns = df.columns[1:]  # 第一列为数据结果，后面的列为特征列

    # 绘制箱型图
    plt.figure(figsize=(12, 8))
    for column in feature_columns:
        plt.boxplot(df[column], labels=[column])
        plt.title('Boxplot of Features')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.show()

# 调用函数并传入Excel文件路径
excel_file_path = 'data_choose.xlsx'  # 替换为你的Excel文件路径
columns_to_encode = ['Protocol_type', 'Service', 'Flag']
#change_class(excel_file_path) #class转int
#encode_columns(excel_file_path, columns_to_encode) #str编码
calculate_correlation_matrix_and_plot_heatmap(excel_file_path) #计算相关性矩阵
#visualize_class_correlation(excel_file_path) #计算和class相关的数据
#filter_low_correlation_features(excel_file_path) #过滤掉与class相关性过小的数据列
#draw_boxplots(excel_file_path) #绘制箱型图以判断是否有离群数据