import pandas as pd

# 读取txt文件
txt_file = 'train.txt'  # 替换为你的txt文件路径
data = []
with open(txt_file, 'r') as file:
    for line in file:
        # 移除换行符并按逗号分隔特征值
        features = line.strip().split(',')
        data.append(features)

# 定义列名
columns = [
    'Duration', 'Protocol_type', 'Service', 'Flag', 'Src_bytes', 'Dst_bytes', 'Land',
    'Wrong_fragment', 'Urgent', 'Hot', 'Num_failed_logins', 'Logged_in', 'Num_compromised',
    'Root_shell', 'Su_attempted', 'Num_root', 'Num_file_creations', 'Num_shells',
    'Num_access_files', 'Num_outbound_cmds', 'Is_host_login', 'Is_guest_login', 'Count',
    'Srv_count', 'Serror_rate', 'Srv_serror_rate', 'Rerror_rate', 'Srv_rerror_rate',
    'Same_srv_rate', 'Diff_srv_rate', 'Srv_diff_host_rate', 'Dst_host_count',
    'Dst_host_srv_count', 'Dst_host_same_srv_rate', 'Dst_host_diff_srv_rate',
    'Dst_host_same_src_port_rate', 'Dst_host_srv_diff_host_rate', 'Dst_host_serror_rate',
    'Dst_host_srv_serror_rate', 'Dst_host_rerror_rate', 'Dst_host_srv_rerror_rate',
    'Class', 'Index'
]

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=columns)

# 将数字特征转换为数值形式
numeric_features = [
    'Duration', 'Src_bytes', 'Dst_bytes', 'Land', 'Wrong_fragment', 'Urgent', 'Hot',
    'Num_failed_logins', 'Logged_in', 'Num_compromised', 'Root_shell', 'Su_attempted',
    'Num_root', 'Num_file_creations', 'Num_shells', 'Num_access_files', 'Num_outbound_cmds',
    'Is_host_login', 'Is_guest_login', 'Count', 'Srv_count', 'Serror_rate', 'Srv_serror_rate',
    'Rerror_rate', 'Srv_rerror_rate', 'Same_srv_rate', 'Diff_srv_rate', 'Srv_diff_host_rate',
    'Dst_host_count', 'Dst_host_srv_count', 'Dst_host_same_srv_rate', 'Dst_host_diff_srv_rate',
    'Dst_host_same_src_port_rate', 'Dst_host_srv_diff_host_rate', 'Dst_host_serror_rate',
    'Dst_host_srv_serror_rate', 'Dst_host_rerror_rate', 'Dst_host_srv_rerror_rate'
]
df[numeric_features] = df[numeric_features].astype(float)




# 将数据写入Excel文件
excel_file = 'train.xlsx'  # 输出的Excel文件路径
df.to_excel(excel_file, index=False)

print("数据已成功写入Excel文件！")
