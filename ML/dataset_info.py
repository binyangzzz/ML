import openpyxl


def read_and_process_file(input_file):
    # 读取文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 创建一个列表，用于存储处理后的数据
    processed_data = []

    # 遍历每一行数据并处理
    for index, line in enumerate(lines, start=1):
        # 去除换行符
        line = line.strip()
        # 切分数据，以冒号为界限
        parts = line.split(':')
        # 将序号、冒号前的数据和冒号后的数据加入到处理后的列表中
        processed_data.append([index, parts[0], parts[1]])

    return processed_data


def write_to_excel(data, output_file):
    # 创建一个新的Excel工作簿
    wb = openpyxl.Workbook()
    # 获取默认的工作表
    ws = wb.active
    # 写入表头
    ws.append(['序号', '冒号前的数据', '冒号后的数据'])
    # 写入数据
    for row in data:
        ws.append(row)
    # 保存Excel文件
    wb.save(output_file)
    print(f"处理完成，结果已写入到 {output_file} 文件中。")


def main():
    input_file = 'dataset.txt'
    output_file = 'dataset_info.xlsx'
    processed_data = read_and_process_file(input_file)
    write_to_excel(processed_data, output_file)


if __name__ == "__main__":
    main()
