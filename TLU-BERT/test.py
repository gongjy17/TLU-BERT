import pandas as pd
import re
import os
import glob

# # 读取 CSV 文件
# csv_file_path = '/home/gongjingyuan/PycharmProjects/IMD-IDS/Combined/label_syslog_combined.csv'  # 替换为您的 CSV 文件路径
# df = pd.read_csv(csv_file_path)
#
# # 确保“Content”列存在
# if 'Content' in df.columns:
#     # 定义函数来处理字符串
#     def clean_text(text):
#         # 删除 "," 和 "-" 字符
#         text = text.replace('"', '').replace('-', '').replace(',', '')
#         # 使用正则表达式替换指定字符为单个空格
#         text = re.sub(r'[\/\[\]()<>:]', ' ', text)
#         # 返回处理后的文本
#         return text.strip()
#
#     # 应用处理函数到'Content'列
#     df['Cleaned_Content'] = df['Content'].apply(clean_text)
#
#     # 将处理后的字符串逐行写入 TXT 文件
#     with open('/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/vocab/syslog.txt', 'w', encoding='utf-8') as f:
#         for line in df['Cleaned_Content']:
#             f.write(line + '\n')
#
#     print("内容成功写入到 output.txt 文件中")
# else:
#     print("CSV 文件中没有 'Content' 列")

# # 设置要合并的 TXT 文件夹路径
# folder_path = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/vocab'  # 替换为您的文件夹路径
# output_file_path = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/vocab/vocab_dataset.txt'  # 输出文件名
#
# # 查找该文件夹下所有的 TXT 文件
# txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
#
# # 打开输出文件进行写入
# with open(output_file_path, 'w', encoding='utf-8') as output_file:
#     for file_path in txt_files:
#         # 打开每个 TXT 文件并读取内容
#         with open(file_path, 'r', encoding='utf-8') as input_file:
#             content = input_file.read()
#             output_file.write(content + '\n')  # 写入内容并添加换行
#
# print(f"所有 TXT 文件已成功合并到 {output_file_path}。")

# import os
# import pandas as pd
#
# # 1. 定义文件夹路径
# folder_path = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/pre-train'  # 替换为你的文件夹路径
# output_file = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/pre-train/pre-train_data.tsv'  # 输出合并后的文件名
#
# # 2. 创建一个空的 DataFrame
# merged_data = pd.DataFrame()
#
# # 3. 遍历文件夹中的所有 TSV 文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.tsv'):
#         file_path = os.path.join(folder_path, filename)
#         # 读取 TSV 文件并追加到 merged_data DataFrame中
#         data = pd.read_csv(file_path, sep='\t')
#         merged_data = pd.concat([merged_data, data], ignore_index=True)
#
# # 4. 保存合并后的 DataFrame 为新的 TSV 文件
# merged_data.to_csv(output_file, sep='\t', index=False)
#
# print(f'Merged {len(os.listdir(folder_path))} TSV files into {output_file}')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# 数据准备
models = ['MML-IDS', 'CIDS-Net', 'ML-HIDS', 'IMD-IDS', 'ET-BERT+NeuralLog', 'TLU-BERT']
f1_scores_100 = [84.79, 84.22, 73.56, 95.75, 97.56, 99.65]  # 標記樣本為 100%
f1_scores_40 = [55.35, 59.74, 58.07, 82.24, 98.45, 99.65]    # 標記樣本為 40%
f1_scores_20 = [36.73, 51.10, 45.98, 69.81, 97.93, 99.31]    # 標記樣本為 20%
f1_scores_10 = [17.86, 32.29, 40.15, 52.67, 97.18, 98.78]    # 標记样本为 10%

# 设置横坐标位置
x = range(len(models))

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(x, f1_scores_100, marker='o', label='100%', color='blue', markersize=8)
plt.plot(x, f1_scores_40, marker='s', label='40%', color='orange', markersize=8)
plt.plot(x, f1_scores_20, marker='^', label='20%', color='green', markersize=8)
plt.plot(x, f1_scores_10, marker='d', label='10%', color='red', markersize=8)

# 添加标签和标题
plt.ylabel('Macro F1 Score', fontsize = 16)
plt.xticks(x, models, fontsize = 12)  # 设置模型名称为 x 轴刻度
# 设置 y 轴为指定的刻度
plt.yticks([10, 25, 40, 55, 70, 85, 100], fontsize = 12)  # 设置 y 轴刻度
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())  # 设置 y 轴格式为百分比

plt.legend(fontsize = 14)
plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

# 显示图形
plt.show()

