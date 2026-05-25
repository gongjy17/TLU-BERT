import pandas as pd
import csv
import math
from drain3.drain import Drain, LogCluster

# # 定义文件路径
# traffic_file_path = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0223/traffic.tsv'  # traffic.tsv的实际路径
# parsedlog_file_path = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0223/0223-172.31.69.28_parsed.csv'
#
# # 读取 traffic.tsv 文件
# traffic_df = pd.read_csv(traffic_file_path, sep='\t')
#
# # 读取 0214-172.31.69.25_parsedlog.csv 文件
# log_df = pd.read_csv(parsedlog_file_path)
#
#
# # Step 2: 定义一个函数，根据 start_time 和 end_time 筛选 log 数据
# def get_filtered_text_b(start_time, end_time):
#     # 将 timestamp 列转换为日期时间格式（如果尚未转换）
#     log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
#
#     # 筛选日志数据
#     filtered_logs = log_df[(log_df['timestamp'] >= pd.to_datetime(start_time)) &
#                            (log_df['timestamp'] <= pd.to_datetime(end_time))]
#
#     if not filtered_logs.empty:
#         # 拼接符合条件的 text_b 列
#         return ' '.join(filtered_logs['text_b'].astype(str))
#     else:
#         return 'There is no corresponding log.'
#
#
# # Step 3: 对 traffic_df 的每一行应用函数，并将结果添加到新的列 text_b
# traffic_df['text_b'] = traffic_df.apply(lambda row: get_filtered_text_b(row['start_time'], row['end_time']), axis=1)
#
# # Step 4: 保存结果到更新后的 traffic.tsv 文件
# traffic_df.to_csv('/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0223/combined_0223.tsv', sep='\t', index=False)
#
# with open('/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0221/combined1.tsv', 'r', encoding='utf-8') as tsv_file:
#     # 创建一个csv.reader对象，指定分隔符为制表符
#     tsv_reader = csv.reader(tsv_file, delimiter='\t')
#
#     # 打开TXT文件用于写入
#     with open('/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0221/combined1.txt', 'w', encoding='utf-8') as txt_file:
#         for row in tsv_reader:
#             # 检查是否有足够的列
#             if len(row) >= 3:
#                 # 读取第三列内容
#                 third_column = row[2]
#                 # 写入第三列内容到TXT文件
#                 txt_file.write(third_column + '\n')

traffic_tsv_file = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CREME/traffic_multi.tsv'
log_csv_file = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CREME/label_syslog_combined.csv'
log_txt_file = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CREME/syslog.log'
traffic_tsv = pd.read_csv(traffic_tsv_file, sep='\t')
log_csv = pd.read_csv(log_csv_file)
with open(log_txt_file, "r", encoding="utf-8") as file:
    log_txt = file.readlines()
timestamps = traffic_tsv["start_time"].tolist()
timestamps = [math.floor(ts) for ts in timestamps]
logs = []

log_messages = log_csv["Content"].tolist()
log_templates = []
model = Drain()
for message in log_messages:
    cluster, change_type = model.add_log_message(message)
    log_templates.append(cluster.get_template())
log_csv["Template"] = log_templates

for ts in timestamps:
    # 筛选出 Timestamp 列等于当前时间戳的行
    matching_rows = log_csv[log_csv["Timestamp"] == ts]
    matching_rows = matching_rows.drop_duplicates(subset="Template")
    if not matching_rows.empty:
        # 提取 Content 列的字符串并拼接成一个长字符串，用空格隔开
        log = " ".join(matching_rows["Content"].astype(str))
        log = log.replace('"', '').replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ').replace('<', ' ').replace('>', ' ')
        log = log.lower()
    else:
        log = "There is no corresponding log"
    # 将拼接结果添加到 logs 列表中
    logs.append(log)

traffic_tsv = traffic_tsv.drop(columns=["StartTime", "EndTime", "Output"])
traffic_tsv = traffic_tsv.rename(columns={"FeatureString": "Traffic"})
traffic_tsv["Log"] = logs
traffic_tsv.to_csv("/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CREME/sample_CREME.tsv", sep="\t", index=False)

