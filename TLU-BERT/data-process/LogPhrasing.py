import csv
import re
import pandas as pd
from datetime import datetime
import pytz
from drain3.drain import Drain, LogCluster

def extract_syslog_entries(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        csv_writer = csv.writer(outfile)

        # 写入 CSV 文件的表头
        csv_writer.writerow(['timestamp', 'text_b'])

        for line in infile:
            # 提取前18个字符
            timestamp = line[:18]

            # 使用 split() 方法提取第三个 ":" 之后的内容
            parts = line.split(":")
            if len(parts) >= 4:  # 至少要有4个部分才能获取第3个":"之后的内容
                text_b = ":".join(parts[3:]).strip()  # 将第三个部分后的所有部分重新合并
                text_b = text_b.lower()
                text_b = text_b.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
            else:
                text_b = ""  # 如果没有找到第三个 ":", 赋值为空字符串
            # 写入 CSV 格式
            csv_writer.writerow([timestamp, text_b])

# 定义转换为 UNIX 时间戳的函数
def convert_to_unix_timestamp(time_str):
    # 定义时间格式
    local_tz = pytz.timezone('Etc/GMT+4')  # UTC-04:00 对应的时区
    naive_datetime = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S")

    local_datetime = local_tz.localize(naive_datetime)

    # 将本地时间转换为 UTC 时间
    utc_datetime = local_datetime.astimezone(pytz.utc)

    # 返回 UNIX 时间戳
    return int(utc_datetime.timestamp())


# 使用示例
input_file = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0221/0221-172.31.69.28.log'  # 输入的日志文件
output_file = '/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0221/0221-172.31.69.28_parsed.csv'  # 输出的CSV文件

# 打开文件进行读取
with open(input_file, 'r', encoding='utf-8') as infile:
    # 读取所有行
    lines = infile.readlines()

# 处理每一行并覆盖原文件内容
with open(input_file, 'w', encoding='utf-8') as outfile:
    for line in lines:
        # 删除前4个字符并添加 "2018/2/"
        modified_line = '2018/2/' + line[4:]  # 从第五个字符开始
        outfile.write(modified_line)

extract_syslog_entries(input_file, output_file)

df = pd.read_csv(output_file)
# 转换第一列的时间戳
df.iloc[:, 0] = df.iloc[:, 0].apply(convert_to_unix_timestamp)

# 将转换后的 DataFrame 覆盖写入同一个 TSV 文件
df.to_csv(output_file, index=False)

# df = pd.read_csv(output_file)
# messages = df['text_b'].tolist()
# model = Drain()
# templates = []
# for message in messages:
#     cluster, change_type = model.add_log_message(message)
#     templates.append(cluster.get_template())
# df['template'] = templates
# df.to_csv(output_file, index=False)

