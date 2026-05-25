from scapy.all import rdpcap, wrpcap
from scapy.layers.inet import TCP, UDP, IP
import os
from tqdm import tqdm
from collections import defaultdict
import dpkt
import os
import socket

def split_pcap_by_size(input_pcap_path, output_dir, max_packets, file_name_template="split_{index}.pcap"):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_file_index = 0
    current_packet_count = 0
    output_pcap_path = os.path.join(output_dir, file_name_template.format(index=current_file_index))

    # 打开输入PCAP文件
    with open(input_pcap_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)

        # 新文件对应的写入操作
        current_output_pcap = open(output_pcap_path, 'wb')
        writer = dpkt.pcap.Writer(current_output_pcap)

        for timestamp, buf in pcap:
            # 写入包到当前分片
            writer.writepkt(buf, timestamp)
            current_packet_count += 1

            # 如果达到最大包数，关闭当前文件并准备下一个文件
            if current_packet_count >= max_packets:
                current_output_pcap.close()
                current_file_index += 1
                current_packet_count = 0
                output_pcap_path = os.path.join(output_dir, file_name_template.format(index=current_file_index))
                current_output_pcap = open(output_pcap_path, 'wb')
                writer = dpkt.pcap.Writer(current_output_pcap)

        # 关闭最后的输出文件
        current_output_pcap.close()

    print(f"PCAP文件已拆分成 {current_file_index + 1} 个文件。")


def extract_session_key(pkt):
    # 获取源地址、目标地址、源端口、目标端口
    src = pkt['ip'].src
    dst = pkt['ip'].dst
    sport = pkt['tcp'].sport if 'tcp' in pkt else ('udp' in pkt and pkt['udp'].sport)
    dport = pkt['tcp'].dport if 'tcp' in pkt else ('udp' in pkt and pkt['udp'].dport)

    return f"{socket.inet_ntoa(src)}_{sport}_{socket.inet_ntoa(dst)}_{dport}"


def split_pcap_by_session(pcap_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sessions = {}

    # 读取 PCAP 文件
    with open(pcap_file, 'rb') as f:
        pcap_reader = dpkt.pcap.Reader(f)

        for timestamp, buf in pcap_reader:
            try:
                # 解析以太网包
                eth = dpkt.ethernet.Ethernet(buf)
                # 只处理 IP 包
                if isinstance(eth.data, dpkt.ip.IP):
                    ip = eth.data

                    # 对于 TCP 和 UDP 包，提取会话信息
                    if hasattr(ip, 'data'):
                        session_key = extract_session_key(ip.data)

                        if session_key not in sessions:
                            sessions[session_key] = []
                        sessions[session_key].append((timestamp, buf))
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                continue

    # 保存会话到文件
    for session_key, packets in sessions.items():
        output_file = os.path.join(output_dir, f"{session_key}.pcap")

        # 处理重名文件
        count = 1
        while os.path.exists(output_file):
            output_file = os.path.join(output_dir, f"{session_key}_{count}.pcap")
            count += 1

        with open(output_file, 'wb') as out_f:
            pcap_writer = dpkt.pcap.Writer(out_f)
            for timestamp, buf in packets:
                pcap_writer.writepkt(buf, timestamp)

    print(f"保存了 {len(sessions)} 个会话到 {output_dir}")

def process_multiple_pcaps(input_dir, output_dir):
    """
    处理指定目录下的所有 PCAP 文件，按会话拆分并保存到输出目录。

    :param input_dir: 输入的 PCAP 文件所在目录
    :param output_dir: 输出文件保存的目录
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc="Processing PCAP files"):
        if filename.endswith(".pcap"):
            input_file = os.path.join(input_dir, filename)
            split_pcap_by_session(input_file, output_dir)

# 使用示例
input_pcap_file = "/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0221/UCAP172.31.69.28-1.pcap"  # 替换为你的 pcap 文件
input_dir = "/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0221/pcap"
output_dir = "/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0221/splitcap0"  # 替换为你希望保存的目录

# split_pcap_by_size(input_pcap_file, input_dir, 10000000, "0221-172.31.69.28-{index}.pcap")
split_pcap_by_session(input_pcap_file, output_dir)
# process_multiple_pcaps(input_dir, output_dir)
