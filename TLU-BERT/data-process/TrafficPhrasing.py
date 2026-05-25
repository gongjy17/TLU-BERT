import os
import scapy.all as scapy
import pandas as pd
import numpy as np
import binascii
import csv

def cut(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    try:
        remanent_count = len(result[0])%4
    except Exception as e:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result


def bigram_generation(packet_datagram, packet_len=64, flag=True):
    result = ''
    generated_datagram = cut(packet_datagram, 1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '

    return result


def get_feature_packet(label_pcap):
    packets = scapy.rdpcap(label_pcap)
    packet_data_string = ''
    packet_num = 0

    for packet in packets:
        packet_data = packet.copy()
        data = binascii.hexlify(bytes(packet_data))
        packet_string = data.decode()

        # Only consider packet_string with length >=76
        if len(packet_string) >= 76:
            new_packet_string = packet_string[76:]
            packet_data_string += bigram_generation(new_packet_string)
            packet_num = packet_num + 1
        if packet_num >= 5:
            break

    return packet_data_string


def process_pcap_files(directory='/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0221/splitcap0/0', output_file='/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/0221/traffic2.tsv'):
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        # Write the header row
        writer.writerow(["start_time", "end_time", 'text_a'])

        # Process each pcap file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.pcap'):
                filepath = os.path.join(directory, filename)

                # Read packets from PCAP file
                packets = scapy.rdpcap(filepath)

                # Determine start and end time from packets
                start_time = packets[0].time
                end_time = packets[-1].time

                # Use get_feature_packet function
                features = get_feature_packet(filepath)

                features = features.rstrip()

                # Write to the TSV file
                writer.writerow([start_time, end_time, features])


if __name__ == "__main__":
    process_pcap_files()