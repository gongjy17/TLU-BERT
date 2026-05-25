import torch
import shap
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载微调后的模型和分词器
model_name = "/home/gongjingyuan/PycharmProjects/MD-BERT/fine-tuning/HDU-BERT"  # 替换为微调后的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)

# 将模型加载到 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 切换到推理模式

# 输入样本
text = "[CLS] fffc fc01 01ff fffb fb1f 1fff fffa fa1f 1f00 0050 5000 0018 18ff fff0 f0ff fffd fd05 05ff fffc fc21 [SEP] in telnetd connect from 192.168.1.103"  # 替换为实际输入
inputs = tokenizer(text, return_tensors="pt", padding="max_length", add_special_tokens=False, truncation=True, max_length=512)

# 将输入数据加载到设备
inputs = {key: value.to(device) for key, value in inputs.items()}

# 模型推理
with torch.no_grad():
    outputs = model(**inputs)  # 获取模型输出
    logits = outputs.logits  # 提取 logits (未归一化的分数)x 
    attentions = outputs.attentions  # 提取注意力矩阵

# 获取第 12 层的所有头的自注意力矩阵
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
layer = 11  # 注意第 12 层的索引是 11
attn_data = attentions[layer].squeeze(0).detach().cpu().numpy()
# 对所有头的注意力矩阵取平均
average_attention = np.mean(attn_data, axis=0)  # 平均所有头 -> (seq_len, seq_len)

# 转换为 DataFrame，保留 token 作为行和列标签
average_attention_df = pd.DataFrame(average_attention, index=tokens, columns=tokens)

# 计算预测结果
probs = torch.nn.functional.softmax(logits, dim=-1)  # 转换为概率分布
predicted_class = torch.argmax(probs, dim=-1).item()  # 获取预测类别（0 或 1）

# 输出结果
print(f"Input Text: {text}")
print(f"Logits: {logits}")
print(f"Probabilities: {probs}")
print(f"Predicted Class: {predicted_class}")

# 保存第 12 层的平均注意力矩阵为 CSV 文件
output_file = "/home/gongjingyuan/PycharmProjects/MD-BERT/fine-tuning/anyalist/mirai.csv"  # 文件名
average_attention_df.to_csv(output_file, encoding="utf-8")
print(f"Saved average attention matrix to {output_file}")



