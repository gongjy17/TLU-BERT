import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 读取数据
data = pd.read_csv("/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CREME/sample_CREME.tsv", sep="\t")
# 拼接 Traffic 和 Log 列
data['text'] = '[CLS] ' + data['Traffic'] + ' [SEP] ' + data['Log']

# 划分数据集
train_val_data, test_data = train_test_split(data, test_size=0.2, stratify=data['Label'], random_state=38)
train_data, val_data = train_test_split(train_val_data, test_size=0.125, stratify=train_val_data['Label'], random_state=42)  # 验证集占比7:1:2
# train_data, _ = train_test_split(train_data, test_size=0.9, stratify=train_data['Label'], random_state=42)

# 转换为 Dataset 格式
train_dataset = Dataset.from_pandas(train_data[['text', 'Label']])
val_dataset = Dataset.from_pandas(val_data[['text', 'Label']])
test_dataset = Dataset.from_pandas(test_data[['text', 'Label']])

# 构建 DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("/home/gongjingyuan/PycharmProjects/MD-BERT/pre-train/wo-MLM")

# 定义 tokenization 函数
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length = 512)

# 对数据集进行 Tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.rename_column("Label", "labels")
tokenized_datasets.set_format("torch")

# # 加载预训练模型
# model = AutoModelForSequenceClassification.from_pretrained("/home/gongjingyuan/PycharmProjects/MD-BERT/pre-train", num_labels=data['Label'].nunique())
#
# training_args = TrainingArguments(
#     output_dir="/home/gongjingyuan/PycharmProjects/MD-BERT/fine-tuning/nFold",          # 保存结果的路径
#     learning_rate=2e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     save_strategy="epoch",
#     evaluation_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
# )
#
# # 定义评价指标
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = torch.argmax(torch.tensor(logits), dim=-1)
#     return classification_report(labels, predictions, output_dict=True)
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# trainer.train()

# 加载微调好的模型
model_path = "/home/gongjingyuan/PycharmProjects/MD-BERT/fine-tuning/nFold/checkpoint-2540"  # 替换为模型本地路径
model = AutoModelForSequenceClassification.from_pretrained(model_path)

trainer = Trainer(
    model=model,
    eval_dataset=tokenized_datasets["test"],  # 测试集
    tokenizer=tokenizer
)
predictions = trainer.predict(tokenized_datasets["test"])
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1).numpy()
true_labels = tokenized_datasets["test"]['labels'].numpy()
# 计算各项指标
f1 = f1_score(true_labels, predicted_labels)  # 加权 F1 分数
recall = recall_score(true_labels, predicted_labels)  # 加权 Recall
precision = precision_score(true_labels, predicted_labels)  # 加权 Precision
accuracy = accuracy_score(true_labels, predicted_labels)  # 准确率
print(f"F1 Score: {f1:}")
print(f"Recall: {recall:}")
print(f"Precision: {precision:}")
print(f"Accuracy: {accuracy:}")
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# # 计算误报（FP）和漏报（FN）的索引
# false_positives_indices = np.where((predicted_labels == 1) & (true_labels == 0))[0]  # 预测为正类，但真实为负类
# false_negatives_indices = np.where((predicted_labels == 0) & (true_labels == 1))[0]  # 预测为负类，但真实为正类
#
# # 输出误报（FP）的样本
# print("False Positives (FP):")
# for index in false_positives_indices:
#     print(f"Index: {index}, Prediction: {predicted_labels[index]}, True Label: {true_labels[index]}")
#     # 使用 .select 方法获取相关的输入数据
#     input_data = tokenized_datasets["test"].select([index])
#     print("Input Data:", input_data['text'])
#
# # 输出漏报（FN）的样本
# print("\nFalse Negatives (FN):")
# for index in false_negatives_indices:
#     print(f"Index: {index}, Prediction: {predicted_labels[index]}, True Label: {true_labels[index]}")
#     # 使用 .select 方法获取相关的输入数据
#     input_data = tokenized_datasets["test"].select([index])
#     print("Input Data:", input_data['text'])
