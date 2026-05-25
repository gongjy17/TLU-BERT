import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, DataCollatorForLanguageModeling, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import pickle
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('/home/gongjingyuan/PycharmProjects/MD-BERT/vocab-process/vocab')
with open("/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/pre-train/pre-train_data.pkl", "rb") as f:
    data = pickle.load(f)

class CSECICIDS2018_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence_a, sentence_b, label = self.data[idx]

        # 使用分词器编码句子对
        encoded = self.tokenizer(
            sentence_a,
            sentence_b,
            max_length=self.max_length,
            padding="max_length",
            truncation="only_second",
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded["token_type_ids"].squeeze(0),
            "next_sentence_label": torch.tensor(label, dtype=torch.long)
        }

dataset = CSECICIDS2018_Dataset(data, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
data_loader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)
batch = next(iter(data_loader))

# 定义模型配置
config = BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    type_vocab_size=2,
)

model = BertForMaskedLM(config)
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(data_loader) * 10
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.train()
epochs = 10
for epoch in range(epochs):
    loop = tqdm(data_loader, leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

    model.save_pretrained("/home/gongjingyuan/PycharmProjects/MD-BERT/pre-train/wo-STP")
    tokenizer.save_pretrained("/home/gongjingyuan/PycharmProjects/MD-BERT/pre-train/wo-STP")