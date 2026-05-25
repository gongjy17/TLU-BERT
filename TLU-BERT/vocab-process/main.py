from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import BertTokenizer

# Step 1: 初始化分词器和WordPiece模型
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# 设置分词器预处理器
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 设置训练器
trainer = trainers.WordPieceTrainer(
    vocab_size=100000,  # 标准BERT词汇大小
    min_frequency=2,
    limit_alphabet=1000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
)

# 训练分词器
dataset_path = "/home/gongjingyuan/PycharmProjects/MD-BERT/datasets/CSE-CIC-IDS2018/vocab/vocab_dataset.txt"  # 数据集路径
tokenizer.train(files=[dataset_path], trainer=trainer)

# Step 2: 保存词汇表
output_directory = "/home/gongjingyuan/PycharmProjects/MD-BERT/vocab-process/vocab"
vocab_file_path = f"{output_directory}/vocab.txt"
tokenizer.model.save(output_directory)

# Step 3: 使用保存的词汇表测试分词
bert_tokenizer = BertTokenizer.from_pretrained(output_directory)

# 测试样例
test_sentence = "0d40 4007 072d 2d34 34da da34 34c2 c280 8018 1800 00d3 d319 19fa fa00 0000 0001 0101 0108 080a 0acd cda7 a726 2654 5400 000e 0e22 22af af47 4745 4554 5420 202f 2f3f 3f76 7658 5830 304d 4d4d 4d4f 4f52 5254 5458 583d 3d48 484e 4e48 4863 6331 3143 4350 5074 7426 2662 6245 4533 3370 7050 5071 713d 3d49 4957 574c 4c6a 6a78 0696 960d 0d35 352e 2e2b 2b27 2706 0680 8018 1800 00d3 d32f 2ff7 f700 0000 0001 0101 0108 080a 0acd cdbc bc03 0317 1700 0013 1359 59da da47 4745 4554 5420 202f 2f3f 3f76 7664 6437 373d 3d69 6932 3275 7563 6335 3575 754f 4f66 666a 6a6a 6a42 4242 4276 7637 3750 5058 5837 3753 5330 3020 2048 4854 5454 5450 502f 2f31 312e  root  cmd      -x /usr/lib/php/sessionclean   && /usr/lib/php/sessionclean  dhcprequest of 172.31.69.25 on eth0 to 172.31.69.1 port 67  xid=0x6db0b5ec  dhcpack of 172.31.69.25 from 172.31.69.1 bound to 172.31.69.25 -- renewal in 1699 seconds."
tokenized_output = bert_tokenizer.tokenize(test_sentence)

# 打印结果
print("Tokenized output:", tokenized_output)
print("Vocabulary saved at:", vocab_file_path)
