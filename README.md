# TLU-BERT：基于流量—日志统一表征的多数据源入侵检测方法
这是论文《TLU-BERT：基于流量 — 日志统一表征的多数据源入侵检测方法》的源代码仓库。通过融合网络流量数据与系统日志数据的特征，构建统一的表征空间，解决传统入侵检测方法中单一数据源信息片面、多源数据特征异构难以融合的问题，最终实现更高精度的网络入侵行为检测。
# 基础环境
Python 3.8+
CUDA 11.3+（可选，GPU 加速训练）
PyTorch 1.10+ / TensorFlow 2.8+（二选一，根据模型实现框架）
# 依赖安装
pandas>=1.5.3
numpy>=1.23.5
scikit-learn>=1.2.2
scapy>=2.5.0  # 流量包解析
logparser>=0.1.0  # 日志解析
torch>=1.10.0  # 若使用PyTorch
tensorflow>=2.8.0  # 若使用TensorFlow
matplotlib>=3.6.3  # 结果可视化
seaborn>=0.12.2
tqdm>=4.64.1  # 进度条
scipy>=1.9.3
# 数据集
预训练数据集：
CIC-IDS-2018数据集——https://www.unb.ca/cic/datasets/ids-2018.html
微调数据集：
https://github.com/buihuukhoi/CREME
# 代码结构
├── data/                  # 数据目录
│   ├── raw/               # 原始数据（需自行放入）
│   └── processed/         # 预处理后的数据（脚本生成）
├── data_process/          # 数据预处理模块
│   ├── process_traffic.py # 流量数据解析与特征提取
│   ├── process_log.py     # 日志解析与结构化特征提取
│   └── fuse_features.py   # 流量-日志特征融合（统一表征）
├── model/                 # 模型定义模块
│   ├── base_model.py      # 基础网络（如MLP、CNN、Transformer）
│   ├── fusion_module.py   # 多源特征融合模块（统一表征核心）
│   └── detector.py        # 入侵检测分类器（基于统一表征）
├── train/                 # 模型训练模块
│   ├── train.py           # 训练主脚本
│   └── config.py          # 训练超参数配置（学习率、批次大小、 epochs等）
├── test/                  # 模型测试与评估模块
│   ├── test.py            # 测试主脚本
│   └── evaluate.py        # 评估指标计算（ACC、Precision、Recall、F1、AUC）
├── utils/                 # 工具函数
│   ├── metrics.py         # 评估指标实现
│   ├── logger.py          # 日志记录
│   └── data_loader.py     # 数据加载器
├── results/               # 结果保存目录（训练日志、评估报告、可视化图）
└── requirements.txt       # 依赖清单
