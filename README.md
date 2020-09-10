# 零基础入门NLP - 新闻文本分类 正式赛第一名方案
nano- 康一帅
## 简介
### 环境
* `Tensorflow` == 1.14.0
* `Keras` == 2.3.1
* `bert4keras` == 0.8.4
### 文件说明
* `EDA`：用于探索性数据分析。
* `data_utils`：用于预训练语料的构建。
* `pretraining`：用于Bert的预训练。
* `train`：用于新闻文本分类模型的训练。
* `pred`：用于新闻文本分类模型的预测。
### 其他
* 数据集：[下载地址](https://pan.baidu.com/s/1t33R14RCO9_-1mBa6D_a9A)（z8vl）
* 预训练语料：[下载地址](https://pan.baidu.com/s/1f2FVD4BGQEgTQWOUXBnPpw)（72ml）
* 预训练模型：[下载地址](https://pan.baidu.com/s/18zCs045LhDIaS_Q_6cWSSw)（32b6）
## 赛题分析
### 赛题背景
通过这道赛题可以引导大家走入自然语言处理的世界，带大家接触NLP的预处理、模型构建和模型训练等知识点。
### 任务目标
要求选手根据新闻文本字符对新闻的类别进行分类，这是一个经典文本分类问题。
### 数据示例
![](https://github.com/kangyishuai/NEWS-TEXT-CLASSIFICATION/blob/master/imgs/0.png)
### 文本长度
* 训练集共200,000条新闻，每条新闻平均907个字符，最短的句子长度为2，最长的句子长度为57921，其中75%以下的数据长度在1131以下。
* 测试集共50,000条新闻，每条新闻平均909个字符，最短句子长度为14，最长句子41861,75%以下的数据长度在1133以下。
* 训练集和测试集就长度来说似乎是同一分布。
![](https://github.com/kangyishuai/NEWS-TEXT-CLASSIFICATION/blob/master/imgs/1.png)
### 标签分布
* 赛题的数据集类别分布存在较为不均匀的情况。在训练集中科技类新闻最多，其次是股票类新闻，最少的新闻是星座新闻。
![](https://github.com/kangyishuai/NEWS-TEXT-CLASSIFICATION/blob/master/imgs/2.png)
## 总体思路
![](https://github.com/kangyishuai/NEWS-TEXT-CLASSIFICATION/blob/master/imgs/3.png)
### 数据划分
* 使用StratifiedKFold交叉验证。StratifiedKFold能够确保抽样后的训练集和验证集的样本分类比例和原原始数据集基本一致。
* 利用全部数据，获得更多信息。
* 降低方差，提高模型性能。
![](https://github.com/kangyishuai/NEWS-TEXT-CLASSIFICATION/blob/master/imgs/4.png)
### 模型设计思路
* 由于文本长度较大，而Bert输入文本长度不能超过512（如果是自己预训练的Bert，长度可以不局限于512），所以需要进行文本截断。
* 文本截断后，输入大小为[batch_size, max_segment, maxlen]，其中batch_size是批大小，max_segment是截断后的最大句子数量，maxlen是每个句子的最大长度。
* 将输入reshape后输入Bert，得到大小为[batch_size*max_segment, maxlen]的句向量。
* 将句向量reshape后输入注意力层。
* 最后接全连接层进行分类。
### 预测
* 预测就是将多个模型预测结果计算算术平均即可。
## 技巧尝试
### 预训练
* 使用训练集和测试集一共25万数据对Bert进行预训练。
### 对抗训练
* 对抗训练属于对抗防御的一种，它构造了一些对抗样本加入到原数据集中，希望增强模型对对抗样本的鲁棒性。
* 详见[对抗训练浅谈：意义、方法和思考（附Keras实现）](https://kexue.fm/archives/7234)。
### 梯度累积优化器
* 由于显存有限，无法使用较大的batch size进行训练，梯度累积优化器可以实现使用小的batch size实现大batch size的效果了——只要你愿意花n倍的时间，可以达到n倍batch size的效果，而不需要增加显存。
* 详见[用时间换取效果：Keras梯度累积优化器](https://kexue.fm/archives/6794)
## 经验总结
* 由于训练集和测试集的分布基本一致，所以在验证集和测试集上的结果也基本一致。
* 使用Bert可以轻松上到0.96+。
* Bert的预训练和微调过程都很消耗时间（硬件条件有限的情况下）。如果时间充裕，预训练更多步，预训练模型的效果应该还能提升。
* 调参和模型融合是很有效的提升手段。