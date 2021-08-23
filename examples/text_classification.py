# 数据集类型是微博情感分类 来源https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb

import sys,os
import pandas as pd
import random
import numpy as np
from torch.utils.data import DataLoader
from nlp_basictasks.tasks import cls
from nlp_basictasks.evaluation import clsEvaluator
from nlp_basictasks.readers.cls import getExamplesFromData

device='cuda'
data_path='/data/nfs14/nfs/aisearch/asr/xhsun/datasets/weibo_senti_100k.csv'
model_path='/data/nfs14/nfs/aisearch/asr/xhsun/CommonModel/chinese-roberta-wwm/'
output_path="/data/nfs14/nfs/aisearch/asr/xhsun/tmp_model/"#output_path是指保存模型的路径
tensorboard_logdir='/data/nfs14/nfs/aisearch/asr/xhsun/tmp_model/log'
batch_size=128
optimizer_params={'lr':5e-5}
epochs=10

pd_all = pd.read_csv(data_path)

print('评论数目（总体）：%d' % pd_all.shape[0])
print('评论数目（正向）：%d' % pd_all[pd_all.label==1].shape[0])
print('评论数目（负向）：%d' % pd_all[pd_all.label==0].shape[0])

print(pd_all.sample(20))
print(len(pd_all))
random_idx=np.random.permutation(len(pd_all))

sentences=pd_all['review'].values[random_idx].tolist()
labels=pd_all['label'].values[random_idx].tolist()
print("数据集的总量 : ",len(sentences),len(labels))

label2id={'0':0,'1':1}
dev_ratio=0.2
dev_nums=int(len(sentences)*dev_ratio)
train_nums=len(sentences)-dev_nums
print("验证集的数量 : ",dev_nums)
train_sentences=sentences[:train_nums]
train_labels=labels[:train_nums]
dev_sentences=sentences[-dev_nums:]
dev_labels=labels[-dev_nums:]
train_examples,max_seq_len=getExamplesFromData(sentences=train_sentences,labels=train_labels,label2id=label2id,mode='train',return_max_len=True)
dev_examples=getExamplesFromData(sentences=dev_sentences,labels=dev_labels,label2id=label2id,mode='dev')

max_seq_len=min(512,max_seq_len)
print('数据集中最长的句子长度 : ',max_seq_len)
cls_model=cls(model_path=model_path,
                label2id=label2id,
                max_seq_length=max_seq_len,
                device=device,
                tensorboard_logdir=tensorboard_logdir)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

evaluator=clsEvaluator(sentences=dev_sentences,label_ids=dev_labels,write_csv=False,label2id=label2id)

cls_model.fit(is_pairs=False,
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            output_path=output_path,
            epochs=epochs,
            optimizer_params=optimizer_params)

