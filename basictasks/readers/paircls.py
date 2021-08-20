from typing import Union, List
import numpy as np
import random,sys,os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from log import logging
logger=logging.getLogger(__name__)

from .base_reader import convert_to_tensor

SPECIAL_TOKEN_NUMS=3#[CLS] [SEP] and [SEP]

class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', text_list: List[str] = None,  label = 0):
        """
        双句子分类
        text_list是一个有两个str的list
        """
        self.guid = guid
        self.text_list = text_list
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, text pairs : {}".format(str(self.label), "; ".join(self.text_list))

class InputFeatures:
    def __init__(self, input_ids,token_type_ids,attention_mask):
        self.input_ids=input_ids
        self.token_type_ids=token_type_ids
        self.attention_mask=attention_mask
    def __str__(self) -> str:
        return "<InputFeature> input_ids: {}\n token_type_ids: {}\n attention_mask: {}\n".format(' '.join([str(x) for x in self.input_ids]),
                                                                                            ' '.join([str(x) for x in self.token_type_ids]),
                                                                                            ' '.join([str(x) for x in self.attention_mask]))


def convert_examples_to_features(examples, tokenizer, max_seq_len):
    if isinstance(examples[0],list):
        assert type(examples[0][0])==type(examples[0][1])==str
        examples=[InputExample(text_list=text_list) for text_list in examples]
        #传进来的每一个值是list不是Example类型
        
    features = []
    labels=[]
    assert len(examples[0].text_list)==2
    max_len_this_batch=0
    for example in examples:
        length=len(tokenizer.tokenize(example.text_list[0]))
        length+=len(tokenizer.tokenize(example.text_list[1]))
        length+=SPECIAL_TOKEN_NUMS
        if length>max_len_this_batch:
            max_len_this_batch=length

    max_seq_len=min(max_seq_len,max_len_this_batch)

    for example_index, example in enumerate(examples):
        tokens = []
        token_type_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)
        
        tokens_text_a = tokenizer.tokenize(example.text_list[0])
        tokens += tokens_text_a
        token_type_ids += [0] * len(tokens_text_a)
        tokens.append('[SEP]')
        token_type_ids.append(0)
        
        tokens_text_b = tokenizer.tokenize(example.text_list[1])
        tokens += tokens_text_b
        token_type_ids += [1] * len(tokens_text_b)
        tokens.append('[SEP]')
        token_type_ids.append(1)

        if len(tokens) >= max_seq_len:
            tokens = tokens[:max_seq_len]
            token_type_ids = token_type_ids[:max_seq_len]

        seq_len = len(tokens)
        pad_len = max_seq_len - seq_len
        # print(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids += [0] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
        token_type_ids += [0] * pad_len
        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == max_seq_len
        labels.append(example.label)

        features.append(InputFeatures(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask))
        # if example_index < 5:
        #     logging.info("*** Example ***")
        #     logging.info("example_index: %s" % (example_index))
        #     logging.info("tokens: %s" % " ".join(tokens))
        #     logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logging.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logging.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     logging.info("label_ids : %s" % " ".join([str(example.label)]))
    
    features=convert_to_tensor(features=features)
    labels=torch.LongTensor(labels)
    return features,labels

def getExamples(file_path,label2id,split_token='\t',sentence1_idx=0,sentence2_idx=1,label_idx=2,mode="train",filter_heads=False):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    train_data=[]
    if filter_heads:
        logger.info("Heads like : {}".format(lines[0]))
        lines=lines[1:]
    for line in lines:
        line_split = line.strip().split(split_token)
        assert len(line_split)==3
        sentence1,sentence2,label = line_split[sentence1_idx],line_split[sentence2_idx],line_split[label_idx]
        train_data.append(InputExample(text_list=[sentence1,sentence2],label=label2id[str(label)]))
    logger.info("*****************************Logging some {} examples*****************************".format(mode))
    logger.info("Total {} nums is : {}".format(mode,len(train_data)))
    for _ in range(5):
        i=random.randint(0,len(train_data)-1)
        logger.info("\t".join(train_data[i].text_list)+"\t"+str(train_data[i].label))
    return train_data

def getTripletExamples(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    train_data=[]
    for line in lines:
        line_split = line.strip().split('\t')
        anchor,pos,neg = line_split
        train_data.append(InputExample(texts=[anchor,pos,neg],label=0))
    return train_data