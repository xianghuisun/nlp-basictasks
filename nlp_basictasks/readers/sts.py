from typing import Union, List
import numpy as np
import random,sys,os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from log import logging
logger=logging.getLogger(__name__)
from .base_reader import convert_to_tensor
SPECIAL_TOKEN_NUMS=2#[CLS] [SEP]

class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', text_list: List[str] = None,  label = 0):
        """
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



def getExamples(file_path,label2id,isCL=False,mode="train",sentence1_idx=0,sentence2_idx=1,label_idx=2,filter_heads=False):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    if filter_heads:
        logger.info("Heads like : {}".format(lines[0]))
        lines=lines[1:]
    train_data=[]
    for line in lines:
        line_split = line.strip().split('\t')
        sentence1 = line_split[sentence1_idx]
        sentence2 = line_split[sentence2_idx]
        label = line_split[label_idx]
        if isCL==True:
            if str(label)=='1':
                train_data.append(InputExample(text_list=[sentence1,sentence2],label=label2id[label]))
        else:
            train_data.append(InputExample(text_list=[sentence1,sentence2],label=label2id[label]))
    logger.info("*****************************Logging some {} examples*****************************".format(mode))
    logger.info("Total {} nums is : {}".format(mode,len(train_data)))
    for _ in range(5):
        i=random.randint(0,len(train_data)-1)
        logger.info("\t".join(train_data[i].text_list)+"\t"+str(train_data[i].label))
    return train_data


def convert_examples_to_features(examples, tokenizer, max_seq_len=None):
    '''
    return like [{'input_ids':tensor,'attention_mask':tensor,'token_type_ids':tensor},{'input_ids':tensor,'attention_mask':tensor,'token_type_ids':tensor}]
    max_seq_len means the max seq_len for single sentence, not pair sentence
    '''
    def get_Feature_from_sentence(sentence,max_len):
        tokens = []
        token_type_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)
        
        tokens_text_a = tokenizer.tokenize(sentence)
        tokens += tokens_text_a
        token_type_ids += [0] * len(tokens_text_a)
        tokens.append('[SEP]')
        token_type_ids.append(0)
        if len(tokens) >= max_len:
            tokens = tokens[:max_len]
            token_type_ids = token_type_ids[:max_len]

        seq_len = len(tokens)
        pad_len = max_len - seq_len
        # print(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids += [0] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
        token_type_ids += [0] * pad_len
        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == max_len
        features=InputFeatures(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return features

    if isinstance(examples[0],list):
        assert type(examples[0][0])==type(examples[0][1])==str
        examples=[InputExample(text_list=text_list) for text_list in examples]
        #传进来的每一个值是list不是Example类型
        
    features_of_a = []
    features_of_b = []
    labels=[]
    assert len(examples[0].text_list)==2
    max_len_of_1=0
    max_len_of_2=0

    for example in examples:
        length_of_1=len(tokenizer.tokenize(example.text_list[0]))
        length_of_2=len(tokenizer.tokenize(example.text_list[1]))
        if length_of_1+SPECIAL_TOKEN_NUMS>max_len_of_1:
            max_len_of_1=length_of_1
        if length_of_2+SPECIAL_TOKEN_NUMS>max_len_of_2:
            max_len_of_2=length_of_2     

    if max_seq_len!=None:
        max_len_of_1=min(max_seq_len,max_len_of_1)
        max_len_of_2=min(max_seq_len,max_len_of_2)

    for example_index, example in enumerate(examples):
        feature_of_a=get_Feature_from_sentence(sentence=example.text_list[0],max_len=max_len_of_1)
        feature_of_b=get_Feature_from_sentence(sentence=example.text_list[1],max_len=max_len_of_2)
        
        features_of_a.append(feature_of_a)
        features_of_b.append(feature_of_b)
        labels.append(example.label)
    
    features_of_a=convert_to_tensor(features=features_of_a)
    features_of_b=convert_to_tensor(features=features_of_b)
    #{"input_ids":batch_size,max_seq_len,...}
    labels=torch.LongTensor(labels)
    return features_of_a,features_of_b,labels


def convert_sentences_to_features(sentences,tokenizer,max_seq_len=None):
    assert type(sentences)==list
    tokenized_sentences=[tokenizer.tokenize(sen) for sen in sentences]
    max_len=max([len(sen)+2 for sen in tokenized_sentences])
    if max_seq_len!=None:
        max_len=min(max_seq_len,max_len)

    all_sen_features=[]
    for tokenized_sen in tokenized_sentences:
        tokens = []
        token_type_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)

        tokens += tokenized_sen
        token_type_ids += [0] * len(tokenized_sen)
        tokens.append('[SEP]')
        token_type_ids.append(0)
        if len(tokens) >= max_len:
            tokens = tokens[:max_len]
            token_type_ids = token_type_ids[:max_len]

        seq_len = len(tokens)
        pad_len = max_len - seq_len
        # print(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids += [0] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
        token_type_ids += [0] * pad_len
        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == max_len
        features=InputFeatures(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        all_sen_features.append(features)
    
    return convert_to_tensor(all_sen_features)#{"input_ids":tensor,...}