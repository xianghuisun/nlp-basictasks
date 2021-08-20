from typing import Text, Union, List
import numpy as np
import random,sys,os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from log import logging
logger=logging.getLogger(__name__)

from .base_reader import convert_to_tensor

SPECIAL_TOKEN_NUMS=2#[CLS] and [SEP]

class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid = '', text = None,  label = 0):
        """
        单句子分类
        text是string类型的句子
        """
        self.guid = guid
        self.text = text
        self.label = label


    def __str__(self):
        return "<InputExample> label: {}, text: {}".format(str(self.label), self.text)

class InputFeatures:
    def __init__(self, input_ids,token_type_ids,attention_mask):
        '''
        单句子分类，token_type_ids应该是全部值为0的list
        '''
        self.input_ids=input_ids
        self.token_type_ids=token_type_ids
        self.attention_mask=attention_mask
    def __str__(self) -> str:
        return "<InputFeature> input_ids: {}\n token_type_ids: {}\n attention_mask: {}\n".format(' '.join([str(x) for x in self.input_ids]),
                                                                                            ' '.join([str(x) for x in self.token_type_ids]),
                                                                                            ' '.join([str(x) for x in self.attention_mask]))


def convert_examples_to_features(examples, tokenizer, max_seq_len):
    if isinstance(examples[0],str):
        #说明此时examples中的每一个元素是一个句子，还没有转为InputExample
        examples=[InputExample(text=text) for text in examples]
        
    features = []
    labels=[]
    assert type(examples[0].text)==str#单句子分类
    max_len_this_batch=0
    for example in examples:
        length=len(tokenizer.tokenize(example.text))
        length+=SPECIAL_TOKEN_NUMS
        if length>max_len_this_batch:
            max_len_this_batch=length

    max_seq_len=min(max_seq_len,max_len_this_batch)
    for example_index, example in enumerate(examples):
        tokens = []
        token_type_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)
        
        tokens_text_a = tokenizer.tokenize(example.text)
        tokens += tokens_text_a
        token_type_ids += [0] * len(tokens_text_a)
        tokens.append('[SEP]')
        token_type_ids.append(0)

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

def getExamplesFromFile(file_path,label2id,mode="train"):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    examples=[]
    for line in lines:
        line_split = line.strip().split('\t')
        sentence,label = line_split
        assert type(sentence)==str

        examples.append(InputExample(text=sentence,label=label2id[label]))

    logger.info("*****************************Logging some {} examples*****************************".format(mode))
    logger.info("Total {} nums is : {}".format(mode,len(examples)))
    for _ in range(5):
        i=random.randint(0,len(examples)-1)
        logger.info('\n'+examples[i].text+"\t"+str(examples[i].label))
    return examples

def getExamplesFromData(sentences,labels,label2id,mode="train",return_max_len=False):
    examples=[]
    max_seq_len=0
    for sentence,label in zip(sentences,labels):
        if len(sentence)>max_seq_len:
            max_seq_len=len(sentence)
        examples.append(InputExample(text=sentence,label=label2id[str(label)]))

    logger.info("*****************************Logging some {} examples*****************************".format(mode))
    logger.info("Total {} nums is : {}".format(mode,len(examples)))
    for _ in range(5):
        i=random.randint(0,len(examples)-1)
        logger.info('\n'+examples[i].text+"\t"+str(examples[i].label))

    if return_max_len:
        return examples,max_seq_len
    else:
        return examples

