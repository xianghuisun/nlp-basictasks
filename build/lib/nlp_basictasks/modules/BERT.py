import torch
from torch import Tensor
from torch import nn
from torch import functional as F
from typing import Union, Tuple, List, Iterable, Dict
import os,sys
import json

from transformers import BertModel,BertConfig,BertTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from log import logging
logger=logging.getLogger(__name__)

class BERT():
    def __init__(self,bert_model_path) -> None:
        self.config=BertConfig.from_pretrained(bert_model_path)
        self.bert=BertModel.from_pretrained(bert_model_path)
        self.tokenizer=BertTokenizer.from_pretrained(bert_model_path)

    def __call__(self,input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, embedding_for_adv=None):
        encoded_layers, pooled_output=self.bert(input_ids, 
                                                token_type_ids=token_type_ids, 
                                                attention_mask=attention_mask, 
                                                output_all_encoded_layers=output_all_encoded_layers, 
                                                embedding_for_adv=embedding_for_adv)

        return encoded_layers, pooled_output
    
    def save(self,output_path):
        self.bert.save_pretrained(output_path,save_config=False)#下面已经save，不用save两次，虽然没什么影响
        self.config.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
    
    @staticmethod
    def load(input_path):
        return BERT(input_path)