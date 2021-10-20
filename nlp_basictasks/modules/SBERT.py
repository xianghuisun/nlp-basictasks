import torch
from torch import Tensor
from torch import nn
from torch import functional as F
from typing import Union, Tuple, List, Iterable, Dict
import os,sys
import json
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from log import logging
logger=logging.getLogger(__name__)

class SBERT(nn.Module):
    def __init__(self,bert_model_path,device='cpu') -> None:
        super(SBERT,self).__init__()
        self.sbert=SentenceTransformer(bert_model_path,device=device)
        logger.info("Using devide : {}".format(device))

    def forward(self,features):
        return self.sbert(features)#get token_embeddings abd cls_token_embeddings and sentence_embeddings
    
    def save(self,output_path):
        self.sbert.save(output_path)
    
    @staticmethod
    def load(input_path):
        return SBERT(input_path)