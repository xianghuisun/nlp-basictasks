import numpy as np
import os,json,sys
from typing import Dict, Sequence, Type, Callable, List, Optional
import torch
from torch import nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from tensorboardX import SummaryWriter

from modules.transformers import BertModel,BertConfig,BertTokenizer
from modules.MLP import MLP
from log import logging
logger=logging.getLogger(__name__)

'''
将NER数据处理成Squad形式，每一条example形如:
{
    'context':'在那最为艰难的日子里，我追思父亲历尽沧桑的人生之路，他从一个辛亥革命时。。。',
    'qas':[{'question':'组织包括公司,政府党派,学校,政府,新闻机构','answer':'xxx'},
            {'question':'按照地理位置划分的国家,城市,乡镇,大洲','answer':'xxx'},
            {'question':'人名和虚构的人物形象','answer':'xxx'}]
}
'''

class Mrc(nn.Module):
    '''
    ClsHead不区分是单句子还是双句子，因为处理逻辑是一样的
    '''
    def __init__(self, model_path,
                state_dict=None,
                is_finetune=False):
        super().__init__()
        if is_finetune==False:
            logger.info("Loading model from {}, which is from huggingface model".format(model_path))
            self.load_huggingface_model(bert_model_path=model_path)
        else:
            self.load_finetuned_model(model_path=model_path)
            logger.info("Loading model from {}, which has been finetuned.".format(model_path))

    def load_huggingface_model(self,bert_model_path):
        self.config=BertConfig.from_pretrained(bert_model_path)
        self.bert=BertModel.from_pretrained(bert_model_path)
        self.tokenizer=BertTokenizer.from_pretrained(bert_model_path)
        self.start_outputs_layer=MLP(in_features=self.config.hidden_size,out_features=1)
        self.end_outputs_layer=MLP(in_features=self.config.hidden_size,out_features=1)
        #预测每一个位置是起始位置或者终止位置的概率。此时的head_layer相当于两个二分类器，与传统的MRC输出层不同
        #传统的MRC输出层是在所有的token上做softmax，而此时为了能够预测多个span，所以不能在所有的token上做softmax


    def load_finetuned_model(self,model_path):
        bert_save_path=os.path.join(model_path,"BERT")#save的时候将BERT保存在model_path下的BERT文件夹中
        self.config=BertConfig.from_pretrained(bert_save_path)
        self.bert=BertModel.from_pretrained(bert_save_path)
        self.tokenizer=BertTokenizer.from_pretrained(bert_save_path)

        start_save_path=os.path.join(model_path,'MLP_start')
        end_save_path=os.path.join(model_path,"MLP_end")
        self.start_outputs_layer=MLP.load(input_path=start_save_path)
        self.end_outputs_layer=MLP.load(input_path=end_save_path)

    def save(self,output_path):
        bert_save_path=os.path.join(output_path,"BERT")
        self.bert.save_pretrained(bert_save_path,save_config=False)#下面已经save，不用save两次，虽然没什么影响
        self.config.save_pretrained(bert_save_path)
        self.tokenizer.save_pretrained(bert_save_path)

        start_save_path=os.path.join(output_path,'MLP_start')
        end_save_path=os.path.join(output_path,"MLP_end")
        self.start_outputs_layer.save(start_save_path)
        self.end_outputs_layer.save(end_save_path)
    
    def forward(self,input_ids,attention_mask=None,token_type_ids=None,output_all_encoded_layers=False,embedding_for_adv=None):
        '''
        input_ids.size()==attention_mask.size()==token_type_ids.size()==position_ids.size()==
        (batch_size,seq_length)
        start_positions.size()==(batch_size,)==end_positions.size()
        如果不输出所有中间层的hidden_states，那么sequence_outputs就是tensor而不是list
        否则sequence_outputs就是长度为层数的list
        '''
        (sequence_outputs,pooled_output)=self.bert(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask,
                                                   output_all_encoded_layers=output_all_encoded_layers,
                                                   embedding_for_adv=embedding_for_adv)
        #要注意到sequence_output[0]与pooled_output的区别在于pooled_output是经过一层tanh的
        start_logits=self.start_outputs_layer(sequence_outputs)#(batch_size,seq_len,1)
        end_logits=self.start_outputs_layer(sequence_outputs)#(batch_size,seq_len,1)
        #此时start_logits的每一个元素代表当前位置可以作为开始位置的prob，注意到每一个元素都是独立的，而不是
        #所有的token做softmax，end_logits同理。这样便可以解决传统的MRC处理方式只能预测一个span的问题
        start_logits=start_logits.squeeze(-1)
        end_logits=end_logits.squeeze(-1)
        # 2 (batch_size,seq_len)
        # if start_positions is not None and end_logits is not None:
        #     assert start_positions.dim()==end_positions.dim()<start_logits.dim()
        #     ignore_index=start_logits.size(1)#对于长度超出给定目标的长度，忽略
        #     start_positions.clamp_(0,ignore_index)
        #     end_positions.clamp_(0,ignore_index)
        #     loss_fct=nn.CrossEntropyLoss(ignore_index=ignore_index)
        #     start_loss=loss_fct(start_logits,start_positions)
        #     end_loss=loss_fct(end_logits,end_positions)
        return start_logits,end_logits#(batch_size,seq_len)


class MrcDoNer():
    def __init__(self,model_path,
                max_seq_length:int = 128,
                device:str = None,
                state_dict=None,
                is_finetune=False,
                tensorboard_logdir = None,
                do_FGV=False):
        self.model=Mrc(model_path=model_path,state_dict=state_dict,is_finetune=is_finetune)
        if tensorboard_logdir!=None:
            os.makedirs(tensorboard_logdir,exist_ok=True)
            self.tensorboard_writer=SummaryWriter(tensorboard_logdir)
        else:
            self.tensorboard_writer=None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Use pytorch device: {}".format(device))
        self._target_device = torch.device(device)
        self.model.to(self._target_device)

