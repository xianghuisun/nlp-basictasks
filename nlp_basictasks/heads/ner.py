import numpy as np
import os,json,sys
from typing import Dict, Sequence, Type, Callable, List, Optional
import torch
from torch import nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from modules.transformers import BertModel,BertConfig,BertTokenizer
from modules.MLP import MLP
from modules.LSTM import LSTMLayer
from log import logging
logger=logging.getLogger(__name__)

'''
BertModel inherite BertForPretrainedModel inherite PreTrainedModel

In PreTrainedModel
@classmethod
def from_pretrained(cls,pretrained_model_name_or_path,*model_args,**kwargs):
    config=kwargs.pop('config',None)
    state_dict=kwargs.pop('state_dict',None)

    #The config will be loaded from pretrained_model_name_or_path/config.json if config not provided
    #instantiate model
    model=cls(config)
    if state_dict is None:
        state_dict=torch.load(pretrained_model_name_or_path/pytorch_model.bin,map_location='cpu')#也就是说模型最开始是加载到cpu上的
    missing_keys,unexpected_keys,error_msgs=[],[],[]

    def load(module: nn.Module, prefix=''):
        module._load_from_state_dict(state_dict,prefix,missing_keys,unexpected_keys,error_msgs)
        for name,child in module._modules.items():
            if child:
                load(child,prefix+name+'.')
    
    load(model,prefix='' if hasattr(model,'bert') else 'bert.')
    #prefix用来解决加载不同模型保存的pytorch_model.bin出现的keys不匹配问题
    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    return model
'''

class CRF(nn.Module):
    def __init__(self, num_tags, batch_first=True) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        #一定要有nn.Parameter，用Parameter修饰的tensor才可以被认为是参数,requires_grad=True
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags,num_tags))
        self.init_matrix()

    def init_matrix(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def get_config_dict(self):
        '''
        一定要有dict，这样才能初始化Model
        '''
        return {'num_tags': self.num_tags, 'batch_first':self.batch_first}

    def save(self,output_path):
        '''
        同时保存dict
        '''
        os.makedirs(output_path,exist_ok=True)
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut)

        torch.save(self.state_dict(),os.path.join(output_path, 'pytorch_model.bin'))
    
    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = CRF(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model

    def compute_numerator(self,emissions, tags, mask):
        '''
        emissions.size()=(max_seq_len.batch_size,num_tags)
        tags.size()=(max_seq_len,batch_size)
        mask.size()=(max_seq_len,batch_size)

        when max_seq_len=4,batch_size=3,num_tags=6
        examples: emissions=tensor([[[ 0.5219,  0.8717,  0.5480, -0.8730, -0.1198, -1.0771],
         [ 0.0299, -1.0110, -1.5722, -0.0844,  3.0301, -0.7060],
         [-0.8239, -1.3775, -0.9329, -1.6733,  0.2417, -0.2807]],

        [[ 0.6835, -0.3147, -0.2730,  0.8898, -0.4520, -2.3399],
         [-0.6476,  0.6417, -0.5508,  0.0535, -0.9227,  1.1172],
         [-1.7558,  0.7489, -0.9195,  1.3225, -0.7673, -0.9033]],

        [[-0.2535, -1.0029,  0.4376, -1.4911,  0.0859, -0.4517],
         [ 0.4292, -1.2318,  2.2034, -1.6605, -0.8760, -0.1399],
         [-1.0145,  0.1310, -1.1153, -1.5226,  0.4480, -0.8239]],

        [[-0.6559, -0.6905, -0.2348, -0.5770, -1.1463, -0.3917],
         [-0.0180,  1.0983,  1.4752,  0.7994,  1.4708, -0.6190],
         [ 0.5745,  1.1589, -0.0876, -0.7465, -1.4070,  1.3370]]])

         tags=tensor([[2, 4, 5],
        [0, 2, 4],
        [4, 3, 1],
        [2, 4, 1]])

        mask=tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
        [1, 1, 0]])
        也就是说第三个句子的实际长度是2，pad位置对应的tag是1
        '''
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all() # tests if all elements in this tensor is True，验证所有句子的第一个时间步必须是True

        seq_length, batch_size = tags.shape
        mask = mask.float()
        #tags[0].size()==(batch_size,)，代表的是batch内的所有句子的第一个单词对应的标注，也就是说三个句子的第一个单词对应的标注分别是2,4,5
        score=self.start_transitions[tags[0]]#也就是初始概率矩阵，即初始时刻观测到标注2或4或5的概率分别是多少
        score+=emissions[0,torch.arange(batch_size),tags[0]]#也就是第一个时间步，这三个句子分别观测到2,4,5的观测概率，分别是tensor([ 0.5480,  3.0301, -0.2807])

        #真实路径的分数等于这条路径上的发射分数+转移分数
        for i in range(1,seq_length):
            score+=self.transitions[tags[i-1],tags[i]]*mask[i]#这三个句子从第一个单词的标注转移到第二个单词的标注的转移概率
            score+=emissions[i,torch.arange(batch_size),tags[i]]*mask[i]
            #这三个句子在第i个时间步计算得到的前向概率

        seq_ends=mask.long().sum(0)-1#三个句子的实际长度 [4,4,2]
        last_time_step_tags=tags[seq_ends,torch.arange(batch_size)]#三个句子在最后一个单词的实际标注 [2,4,4]

        score+=self.end_transitions[last_time_step_tags]

        return score

    def compute_denominator(self,emissions,mask):
        '''
        emissions.size()=(max_seq_len.batch_size,num_tags)
        mask.size()=(max_seq_len,batch_size)
        '''
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        mask=mask.bool()
        assert mask[0].all()

        seq_length = emissions.size(0)

        score=self.start_transitions+emissions[0]#所有句子第一个时间步的前向概率,shape==(batch_size,num_tags)

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)#(batch_size,num_tags,1)
            broadcast_emissions = emissions[i].unsqueeze(1)#(batch_size,1,num_tags)

            next_score = broadcast_score + self.transitions + broadcast_emissions#(batch_size,num_tags,num_tags)

            next_score = torch.logsumexp(next_score, dim=1)#(batch_size,num_tags)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions#(batch_size,num_tags)

        return torch.logsumexp(score,dim=1)#(batch_size,)

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'token_mean',
            neg_loglikelihood=True):

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        #print(emissions.size(),tags.size(),mask.size())
        numerator=self.compute_numerator(emissions,tags=tags,mask=mask)
        denominator=self.compute_denominator(emissions=emissions,mask=mask)

        log_likelihood=numerator-denominator
        if reduction=='sum':
            log_likelihood=log_likelihood.sum()
        elif reduction=='mean':
            log_likelihood=log_likelihood.mean()
        elif reduction=='token_mean':
            log_likelihood=log_likelihood.sum()/mask.float().sum()
        else:
            raise Exception("reduction not recognized!")

        if neg_loglikelihood:
            return -log_likelihood
        else:
            return log_likelihood

    def viterbi_decode(self,emissions,mask):
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        #emissions.size()==(max_seq_len,batch_size,num_tags)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        mask=mask.bool()
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]#三个句子在第一个时间步的前向概率
        history = []
        
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)#(batch_size,num_tags,1)
            broadcast_emissions = emissions[i].unsqueeze(1)#(batch_size,1,num_tags)

            next_score = broadcast_score + self.transitions + broadcast_emissions#(batch_size,num_tags,num_tags)

            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)
        #返回的batch_tags_list是长度是3的列表，第一个元素是一个list，长度和第一个句子的长度一致，每一个值是
        #label2id中的id，代表当前时间步的预测
        return best_tags_list

class NerHead(nn.Module):
    def __init__(self, model_path,
                num_labels,
                state_dict=None,
                is_finetune=False,
                use_bilstm=False,
                use_crf=False):
        super().__init__()
        self.num_labels=num_labels
        self.use_bilstm=use_bilstm
        self.use_crf=use_crf

        if is_finetune==False:
            logger.info("Loading model from {}, which is from huggingface model".format(model_path))
            self.load_huggingface_model(bert_model_path=model_path)
        else:
            self.load_finetuned_model(model_path=model_path)
            logger.info("Loading model from {}, which has been finetuned.".format(model_path))

        # bert_model_path=os.path.join(model_path,"BERT")#save的时候将BERT保存在model_path下的BERT文件夹中
        # self.config=BertConfig.from_pretrained(bert_model_path)
        # self.bert=BertModel.from_pretrained(bert_model_path)
        # self.tokenizer=BertTokenizer.from_pretrained(bert_model_path)

    def load_huggingface_model(self,bert_model_path):
        self.config=BertConfig.from_pretrained(bert_model_path)
        self.bert=BertModel.from_pretrained(bert_model_path)
        self.tokenizer=BertTokenizer.from_pretrained(bert_model_path)
        if self.use_bilstm:
            self.lstmLayer=LSTMLayer(input_size=self.config.hidden_size,hidden_size=self.config.hidden_size,num_layers=1,batch_first=True,bidirectional=True)
            self.outputLayer=MLP(in_features=self.config.hidden_size*2,out_features=self.num_labels,bias=True)
        else:
            self.outputLayer=MLP(in_features=self.config.hidden_size,out_features=self.num_labels,bias=True)
        if self.use_crf:
            self.crfLayer=CRF(num_tags=self.num_labels,batch_first=True)

    def load_finetuned_model(self,model_path):
        bert_save_path=os.path.join(model_path,"BERT")#save的时候将BERT保存在model_path下的BERT文件夹中
        self.config=BertConfig.from_pretrained(bert_save_path)
        self.bert=BertModel.from_pretrained(bert_save_path)
        self.tokenizer=BertTokenizer.from_pretrained(bert_save_path)

        if self.use_bilstm:
            bistm_save_path=os.path.join(model_path,'BiLSTM')
            self.lstmLayer=LSTMLayer.load(bistm_save_path)
            self.outputLayer=MLP.load(os.path.join(model_path,'BiLSTM','MLP'))
        else:
            head_save_path=os.path.join(model_path,'MLP')
            self.outputLayer=MLP.load(head_save_path)

        if self.use_crf:
            crf_save_path=os.path.join(model_path,'CRF')
            self.crfLayer=CRF.load(crf_save_path)

    def save(self,output_path):
        bert_save_path=os.path.join(output_path,"BERT")
        self.bert.save_pretrained(bert_save_path,save_config=False)#下面已经save，不用save两次，虽然没什么影响
        self.config.save_pretrained(bert_save_path)
        self.tokenizer.save_pretrained(bert_save_path)
        if self.use_bilstm:
            bilstm_save_path=os.path.join(output_path,'BiLSTM')
            self.lstmLayer.save(bilstm_save_path)
            self.outputLayer.save(os.path.join(output_path,'BiLSTM','MLP'))
        else:
            head_save_path=os.path.join(output_path,'MLP')
            self.outputLayer.save(head_save_path)
        if self.use_crf:
            crf_save_path=os.path.join(output_path,'CRF')
            self.crfLayer.save(crf_save_path)
    
    def forward(self,input_ids,attention_mask=None,token_type_ids=None,label_ids=None,output_all_encoded_layers=False):
        '''
        input_ids.size()==attention_mask.size()==token_type_ids.size()==position_ids.size()==(batch_size,seq_length)
        label_ids.size()==(batch_size,)
        '''
        batch_size,max_seq_len=input_ids.size()
        (sequence_outputs,pooled_output)=self.bert(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask,
                                                   output_all_encoded_layers=output_all_encoded_layers)
        #要注意到sequence_output[0]与pooled_output的区别在于pooled_output是经过一层tanh的
        #assert len(pooled_output.size())==2 and pooled_output.size(1)==self.config.hidden_size
        if self.use_bilstm:
            sequence_outputs=self.lstmLayer(sequence_outputs,attention_mask=attention_mask)
            logits=self.outputLayer(sequence_outputs)
        else:
            logits=self.outputLayer(sequence_outputs)
            
        assert logits.size()==(batch_size,max_seq_len,self.num_labels)
        return logits
        
        # if self.use_crf:
        #     loss=self.crfLayer(emissions=sequence_outputs,tags=label_ids,mask=attention_mask)
        #     return loss
        # else:
        #     return logits
