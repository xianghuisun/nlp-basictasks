import numpy as np
import os,json,sys
from typing import Dict, Sequence, Type, Callable, List, Optional
import torch
from torch import nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from modules.transformers import BertModel,BertConfig,BertTokenizer
from modules.MLP import MLP
from modules.Pooling import Pooling
from log import logging
logger=logging.getLogger(__name__)

'''
heads下的所有modelhead返回的都是logits，骨架中的module和具体的head model都用head进行load和save .head的输入是
就是input_ids这些特征，因此module和heads其实是绑在一起的

在语义相似度匹配任务中，骨架的设定是BiEncoders，heads的设定主要是不同的loss model，包括（softmaxloss，
ContrastiveLoss等）

每一个heads的forward返回的都是logits，比如在SoftmaxLossHead中的logits就是两个句子Pooling后的两个sentence_embeddings拼接后
传入到最终的classifier得到的logits，由于是两个句子，因此为了简便，forward的参数不是input_ids1和input_ids2这样区分
，而是[{'input_ids':tensor,'attention_mask':tensor,'token_type_ids':tensor},{'input_ids':tensor,'attention_mask':tensor,'token_type_ids':tensor}]
'''

class SoftmaxLossHead(nn.Module):
    """
    model应该根据对应的features给出句子的embedding，
    """
    def __init__(self,
                 model_path,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 pooling_mode_mean_last_2_tokens: bool = False,
                 pooling_mode_mean_first_last_tokens: bool = False,
                 is_finetune=False,
                 ):
        super(SoftmaxLossHead, self).__init__()
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_mean_last_2_tokens = pooling_mode_mean_last_2_tokens
        self.pooling_mode_mean_first_last_tokens = pooling_mode_mean_first_last_tokens

        self.num_vectors_concatenated = 0
        if concatenation_sent_rep:
            self.num_vectors_concatenated += 2
        if concatenation_sent_difference:
            self.num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            self.num_vectors_concatenated += 1

        self.config_keys = ['concatenation_sent_rep','concatenation_sent_difference','concatenation_sent_multiplication',
                            'sentence_embedding_dimension', 'pooling_mode_cls_token', 'pooling_mode_mean_tokens',
                            'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens',
                            'pooling_mode_mean_last_2_tokens','pooling_mode_mean_first_last_tokens']
        self.pooling_config = {'pooling_mode_cls_token':self.pooling_mode_cls_token,
                                'pooling_mode_max_tokens':self.pooling_mode_max_tokens,
                                'pooling_mode_mean_tokens':self.pooling_mode_mean_tokens,
                                'pooling_mode_mean_sqrt_len_tokens':self.pooling_mode_mean_sqrt_len_tokens,
                                'pooling_mode_mean_last_2_tokens':self.pooling_mode_mean_last_2_tokens,
                                'pooling_mode_mean_first_last_tokens':self.pooling_mode_mean_first_last_tokens
                                }

        if is_finetune==False:
            logger.info("Loading model from {}, which is from huggingface model".format(model_path))
            self.load_huggingface_model(bert_model_path=model_path)
        else:
            self.load_finetuned_model(model_path=model_path)
            logger.info("Loading model from {}, which has been finetuned.".format(model_path))
        
        #logger.info("Pooling config : {}".format(self.pooling_config))
        logger.info("Softmax loss: #Vectors concatenated: {}".format(self.num_vectors_concatenated))
        logger.info("Pooling policy is ")
        logger.info("After pooling, each sentence embedding has dim: {}".format(self.pooling_layer.pooling_output_dimension))
    def load_huggingface_model(self,bert_model_path):
        self.config=BertConfig.from_pretrained(bert_model_path)
        self.bert=BertModel.from_pretrained(bert_model_path)
        self.tokenizer=BertTokenizer.from_pretrained(bert_model_path)
        self.pooling_config.update({'word_embedding_dimension':self.bert.config.hidden_size})
        self.pooling_layer=Pooling(**self.pooling_config)
        self.head_layer=MLP(in_features=self.num_vectors_concatenated*self.pooling_layer.pooling_output_dimension,out_features=self.num_labels)

    def load_finetuned_model(self,model_path):
        bert_save_path=os.path.join(model_path,"BERT")#save的时候将BERT保存在model_path下的BERT文件夹中
        self.config=BertConfig.from_pretrained(bert_save_path)
        self.bert=BertModel.from_pretrained(bert_save_path)
        self.tokenizer=BertTokenizer.from_pretrained(bert_save_path)

        pooling_save_path=os.path.join(model_path,'Pooling')
        self.pooling_layer.load(pooling_save_path)

        head_save_path=os.path.join(model_path,'MLP')
        self.head_layer=MLP.load(input_path=head_save_path)

    def save(self,output_path):
        bert_save_path=os.path.join(output_path,"BERT")
        self.bert.save_pretrained(bert_save_path,save_config=False)#下面已经save，不用save两次，虽然没什么影响
        self.config.save_pretrained(bert_save_path)
        self.tokenizer.save_pretrained(bert_save_path)

        pooling_save_path=os.path.join(output_path,'Pooling')
        self.pooling_layer.save(pooling_save_path)#主要是保存config.json，Pooling没有参数

        head_save_path=os.path.join(output_path,'MLP')
        self.head_layer.save(head_save_path)

    def forward(self,sentence_features_of_1,sentence_features_of_2=None,output_all_encoded_layers=False,encode_pattern=False):
        '''
        each_features like : {'input_ids':tensor,'attention_mask':tensor,'token_type_ids':tensor},
        input_ids.size()==attention_mask.size()==token_type_ids.size()==position_ids.size()==(batch_size,seq_length)
        label_ids.size()==(batch_size,)
        '''
        #只有在encode模式下的single_batch才是有意义的，不然如果不是encode模式，只传入一个句子，有没有标签，无法返回任何值
        single_batch=False
        if sentence_features_of_2 is None:
            single_batch=True
            try:
                assert encode_pattern
            except:
                raise Exception("只传入了一个batch的句子，然而又不是encode模式，函数无法执行")
            pair_sentence_features=[sentence_features_of_1]
        else:
            pair_sentence_features=[sentence_features_of_1,sentence_features_of_2]
        batch_size,seq_len_1=sentence_features_of_1['input_ids'].size()

        pair_sentence_embeddings=[]
        for sentence_features in pair_sentence_features:
            input_ids=sentence_features['input_ids']
            token_type_ids=sentence_features['token_type_ids']
            attention_mask=sentence_features['attention_mask']

            (sequence_outputs,pooler_output)=self.bert(input_ids=input_ids,
                                                    token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask,
                                                    output_all_encoded_layers=output_all_encoded_layers)
            #要注意到sequence_output[0]与pooled_output的区别在于pooler_output是经过一层tanh的
            if output_all_encoded_layers:
                all_layer_embeddings=sequence_outputs
                token_embeddings=sequence_outputs[-1]
            else:
                all_layer_embeddings=None
                token_embeddings=sequence_outputs

            cls_token_embeddings=pooler_output
            sentence_embedding=self.pooling_layer(token_embeddings=token_embeddings,
                                                cls_token_embeddings=cls_token_embeddings,
                                                attention_mask=attention_mask,
                                                all_layer_embeddings=all_layer_embeddings)
            pair_sentence_embeddings.append(sentence_embedding)
            assert sentence_embedding.size()==(batch_size,self.pooling_layer.pooling_output_dimension)

        if single_batch:
            rep_a=pair_sentence_embeddings[0]
        else:
            rep_a,rep_b=pair_sentence_embeddings

        if encode_pattern == True:
            if single_batch:
                return rep_a
            else:
                return rep_a,rep_b
        try:
            assert sentence_features_of_2 is not None and len(pair_sentence_embeddings)==2
        except:
            raise Exception("encode pattern是False，那么第二个batch不能为空")

        assert rep_a.size()==(batch_size,self.pooling_layer.pooling_output_dimension)==rep_b.size()
        vectors_concat=[]
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)
        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a-rep_b))
        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a*rep_b)
        
        concat_embeddings = torch.cat(vectors_concat, 1)
        assert concat_embeddings.size()==(batch_size,self.num_vectors_concatenated*self.pooling_layer.pooling_output_dimension)
        logits=self.head_layer(concat_embeddings)#(batch_size,num_labels)
        return logits

class STSHead(nn.Module):
    '''
    ClsHead不区分是单句子还是双句子，因为处理逻辑是一样的
    '''
    def __init__(self, model_path,
                num_labels,
                state_dict=None,
                is_finetune=False):
        super().__init__()
        self.num_labels=num_labels

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
        self.head_layer=MLP(in_features=self.config.hidden_size,out_features=self.num_labels)

    def load_finetuned_model(self,model_path):
        bert_save_path=os.path.join(model_path,"BERT")#save的时候将BERT保存在model_path下的BERT文件夹中
        self.config=BertConfig.from_pretrained(bert_save_path)
        self.bert=BertModel.from_pretrained(bert_save_path)
        self.tokenizer=BertTokenizer.from_pretrained(bert_save_path)

        head_save_path=os.path.join(model_path,'MLP')
        self.head_layer=MLP.load(input_path=head_save_path)

    def save(self,output_path):
        bert_save_path=os.path.join(output_path,"BERT")
        self.bert.save_pretrained(bert_save_path,save_config=False)#下面已经save，不用save两次，虽然没什么影响
        self.config.save_pretrained(bert_save_path)
        self.tokenizer.save_pretrained(bert_save_path)
        head_save_path=os.path.join(output_path,'MLP')
        self.head_layer.save(head_save_path)
    
    def forward(self,input_ids,attention_mask=None,token_type_ids=None,label_ids=None,output_all_encoded_layers=False):
        '''
        input_ids.size()==attention_mask.size()==token_type_ids.size()==position_ids.size()==(batch_size,seq_length)
        label_ids.size()==(batch_size,)
        '''
        (sequence_outputs,pooled_output)=self.bert(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask,
                                                   output_all_encoded_layers=output_all_encoded_layers)
        #要注意到sequence_output[0]与pooled_output的区别在于pooled_output是经过一层tanh的
        assert len(pooled_output.size())==2 and pooled_output.size(1)==self.config.hidden_size
        logits=self.head_layer(pooled_output)#(batch_size,num_labels)
        return logits
        # predictions=torch.argmax(logits,dim=1)
        # if label_ids is not None:
        #     loss=nn.CrossEntropyLoss(reduction="mean")(input=logits,target=label_ids)
        #     accuracy=(predictions==label_ids).float().mean()
        #     return loss,accuracy
        # else:
        #     return logits,predictions