import json
from logging import log
from nlp_basictasks.modules.transformers.modeling_bert import BertModel
from nlp_basictasks.modules.transformers.tokenization_bert import BertTokenizer
import numpy as np
import os,sys
from typing import Dict, Sequence, Text, Type, Callable, List, Optional, Union
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from tensorboardX import SummaryWriter
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#print(sys.path)

from log import logging

from modules.utils import get_optimizer,get_scheduler
from heads import SoftmaxLossHead
from modules.Pooling import Pooling
from modules.MLP import MLP
from readers.sts import convert_examples_to_features,convert_sentences_to_features
from .utils import batch_to_device,eval_during_training

logger=logging.getLogger(__name__)

class sts():
    def __init__(self,model_path,
                max_seq_length:int = 32,
                head_name:str = 'softmaxloss',
                head_config: dict = None,
                device:str = None,
                state_dict=None,
                is_finetune=False,
                tensorboard_logdir = None):
        '''
        head_name用来指明使用哪一个loss，包括softmaxloss和各种基于对比学习的loss
        head_config用来给出具体loss下所需要的参数，
        '''
        self.head_name=head_name
        self.head_config=head_config
        self.max_seq_lenth=max_seq_length
        if self.head_name=='softmaxloss':
            if self.head_config is None:
                self.head_config={'concatenation_sent_rep':True,
                                'concatenation_sent_difference':True,
                                'pooling_mode_mean_tokens':True}
            label2id={'0':0,'1':1}
            self.model=SoftmaxLossHead(model_path=model_path,
                                        num_labels=len(label2id),
                                        is_finetune=is_finetune,
                                        **self.head_config)
        
        else:
            raise Exception("Unknown loss {} for sts".format(head_name))

        if tensorboard_logdir!=None:
            self.tensorboard_writer=SummaryWriter(tensorboard_logdir)
        else:
            self.tensorboard_writer=None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))
        self._target_device = torch.device(device)
        self.model.to(self._target_device)

    def smart_batching_collate(self,batch):
        features_of_a,features_of_b,labels=convert_examples_to_features(examples=batch,tokenizer=self.model.tokenizer,max_seq_len=self.max_seq_lenth)
        return features_of_a,features_of_b,labels

    def encode(self, sentences,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False):
        '''
        传进来的sentences只能是single_batch
        '''
        self.model.eval()
        input_is_string=False
        if type(sentences)==str:
            input_is_string=True
            sentences=[sentences]
        if convert_to_tensor:
            convert_to_numpy = False
        if device is None:
            device = self._target_device

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features=convert_sentences_to_features(sentences_batch,tokenizer=self.model.tokenizer)
            features=batch_to_device(features,device)
            with torch.no_grad():
                embeddings=self.model(sentence_features_of_1=features,sentence_features_of_2=None,encode_pattern=True)
                embeddings=embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]#一个hidden_size维度的vector

        return all_embeddings

    def fit(self,
            train_dataloader,
            evaluator = None,
            epochs: int = 1,
            loss_fct = nn.CrossEntropyLoss(),
            scheduler: str = 'WarmupLinear',
            warmup_proportion: float = 0.1,
            optimizer_type = 'AdamW',
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps = None,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            early_stop_patience = 10,
            print_loss_step: Optional[int] = None,
            output_all_encoded_layers: bool = False
            ):

        train_dataloader.collate_fn=self.smart_batching_collate
        if print_loss_step==None:
            print_loss_step=len(train_dataloader)//5
        if evaluator is not None and evaluation_steps==None:
            evaluation_steps=len(train_dataloader)//2
        #一个epoch下打印5次loss，评估2次
        logger.info("一个epoch 下，每隔{}个step会输出一次loss，每隔{}个step会评估一次模型".format(print_loss_step,evaluation_steps))
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)#map_location='cpu'

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)
        warmup_steps = num_train_steps*warmup_proportion

        optimizer = get_optimizer(model=self.model,optimizer_type=optimizer_type,weight_decay=weight_decay,optimizer_params=optimizer_params)
        scheduler = get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        global_step=0
        skip_scheduler = False
        patience = early_stop_patience        
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps=0
            training_loss=0.0

            self.model.zero_grad()
            self.model.train()

            for train_step,(features_of_a,features_of_b,labels) in tqdm(enumerate(train_dataloader)):
                features_of_a=batch_to_device(features_of_a,target_device=self._target_device)
                features_of_b=batch_to_device(features_of_b,target_device=self._target_device)
                features={'sentence_features_of_1':features_of_a,
                        'sentence_features_of_2':features_of_b}
                labels=labels.to(self._target_device)
                features['output_all_encoded_layers']=output_all_encoded_layers
                #print(features.keys(),features["input_ids"].size())
                if use_amp:
                    with autocast():
                        logits=self.model(**features)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    logits=self.model(**features)
                    loss_value=loss_fct(logits,labels)#CrossEntropyLoss
                    training_loss+=loss_value.item()
                    if print_loss_step!=None and train_step>0 and train_step%print_loss_step == 0:
                        training_loss/=print_loss_step
                        logging.info("Epoch : {}, train_step : {}/{}, loss_value : {} ".format(epoch,train_step*(epoch+1),num_train_steps,training_loss))
                        training_loss=0.0
                    if self.tensorboard_writer is not None:
                        self.tensorboard_writer.add_scalar(f"train_loss",loss_value.item(),global_step=global_step)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()
                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps>0 and evaluator is not None and (train_step) % evaluation_steps == 0 :
                    eval_score=self._eval_during_training(evaluator, output_path, epoch, training_steps, callback)
                    if self.tensorboard_writer is not None:
                        self.tensorboard_writer.add_scalar(f"eval_score",float(eval_score),global_step=global_step)
                    if eval_score > self.best_score:
                        if save_best_model:
                            self.model.save(output_path=output_path)
                            logging.info(("In epoch {}, training_step {}, the eval score is {}, previous eval score is {}, model has been saved in {}".format(epoch,train_step*(epoch+1),eval_score,self.best_score,output_path)))
                            self.best_score=eval_score
                    else:
                        patience-=1
                        logging.info(f"No improvement over previous best eval score ({eval_score:.6f} vs {self.best_score:.6f}), patience = {patience}")
                        if patience==0:
                            logging.info("Run our of patience, early stop!")
                            return
                
                    self.model.zero_grad()
                    self.model.train()

    def _eval_during_training(self, evaluator, output_path,  epoch, steps, callback):
        if evaluator is not None:
            score_and_auc = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score_and_auc, epoch, steps)
            return score_and_auc
        return None



class SimilarityRetrieve(nn.Module):
    def __init__(self,bert_model_path,
                pooling_model_path=None,
                pooling_config={'pooling_mode_mean_tokens':True},
                mlp_model_path=None,
                max_seq_length:int = 32,
                device:str = None):
        '''
        
        '''
        super(SimilarityRetrieve,self).__init__()
        #logger.info("Bert model path and pooling model path is required")
        self.max_seq_lenth=max_seq_length
        if not os.path.exists(os.path.join(bert_model_path,'pytorch_model.bin')):
            raise Exception('Not found pytorch_model.bin in bert model path : {}'.format(bert_model_path))
        else:
            assert os.path.exists(os.path.join(bert_model_path,'config.json'))
            self.bert=BertModel.from_pretrained(bert_model_path)
            self.tokenizer=BertTokenizer.from_pretrained(bert_model_path)
            pooling_config.update({'word_embedding_dimension':self.bert.config.hidden_size})
        
        if pooling_model_path==None or not os.path.exists(os.path.join(pooling_model_path,'config.json')):
            self.pooling=Pooling(**pooling_config)
        else:
            self.pooling=Pooling.load(pooling_model_path)
        
        if mlp_model_path is not None:
            logger.info("The MLP model path is not empty, which means you will project the vector after pooling")
            if not os.path.exists(os.path.join(mlp_model_path,'pytorch_model.bin')):
                raise Exception('Not found pytorch_model.bin in MLP model path : {}'.format(mlp_model_path))
            else:
                assert os.path.exists(os.path.join(mlp_model_path,'config.json'))
                self.mlp=MLP.load(mlp_model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))
        self._target_device=device
        self.to(device)

    def smart_batching_collate(self,batch):
        features_of_a,features_of_b,labels=convert_examples_to_features(examples=batch,tokenizer=self.model.tokenizer,max_seq_len=self.max_seq_lenth)
        return features_of_a,features_of_b,labels

    def encode(self, sentences,
               batch_size: int = 32,
               show_progress_bar: bool = True,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False,
               output_all_encoded_layers=False):
        '''
        传进来的sentences只能是single_batch
        '''
        self.eval()
        input_is_string=False
        if type(sentences)==str:
            input_is_string=True
            sentences=[sentences]
        if convert_to_tensor:
            convert_to_numpy = False
        if device is None:
            device = self._target_device

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features=convert_sentences_to_features(sentences_batch,tokenizer=self.tokenizer)
            features=batch_to_device(features,device)
            with torch.no_grad():
                embeddings=self.encoding_feature_to_embedding(sentence_features=features,output_all_encoded_layers=output_all_encoded_layers)
                embeddings=embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]#一个hidden_size维度的vector

        return all_embeddings

    def encoding_feature_to_embedding(self,sentence_features,output_all_encoded_layers=False):
        '''
        each_features like : {'input_ids':tensor,'attention_mask':tensor,'token_type_ids':tensor},
        input_ids.size()==attention_mask.size()==token_type_ids.size()==position_ids.size()==(batch_size,seq_length)
        label_ids.size()==(batch_size,)
        '''
        #只有在encode模式下的single_batch才是有意义的，不然如果不是encode模式，只传入一个句子，有没有标签，无法返回任何值
        batch_size,seq_len_1=sentence_features['input_ids'].size()

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
        sentence_embedding=self.pooling(token_embeddings=token_embeddings,
                                            cls_token_embeddings=cls_token_embeddings,
                                            attention_mask=attention_mask,
                                            all_layer_embeddings=all_layer_embeddings)
        assert sentence_embedding.size()==(batch_size,self.pooling.pooling_output_dimension)
        return sentence_embedding