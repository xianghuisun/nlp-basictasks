import json
from logging import log
import numpy as np
import os,sys
from typing import Dict, Sequence, Text, Type, Callable, List, Optional
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from tensorboardX import SummaryWriter
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#print(sys.path)
from heads import NerHead
from log import logging

from readers.ner import convert_examples_to_features,InputExample
from modules.utils import get_optimizer,get_scheduler
from .utils import batch_to_device,eval_during_training

logger=logging.getLogger(__name__)

class Ner():
    def __init__(self,model_path,
                label2id,
                max_seq_length:int = 64,
                device:str = None,
                use_bilstm=False,
                use_crf=False,
                state_dict=None,
                is_finetune=False,
                tensorboard_logdir = None,
                batch_first=True):

        if label2id is None:
            assert os.path.exists(os.path.join(model_path,'label2id.json'))
            with open(os.path.join(model_path,'label2id.json')) as f:
                label2id=json.load(f)
            logger.info("Load label2id from {}".format(os.path.join(model_path,'label2id.json')))
        else:
            if not os.path.exists(os.path.join(model_path,"label2id.json")):
                with open(os.path.join(model_path,"label2id.json"),'w') as f:
                    json.dump(label2id,fp=f)#及时保存label2id
            logger.info('label2id has been saved in {}'.format(os.path.join(model_path,'label2id.json')))


        self.label2id=label2id
        self.use_crf=use_crf
        self.use_bilstm=use_bilstm
        self.batch_first=batch_first
        assert self.label2id["[PAD]"]==0#将[CLS],[SEP],[PAD]这三个特殊字符全部都看作[PAD]处理，也就是将[PAD]作为一个标签
        self.id2label={id_:label for label,id_ in self.label2id.items()}
        self.num_labels=len(label2id)
        self.max_seq_lenth=max_seq_length
        logger.info("The label2id is\n {}".format(json.dumps(self.label2id,ensure_ascii=False)))

        self.model=NerHead(model_path=model_path,
                            num_labels=self.num_labels,
                            state_dict=state_dict,
                            is_finetune=is_finetune,
                            use_bilstm=use_bilstm,
                            use_crf=use_crf)

        if tensorboard_logdir!=None:
            self.tensorboard_writer=SummaryWriter(tensorboard_logdir)
        else:
            self.tensorboard_writer=None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))
        self._target_device = torch.device(device)
        self.model.to(self._target_device)
        logger.info("Using BiLSTM ? : {}".format(self.use_bilstm))
        logger.info("Using CRF ? : {}".format(self.use_crf))
        
    def smart_batching_collate(self,batch):
        features=convert_examples_to_features(examples=batch,tokenizer=self.model.tokenizer,max_seq_len=self.max_seq_lenth,label2id=self.label2id)
        return features#features.keys()==[input_ids,token_type_ids,attention_mask,label_ids]

    def getLoss(self,logits,labels,attention_mask):
        '''
        last_hidden_state.size()==(batch_size,seq_length,hidden_size) 
        labels.size()==(batch_size,seq_length)
        '''
        if self.use_crf:
            loss=self.model.crfLayer(emissions=logits,tags=labels,mask=attention_mask)
        else:
            mask_pad=attention_mask.view(-1)==1
            logits=logits.view(-1,self.num_labels)[mask_pad]
            labels=labels.view(-1)[mask_pad]
            loss=nn.CrossEntropyLoss(ignore_index=0)(logits,labels)
        # label_one_hot=nn.functional.one_hot(labels,num_classes=self.num_labels).to(labels.device)
        # #print(label_one_hot.size(),logits.size())
        # assert logits.size()==(self.current_batch_size,self.current_seq_len,self.num_labels)==label_one_hot.size()
        # loss=-torch.sum(torch.log_softmax(logits,dim=-1)*label_one_hot)/(self.current_batch_size*self.current_seq_len)
        return loss

    def maskPredictions(self,predictions,attention_mask):
        if type(attention_mask)!=list:
            attention_mask=attention_mask.cpu().tolist()
        masked_result=[]
        for prediction,mask in zip(predictions,attention_mask):
            if 0 not in mask:
                #[1,1,1,1,1,1,1]#此时[SEP]是最后一个
                masked_result.append(prediction[1:-1])
            else:
                #[1,1,1,1,1,1,0]#此时[SEP]是倒数第二个
                end_pos=mask.index(0)
                masked_result.append(prediction[1:end_pos-1])
        return masked_result

    def getPredictions(self,features):
        features=batch_to_device(features,self._target_device)
        attention_mask=features["attention_mask"]#(batch_size,pad_seq_len)
        logits=self.model(**features)#(batch_size,pad_seq_len,num_labels)
        probs=torch.nn.functional.softmax(logits,dim=2)
        if not self.use_crf:
            predicts=torch.argmax(probs,dim=2).cpu().numpy().tolist()
        else:
            predicts=self.model.crfLayer.viterbi_decode(emissions=logits,mask=attention_mask)
        predicts=self.maskPredictions(predictions=predicts,attention_mask=attention_mask)
        #在viterbi_decode返回的已经是不包含有mask的list了
        predicts_tags=[]
        for predict in predicts:
            temp=[]
            for id_ in predict:
                tag=self.id2label[id_]
                temp.append(tag)
            predicts_tags.append(temp)
        return predicts_tags

    def fit(self,
            train_dataloader,
            evaluator = None,
            epochs: int = 1,
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

            for train_step,features_and_labels in tqdm(enumerate(train_dataloader)):
                features_and_labels=batch_to_device(features_and_labels,target_device=self._target_device)
                labels=features_and_labels['label_ids']
                self.current_batch_size,self.current_seq_len=labels.size()
                features_and_labels['output_all_encoded_layers']=output_all_encoded_layers
                #print(features.keys(),features["input_ids"].size())
                if use_amp:
                    with autocast():
                        logits=self.model(**features_and_labels)
                        loss_value = self.getLoss(logits=logits,labels=labels,attention_mask=features_and_labels['attention_mask'])

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    logits=self.model(**features_and_labels)
                    loss_value = self.getLoss(logits=logits,labels=labels,attention_mask=features_and_labels['attention_mask'])
                    
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
                    eval_score=eval_during_training(self, evaluator, output_path, epoch, training_steps, callback)
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

    def predict(self,
                dataloader,
                batch_size:int=32,
                num_workers: int = 0,
                output_all_encoded_layers = False,
                show_progress_bar=False,
                input_is_string=True,
                input_is_list=False):
        '''
        nerEvaluator传进来的是nerEvaluator.seq_in，其中每一个元素正是一个句子
        '''
        if type(dataloader)==str:
            dataloader=[dataloader]
        if input_is_string:
            #说明此时传进来的是句子，不是InputExample类型
            dataloader=[InputExample(seq_in=self.model.tokenizer.tokenize(text)) for text in dataloader]
        if input_is_list:
            dataloader=[InputExample(seq_in=seq_in) for seq_in in dataloader]

        dataloader=DataLoader(dataloader,batch_size=batch_size,num_workers=num_workers,shuffle=False)
        dataloader.collate_fn=self.smart_batching_collate
        
        predictions=[]
        self.model.eval()#不仅仅是bert_model
        self.model.to(self._target_device)
        with torch.no_grad():
            for _ in trange(1, desc="Evaluating", disable=not show_progress_bar):
                for features in dataloader:
                    features['output_all_encoded_layers']=output_all_encoded_layers
                    batch_predicts_tags=self.getPredictions(features)
                    predictions.extend(batch_predicts_tags)
        
        return predictions#(num_eval_examples,seq_len)