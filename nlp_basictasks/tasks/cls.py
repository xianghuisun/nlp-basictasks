import json
import numpy as np
import os,sys
from typing import Dict, Sequence, Type, Callable, List, Optional
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from tensorboardX import SummaryWriter
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#print(sys.path)
from heads import ClsHead
from log import logging

import readers.cls as single_cls
import readers.paircls as pair_cls
from modules.utils import get_optimizer,get_scheduler
from .utils import batch_to_device

logger=logging.getLogger(__name__)

class cls():
    '''
    分类任务上，除了readers中区分单句子还是双句子，
    heads中是不区分的，
    在tasks中只需在fit时指明is_pairs即可
    '''
    def __init__(self,model_path,
                label2id=None,
                max_seq_length:int = 64,
                device:str = None,
                state_dict=None,
                is_finetune=False,
                tensorboard_logdir = None):

        if is_finetune:
            logger.info("Model has been finetuned, so the label2id must be saved in {}".format(os.path.join(model_path,'label2id.json')))
            with open(os.path.join(model_path,'label2id.json')) as f:
                label2id=json.load(f)
        else:
            assert label2id is not None

        self.label2id=label2id
        self.num_labels=len(label2id)
        self.max_seq_lenth=max_seq_length
        logger.info("The label2id is\n {}".format(json.dumps(self.label2id,ensure_ascii=False)))

        self.model=ClsHead(model_path=model_path,
                            num_labels=self.num_labels,
                            state_dict=state_dict,
                            is_finetune=is_finetune)
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

    def smart_batching_collate_of_single(self,batch):
        features,labels=single_cls.convert_examples_to_features(examples=batch,tokenizer=self.model.tokenizer,max_seq_len=self.max_seq_lenth)
        return features,labels
    
    def smart_batching_collate_of_pair(self,batch):
        features,labels=pair_cls.convert_examples_to_features(examples=batch,tokenizer=self.model.tokenizer,max_seq_len=self.max_seq_lenth)
        return features,labels
    
    def fit(self,
            is_pairs,
            train_dataloader,
            evaluator = None,
            epochs: int = 1,
            loss_fct = None,
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
        if not os.path.exists(os.path.join(output_path,"label2id.json")):
            with open(os.path.join(output_path,"label2id.json"),'w') as f:
                json.dump(self.label2id,fp=f)#及时保存label2id
            logger.info("label2id has been saved in {}".format(os.path.join(output_path,"label2id.json")))
        #适用于单句子分类和双句子分类任务
        if is_pairs==False:
            logger.info("当前是单句子分类任务")
            train_dataloader.collate_fn=self.smart_batching_collate_of_single
        else:
            logger.info("当前是双句子分类任务")
            train_dataloader.collate_fn=self.smart_batching_collate_of_pair

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
        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.num_labels == 1 else nn.CrossEntropyLoss()

        global_step=0
        skip_scheduler = False
        patience = early_stop_patience        
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps=0
            training_loss=0.0

            self.model.zero_grad()
            self.model.train()

            for train_step,(features,labels) in tqdm(enumerate(train_dataloader)):
                features=batch_to_device(features,target_device=self._target_device)
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
            score_and_auc = evaluator(self, label2id=self.label2id, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score_and_auc, epoch, steps)
            return score_and_auc
        return None

    def predict(self,
                is_pairs,
                dataloader,
                batch_size:int=32,
                num_workers: int = 0,
                convert_to_numpy: bool = True,
                convert_to_tensor: bool = False,
                output_all_encoded_layers = False,
                show_progress_bar=False):
        if isinstance(dataloader,list):
            #传进来的dataloader是一个List
            dataloader=DataLoader(dataloader,batch_size=batch_size,num_workers=num_workers,shuffle=False)
        if is_pairs==False:
            logger.info('当前是单句子分类任务预测')
            dataloader.collate_fn=self.smart_batching_collate_of_single
        else:
            logger.info("当前是双句子分类任务预测")
            dataloader.collate_fn=self.smart_batching_collate_of_pair
        
        predictions=[]
        self.model.eval()#不仅仅是bert_model
        self.model.to(self._target_device)
        with torch.no_grad():
            for _ in trange(1, desc="Evaluating", disable=not show_progress_bar):
                for features,_ in tqdm(dataloader):
                    features['output_all_encoded_layers']=output_all_encoded_layers
                    features=batch_to_device(features,self._target_device)
                    logits=self.model(**features)
                    probs=torch.nn.functional.softmax(logits,dim=1)#(batch_size,num_labels)
                    predictions.extend(probs)
        if convert_to_tensor:
            predictions=torch.stack(predictions)#(num_eval_examples,num_classes)
        
        elif convert_to_numpy:
            predictions=np.asarray([predict.cpu().detach().numpy() for predict in predictions])

        return predictions#(num_eval_examples,num_classes)