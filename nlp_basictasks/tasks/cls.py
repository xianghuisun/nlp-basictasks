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
                tensorboard_logdir = None,
                pooling_type='cls',
                do_FGV=False):

        if is_finetune:
            logger.info("Model has been finetuned, so the label2id must be saved in {}".format(os.path.join(model_path,'label2id.json')))
            with open(os.path.join(model_path,'label2id.json')) as f:
                label2id=json.load(f)
        else:
            assert label2id is not None

        self.label2id=label2id
        self.do_FGV=do_FGV
        self.num_labels=len(label2id)
        self.max_seq_lenth=max_seq_length
        self.pooling_type=pooling_type
        logger.info("The label2id is\n {}".format(json.dumps(self.label2id,ensure_ascii=False)))
        logger.info("Pooling type is {} for classification task.".format(self.pooling_type))
        logger.info("Doing attack traing : {}".format(do_FGV))
        self.model=ClsHead(model_path=model_path,
                            num_labels=self.num_labels,
                            state_dict=state_dict,
                            is_finetune=is_finetune,
                            pooling_type=pooling_type)
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

    def getLoss(self,features,labels,loss_fct,eps=1e-10,noise_coeff=0.01):
        '''
        eps和noise_coeff是做对抗时用到的
        PyTorch一些函数说明：
        Tensor.grad 首先tensor的梯度在forward阶段存储的是None,当调用backward()时，tensor.grad将会存储
        自身在计算图中的梯度。如果不进行清零，那么下一个batch的backward()计算的梯度将会进行累加
        detach_，a.detach_()操作会使得a变成叶子节点，从计算图中分离出来，a之前的计算图将不被传回梯度
        也就是a阻断了a之前的计算图，a变成了叶子节点，不过a同样不会被更新。

        对抗训练的流程是，前向计算得到normal_loss,然后调用backward得到embedding的梯度，取出embedding的梯度
        计算扰动，将扰动重新输入到模型中得到adv_loss。
        流程中需要注意的事项包括：1) normal_loss.backward()之前，需要指明embedding.retain_grad=True
        2) 对抗的扰动取自embedding.grad，不过embedding并不能作为对抗阶段计算图的一部分，而且扰动也不需要更新，所以
        一定要有：embedding.grad.detach_()
        3) 扰动的前向计算时，需要便利当前model的每一个参数，如果该参数的梯度不空，那么清空该参数的梯度，不然就会和
        对抗阶段的参数梯度出现累计
        '''
        if self.do_FGV==False:
            logits=self.model(**features)
            loss_value = loss_fct(logits, labels)
        else:
            ###########################################正常的前向计算##############################
            logits=self.model(**features)
            current_word_embedding=self.model.bert.get_most_recent_embedding()#(bsz,seq_len,dim)
            current_word_embedding.retain_grad()#如果想要保存非叶子节点的grad，需要指明retain_grad
            normal_loss = loss_fct(logits, labels)
            normal_loss.backward(retain_graph=True)
            ###########################################对抗阶段###################################
            unnormalized_noise=current_word_embedding.grad.detach_()#如果没有指明retain_grad(),那么此时的梯度是None
            #unnormalized_noise.required_grad=False，将current_word_embedding从对抗阶段的计算图中分离出来
            normalized_noise=unnormalized_noise/(unnormalized_noise.norm(p=2,dim=-1).unsqueeze(dim=-1)+eps)#x/||x||
            #unnormalized_noise只是noise,unnormalized_noise作为对抗阶段的叶子节点，是不需要更新的，这和normal阶段的
            #emmbedding是不同的，normal阶段的embedding需要更新，我们只需要更新的是网络参数，使得网络参数
            #对扰动鲁棒，而扰动当然是不需要更新的，即使它是从embedding的梯度得到的
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()#已经得到了noise,所以其余的节点的梯度都可以清零
                    #比避免loss_value调用backward时出现梯度累计

            adv=noise_coeff*normalized_noise
            #接下来就是把扰动添加到word_embedding上
            adv_embedding=current_word_embedding+adv
            #将adv_embedding视为新的embedding重新过12层BERT
            
            features.update({'embedding_for_adv':adv_embedding})
            adv_logits=self.model(**features)
            adv_loss=loss_fct(adv_logits,labels)

            loss_value=(adv_loss+normal_loss)/2

        return loss_value

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

        if self.pooling_type not in ['cls','last_layer']:
            output_all_encoded_layers=True
        if not os.path.exists(os.path.join(output_path,"label2id.json")):
            os.makedirs(output_path,exist_ok=True)
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
                        loss_value=self.getLoss(features=features,labels=labels,loss_fct=loss_fct)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss_value=self.getLoss(features=features,labels=labels,loss_fct=loss_fct)
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
        if self.pooling_type not in ['cls','last_layer']:
            output_all_encoded_layers=True

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