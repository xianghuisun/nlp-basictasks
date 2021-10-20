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

from log import logging

logger=logging.getLogger(__name__)

class Trainer:
    def __init__(self,
                epochs,
                save_best_model=True,
                output_path=None,
                early_stop_patience = 10,
                tensorboard_logdir=None,
                max_grad_norm: float=1
                ):
        self.output_path=output_path
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)
        self.epochs=epochs
        self.save_best_model=save_best_model
        self.early_stop_patience=early_stop_patience
        if tensorboard_logdir!=None:
            self.tensorboard_writer=SummaryWriter(tensorboard_logdir)
        else:
            self.tensorboard_writer=None
        self.max_grad_norm=max_grad_norm

    def train(self,
              train_dataloader,
              model,
              optimizer,
              scheduler,
              evaluator=None,
              print_loss_step = None,
              evaluation_steps = None,
              show_progress_bar = None
              ):
        if print_loss_step==None:
            print_loss_step=len(train_dataloader)//5
        if evaluator is not None and evaluation_steps==None:
            evaluation_steps=len(train_dataloader)//2
        #一个epoch下打印5次loss，评估2次
        logger.info("一个epoch 下，每隔{}个step会输出一次loss，每隔{}个step会评估一次模型".format(print_loss_step,evaluation_steps))
        global_step=0
        skip_scheduler = False
        patience = self.early_stop_patience     
        best_score = -9999999
        num_train_steps = int(len(train_dataloader) * self.epochs)

        for epoch in trange(self.epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps=0
            training_loss=0.0
            model.zero_grad()
            model.train()

            for train_step,batch_inputs in tqdm(enumerate(train_dataloader)):
                loss=model(batch_inputs)
                training_loss+=loss.item()

                if print_loss_step!=None and train_step>0 and train_step%print_loss_step == 0:
                    training_loss/=print_loss_step
                    logging.info("Epoch : {}, train_step : {}/{}, loss_value : {} ".format(epoch,train_step*(epoch+1),num_train_steps,training_loss))
                    training_loss=0.0
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar(f"train_loss",loss.item(),global_step=global_step)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                optimizer.zero_grad()
                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluator is not None and evaluation_steps>0 and (train_step) % evaluation_steps == 0 :
                    eval_score=evaluator(model,output_path=self.output_path, epoch=epoch, steps=training_steps)
                    if self.tensorboard_writer is not None:
                        self.tensorboard_writer.add_scalar(f"eval_score",float(eval_score),global_step=global_step)
                    if eval_score > best_score:
                        if self.save_best_model:
                            model.save(output_path=self.output_path)
                            logging.info(("In epoch {}, training_step {}, the eval score is {}, previous eval score is {}, model has been saved in {}".format(epoch,train_step*(epoch+1),eval_score,best_score,self.output_path)))
                            best_score=eval_score
                    else:
                        patience-=1
                        logging.info(f"No improvement over previous best eval score ({eval_score:.6f} vs {best_score:.6f}), patience = {patience}")
                        if patience==0:
                            logging.info("Run our of patience, early stop!")
                            return
                
                    model.zero_grad()
                    model.train()

    # def _eval_during_training(self, evaluator, output_path,  epoch, steps):
    #     if evaluator is not None:
    #         score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
    #         return score
    #     return None
