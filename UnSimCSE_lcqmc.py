import nlp_basictasks
import os,json
import numpy as np
import torch
import torch.nn as nn
import random
from tqdm.autonotebook import tqdm, trange
from torch.utils.data import DataLoader
from nlp_basictasks.modules import SBERT
from nlp_basictasks.modules.transformers import BertTokenizer
from nlp_basictasks.readers.sts import InputExample,convert_examples_to_features,getExamples,convert_sentences_to_features
from nlp_basictasks.modules.utils import get_optimizer,get_scheduler
from nlp_basictasks.Trainer import Trainer
from nlp_basictasks.evaluation import stsEvaluator
from sentence_transformers import SentenceTransformer,models
model_path1='/data/nfs14/nfs/aisearch/asr/xhsun/bwbd_recall/distill-simcse/'
model_path2="/data/nfs14/nfs/aisearch/asr/xhsun/bwbd_recall/distiluse-base-multilingual-cased-v1/"
model_path3='/data/nfs14/nfs/aisearch/asr/xhsun/CommonModel/chinese-roberta-wwm/'
data_folder='/data/nfs14/nfs/aisearch/asr/xhsun/datasets/lcqmc/'
train_file=os.path.join(data_folder,'lcqmc_train.tsv')
dev_file=os.path.join(data_folder,'lcqmc_dev.tsv')
# data_folder='/data/nfs14/nfs/aisearch/asr/xhsun/bwbd_recall/ssc_data/sbert_ce_portion_5'
# train_file=os.path.join(data_folder,'train.txt')
# dev_file=os.path.join(data_folder,'dev.txt')
# tokenizer=BertTokenizer.from_pretrained(os.path.join(model_path1,'0_Transformer'))

class SimCSE(nn.Module):
    def __init__(self,
                 bert_model_path,
                 is_sbert_model=True,
                temperature=0.05,
                is_distilbert=False,
                device='cpu'):
        super(SimCSE,self).__init__()
        if is_sbert_model:
            self.encoder=SentenceTransformer(model_name_or_path=bert_model_path,device=device)
        else:
            word_embedding_model = models.Transformer(bert_model_path, max_seq_length=max_seq_len)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            self.encoder=SentenceTransformer(modules=[word_embedding_model, pooling_model],device=device)
        self.temperature=temperature
        self.is_distilbert=is_distilbert#蒸馏版本的BERT不支持token_type_ids
    def cal_cos_sim(self,embeddings1,embeddings2):
        embeddings1_norm=torch.nn.functional.normalize(embeddings1,p=2,dim=1)
        embeddings2_norm=torch.nn.functional.normalize(embeddings2,p=2,dim=1)
        return torch.mm(embeddings1_norm,embeddings2_norm.transpose(0,1))#(batch_size,batch_size)
        
    def forward(self,batch_inputs):
        '''
        为了实现兼容，所有model的batch_inputs最后一个位置必须是labels，即使为None
        get token_embeddings,cls_token_embeddings,sentence_embeddings
        sentence_embeddings是经过Pooling层后concat的embedding。维度=768*k，其中k取决于pooling的策略
        一般来讲，只会取一种pooling策略，要么直接cls要么mean last or mean last2 or mean first and last layer，所以sentence_embeddings的维度也是768
        '''
        batch1_features,batch2_features,_=batch_inputs
        if self.is_distilbert:
            del batch1_features['token_type_ids']
            del batch2_features['token_type_ids']
        batch1_embeddings=self.encoder(batch1_features)['sentence_embedding']
        batch2_embeddings=self.encoder(batch2_features)['sentence_embedding']
        cos_sim=self.cal_cos_sim(batch1_embeddings,batch2_embeddings)/self.temperature#(batch_size,batch_size)
        batch_size=cos_sim.size(0)
        assert cos_sim.size()==(batch_size,batch_size)
        labels=torch.arange(batch_size).to(cos_sim.device)
        return nn.CrossEntropyLoss()(cos_sim,labels)
    
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
        return self.encoder.encode(sentences=sentences,
                                         batch_size=batch_size,
                                         show_progress_bar=show_progress_bar,
                                         output_value=output_value,
                                         convert_to_numpy=convert_to_numpy,
                                         convert_to_tensor=convert_to_tensor,
                                         device=device,
                                         normalize_embeddings=normalize_embeddings)
    
    def save(self,output_path):
        os.makedirs(output_path,exist_ok=True)
        with open(os.path.join(output_path, 'model_param_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(output_path), fOut)
        self.encoder.save(output_path)
        
    def get_config_dict(self,output_path):
        '''
        一定要有dict，这样才能初始化Model
        '''
        return {'output_path':output_path,'temperature': self.temperature, 'is_distilbert': self.is_distilbert}
    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'model_param_config.json')) as fIn:
            config = json.load(fIn)
        return SimCSE(**config)

tokenizer=BertTokenizer.from_pretrained(model_path3)
max_seq_len=64
batch_size=128
device='cuda'
epochs=5
optimizer_type='AdamW'
scheduler='WarmupLinear'
warmup_proportion=0.1
optimizer_params={'lr': 2e-5}
weight_decay=0.01
output_path='/data/nfs14/nfs/aisearch/asr/xhsun/bwbd_recall/unSimCSE_lcqmc'
tensorboard_logdir=os.path.join(output_path,'log')

#create model
simcse=SimCSE(bert_model_path=model_path3,is_distilbert=False,device=device,is_sbert_model=False)

#create dataloader
train_sentences=[]
with open(train_file) as f:
    lines=f.readlines()
    for line in lines[1:]:
        line_split=line.strip().split('\t')
        train_sentences.append(line_split[0])
print(len(train_sentences))
print(train_sentences[:2])
#random.shuffle(train_sentences)
train_sentences=train_sentences[:12000]
print(len(train_sentences))
print(train_sentences[:2])
train_examples=[InputExample(text_list=[sentence,sentence],label=1) for sentence in train_sentences]
train_dataloader=DataLoader(train_examples,shuffle=True,batch_size=batch_size)

def batch_to_device(batch,target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch
def smart_batching_collate(batch):
    features_of_a,features_of_b,labels=convert_examples_to_features(examples=batch,tokenizer=tokenizer,max_seq_len=max_seq_len)
    features_of_a=batch_to_device(features_of_a,target_device=torch.device(device))
    features_of_b=batch_to_device(features_of_b,target_device=torch.device(device))
    labels=labels.to(torch.device(device))
    return features_of_a,features_of_b,labels
train_dataloader.collate_fn=smart_batching_collate
print(train_examples[0])

#create evaluator
dev_examples=getExamples(dev_file,label2id={"0":0,"1":1},filter_heads=True,mode='dev',isCL=False)
dev_sentences=[example.text_list for example in dev_examples]
dev_labels=[example.label for example in dev_examples]
print(dev_sentences[0],dev_labels[0])
sentences1_list=[sen[0] for sen in dev_sentences]
sentences2_list=[sen[1] for sen in dev_sentences]
evaluator=stsEvaluator(sentences1=sentences1_list,sentences2=sentences2_list,batch_size=64,write_csv=True,scores=dev_labels)
for i in range(5):
    print(evaluator.sentences1[i],evaluator.sentences2[i],evaluator.scores[i])

evaluator(model=simcse)

# train model
num_train_steps = int(len(train_dataloader) * epochs)
warmup_steps = num_train_steps*warmup_proportion
optimizer = get_optimizer(model=simcse,optimizer_type=optimizer_type,weight_decay=weight_decay,optimizer_params=optimizer_params)
scheduler = get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

trainer=Trainer(epochs=epochs,output_path=output_path,tensorboard_logdir=tensorboard_logdir,early_stop_patience=20)
trainer.train(train_dataloader=train_dataloader,
             model=simcse,
             optimizer=optimizer,
             scheduler=scheduler,
             evaluator=evaluator,
             )