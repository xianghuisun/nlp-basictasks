import torch
import jieba
import random

def convert_to_tensor(features):
    features_tensor={key:[] for key in features[0].__dict__.keys()}
    for feature in features:
        for key,value in feature.__dict__.items():
            features_tensor[key].append(value)
    features_tensor={key:torch.LongTensor(value) for key,value in features_tensor.items()}
    return features_tensor

def ShuffleAndCutOff(sentence,need_jieba=True,cutoff_rate=0.0,use_dict_path=None):
    if need_jieba:
        if use_dict_path != None:
            jieba.load_userdict(use_dict_path)
        sentence=list(jieba.cut(sentence))
    
    sentence_len=len(sentence)
    index=list(range(sentence_len))
    random.shuffle(index)
    shuffled_sentence=[sentence[i] for i in index]

    cut_nums=int(len(shuffled_sentence)*cutoff_rate)
    cut_pos=random.sample(range(sentence_len),k=cut_nums)

    cutoff_sentence=[shuffled_sentence[i] for i in range(sentence_len) if i not in cut_pos]
    return cutoff_sentence