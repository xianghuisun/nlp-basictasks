import logging
import os
import csv
from typing import List
from sklearn import metrics
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from log import logging
from copy import deepcopy
#logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

'''
这个Evaluator的目的是通过黑盒测试的方法找出影响一个句子分类结果的字或者词
'''

class attackEvaluator:
    def __init__(self,model,is_pairs=False) -> None:
        self.model=model
        self.is_pairs=is_pairs
    
    def __call__(self, sentence,label,label2id,convert_to_numpy=True):
        '''
        prob should be tuple or list
        传入的sentences和labels一一对应，
        传出的数据格式为: [changed_sentence,changed_label,changed_prob,word]
        也就是给出每一个句子之前的标签和预测这个标签的概率，以及删除了word之后预测的结果
        '''
        tag_id=label2id[label]
        sentence_list=list(sentence)
        predict_result=[]
        for i in range(len(sentence_list)):
            sentence_list_=deepcopy(sentence_list)
            word=sentence_list_[i]
            del sentence_list_[i]
            input_sentence=''.join(sentence_list_)
            predict_probs=self.model.predict(is_pairs=self.is_pairs,dataloader=[input_sentence],convert_to_numpy=convert_to_numpy)[0]
            if np.argmax(predict_probs)!=tag_id and max(predict_probs)>=0.8:
                #说明此时预测的类别发生变化
                predict_result.append(input_sentence)
                predict_result.append(np.argmax(predict_probs))
                predict_result.append(predict_probs)
                predict_result.append(word)
                return predict_result

        #说明此时单个字的删除还不足以fool the model
        for i in range(len(sentence_list)-1):
            sentence_list_=deepcopy(sentence_list)
            word=''.join([sentence_list_[i],sentence_list_[i+1]])
            del sentence_list_[i]
            del sentence_list_[i]#删除一个后，这个id就是原来的下一个id
            input_sentence=''.join(sentence_list_)
            predict_probs=self.model.predict(is_pairs=self.is_pairs,dataloader=[input_sentence],convert_to_numpy=convert_to_numpy)[0]
            if np.argmax(predict_probs)!=tag_id and max(predict_probs)>=0.8:
                #说明此时预测的类别发生变化
                predict_result.append(input_sentence)
                predict_result.append(np.argmax(predict_probs))
                predict_result.append(predict_probs)
                predict_result.append(word)
                return predict_result
        
        return [sentence,tag_id]