import torch
from torch import Tensor
from torch import nn
from torch import functional as F
from typing import Union, Tuple, List, Iterable, Dict
import os,sys
import json

from .utils import fullname,import_from_string
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from log import logging
logger=logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, in_features, 
                        out_features, 
                        bias = True, 
                        activation_function=nn.Tanh(), 
                        init_weight: Tensor = None, 
                        init_bias: Tensor = None):
        super(MLP,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = activation_function
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)

    def forward(self, features):
        '''
        The output shape is like features.shape except last dim
        '''
        return self.linear(features)

    def get_config_dict(self):
        '''
        一定要有dict，这样才能初始化Model
        '''
        return {'in_features': self.in_features, 'out_features': self.out_features, 'bias': self.bias, 'activation_function': fullname(self.activation_function)}

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

        config['activation_function'] = import_from_string(config['activation_function'])()
        #上一行是因为激活函数是函数，不是整数，所以要给出位置
        model = MLP(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model