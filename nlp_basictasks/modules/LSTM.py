import torch
from torch import Tensor
from torch import nn
from torch import functional as F
from typing import Union, Tuple, List, Iterable, Dict
import os,sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from log import logging
logger=logging.getLogger(__name__)

class LSTMLayer(nn.Module):
    def __init__(self, input_size, 
                        hidden_size, 
                        num_layers, 
                        bias=True, 
                        batch_first=True, 
                        dropout=0.0,
                        bidirectional=True):
        super(LSTMLayer,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.bias = bias
        self.batch_first=batch_first
        self.dropout=dropout
        self.bidirectional=bidirectional
        self.lstmLayer=nn.LSTM(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                bias=self.bias,
                                batch_first=self.batch_first,
                                dropout=self.dropout,
                                bidirectional=self.bidirectional)

    def forward(self, sequence_output, attention_mask):
        input_lengths=attention_mask.sum(1)
        packed_inputs_of_lstm=torch.nn.utils.rnn.pack_padded_sequence(input=sequence_output,lengths=input_lengths,batch_first=self.batch_first,enforce_sorted=False)
        sequence_output,_=self.lstmLayer(packed_inputs_of_lstm)
        sequence_output,_=torch.nn.utils.rnn.pad_packed_sequence(sequence_output,batch_first=self.batch_first)
        #sequence_output.size()==(bsa,max_seq_len,hidden_size*2)
        return sequence_output

    def get_config_dict(self):
        '''
        一定要有dict，这样才能初始化Model
        '''
        return {'input_size': self.input_size, 
                'hidden_size': self.hidden_size, 
                'bias': self.bias,
                'num_layers':self.num_layers,
                'batch_first':self.batch_first,
                'dropout':self.dropout,
                'bidirectional':self.bidirectional}

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

        model = LSTMLayer(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model