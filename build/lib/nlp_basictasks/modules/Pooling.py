import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os,sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from log import logging
logger=logging.getLogger(__name__)

class Pooling(nn.Module):
    """
    Pooling层的操作就是根据指定的条件，如(cls,max,mean,first_last,last_2等)对token_embeddings进行相应的pooling操作
    然后拼接每一个pooling操作的result，update到features["sentence_embedding"]输出features。
    这个sentence_embedding的维度显然是768*pooling操作的个数
    self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)会将Pooling输出的向量维度返回给
    SentenceTransformer的model.get_pooling_dimension，然后这个值回传给Loss模型，这样就可以知道在loss中知道sentence_embedding的维度了
    loss中的条件是concatenation_sent_rep和difference以及multiplication，然后再计算loss
    也就是说，假如pooling方式选择mean+last-first，那么句子向量维度就是768*2。loss中拼接的条件是rep+difference+multiplication
    那么此时传进给loss的维度是(768*2*2)+(768*2)+(768*2)
    """

    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 pooling_mode_mean_last_2_tokens: bool = False,
                 pooling_mode_mean_first_last_tokens: bool = False,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'pooling_mode_cls_token', 'pooling_mode_mean_tokens',
                            'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens',
                            'pooling_mode_mean_last_2_tokens','pooling_mode_mean_first_last_tokens']

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_mean_last_2_tokens = pooling_mode_mean_last_2_tokens
        self.pooling_mode_mean_first_last_tokens = pooling_mode_mean_first_last_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens,
                                       pooling_mode_mean_sqrt_len_tokens, pooling_mode_mean_last_2_tokens,pooling_mode_mean_first_last_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)
        logger.info("Pooling config : {}".format(self.get_config_dict()))
        logger.info("Pooling output dimension is {}".format(self.pooling_output_dimension))

    def forward(self, token_embeddings, cls_token_embeddings, attention_mask, all_layer_embeddings=None, token_weights_sum=None):
        '''
        token_embeddings就是sequence_output，注意sequence_output是包含有cls的
        cls_token_embeddings就是pooled_output
        '''
        #assert features["all_layer_embeddings"][-1].sum() == features["token_embeddings"].sum()

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token_embeddings)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if token_weights_sum is not None:
                sum_mask = token_weights_sum.unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        if all_layer_embeddings is not None and self.pooling_mode_mean_last_2_tokens:  # avg of last 2 layers
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            token_embeddings_last1 = all_layer_embeddings[-1]
            sum_embeddings_last1 = torch.sum(token_embeddings_last1 * input_mask_expanded, 1)
            sum_embeddings_last1 = sum_embeddings_last1 / sum_mask

            token_embeddings_last2 = all_layer_embeddings["all_layer_embeddings"][-2]
            sum_embeddings_last2 = torch.sum(token_embeddings_last2 * input_mask_expanded, 1)
            sum_embeddings_last2 = sum_embeddings_last2 / sum_mask

            output_vectors.append((sum_embeddings_last1 + sum_embeddings_last2) / 2)

        if all_layer_embeddings is not None and self.pooling_mode_mean_first_last_tokens:  # avg of the first and the last layers
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            token_embeddings_first = all_layer_embeddings[0]
            sum_embeddings_first = torch.sum(token_embeddings_first * input_mask_expanded, 1)
            sum_embeddings_first = sum_embeddings_first / sum_mask

            token_embeddings_last = all_layer_embeddings[-1]
            sum_embeddings_last = torch.sum(token_embeddings_last * input_mask_expanded, 1)
            sum_embeddings_last = sum_embeddings_last / sum_mask

            output_vectors.append((sum_embeddings_first + sum_embeddings_last) / 2)

        output_vector = torch.cat(output_vectors, 1)
        assert output_vector.size(1)==self.pooling_output_dimension
        return output_vector

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        os.makedirs(output_path,exist_ok=True)
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)
        torch.save(self.state_dict(),os.path.join(output_path, 'pytorch_model.bin'))
        
    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)
        logger.info("Pooling config : ",config)
        return Pooling(**config)