import json,logging
from os import path
import torch
from typing import List,Optional

from .base_reader import convert_to_tensor

class InputExample:
    def __init__(self,guid='', seq_in: list=None, seq_out: list=None):
        '''
        为了避免混淆tokenize后用' '.join得到的str和普通的没有空格的str，强制传入了seq_in是list
        '''
        self.guid=guid
        assert type(seq_in)==list
        if seq_out is None:
            seq_out=['O']*len(seq_in)
        assert type(seq_out)==list
        assert len(seq_in)==len(seq_out)
        self.seq_in=seq_in
        self.seq_out=seq_out
    def __str__(self) -> str:
        return "<InputExample> seq_in: {}\n seq_out: {}\n".format(' '.join(self.seq_in),' '.join(self.seq_out))

class InputFeatures:
    def __init__(self, 
                input_ids=[101,2587,2587,2587,102],
                token_type_ids=[0,0,0,0,0],
                attention_mask=[1,1,1,1,1],
                label_ids=[0,0,0,0,0]):
        self.input_ids=input_ids
        self.token_type_ids=token_type_ids
        self.attention_mask=attention_mask
        self.label_ids=label_ids
    def __str__(self) -> str:
        return "<InputFeature> input_ids: {}\n token_type_ids: {}\n attention_mask: {}\n label_ids: {}\n".format(' '.join([str(x) for x in self.input_ids]),
                                                                                                                ' '.join([str(x) for x in self.token_type_ids]),
                                                                                                                ' '.join([str(x) for x in self.attention_mask]),
                                                                                                                ' '.join([str(x) for x in self.label_ids]))


def generate_SeqIn_and_SeqOut_based_Examples(example,tokenizer):
    '''
    example形如
    {"text": "五一劳动节是农历几时", "intent": "Calendar-Query", "slots": {"datetime_date": "五一劳动节"}}
    {"text": "快帮我查一下明天是星期几", "intent": "Calendar-Query", "slots": {"datetime_date": "明天"}}
    {"text": "看一下2019年有哪些好看的动作片", "intent": "Film-Query", "slots": {"year": "2019年"}}
    '''
    text=example["text"]
    slots=example["slots"]
    seq_in=tokenizer.tokenize(text)
    seq_out=['O' for _ in range(len(seq_in))]
    assert len(seq_out)==len(seq_in)
    need_print=False
    for slot_name,slot_values in slots.items():
        if type(slot_values)==str:
            slot_values=[slot_values]
        multiple_slots_in_text=False
        multiple_slot_begin_pos=[]
        multiple_slot_end_pos=[]
        for slot_value in slot_values:
            slot_value_tokens=tokenizer.tokenize(slot_value)
            slot_value_length=len(slot_value_tokens)
            slot_begin_pos=-1
            slot_end_pos=-1
            if len(text)==len(seq_in):
                #没有数字和英文字母出现
                assert slot_value_length==len(slot_value)
                slot_begin_pos=text.find(slot_value)
                if slot_begin_pos!=-1:
                    slot_end_pos=slot_begin_pos+slot_value_length
                    if text.count(slot_value)>1:
                        #也就是说除去slot_begin_pos位置后面还有当前的slot_value
                        new_text=text[slot_end_pos:]
                        new_pointer=slot_end_pos
                        while new_text.find(slot_value)!=-1:
                            #print(new_text,new_text.find(slot_value)+slot_value_length+new_pointer,new_pointer)
                            multiple_slot_begin_pos.append(new_text.find(slot_value)+new_pointer)
                            multiple_slot_end_pos.append(new_text.find(slot_value)+slot_value_length+new_pointer)
                            new_text=text[new_text.find(slot_value)+slot_value_length+new_pointer:]
                            new_pointer=text.find(new_text)
                            multiple_slots_in_text=True
            else:
                #有数字或者英文字母甚至是空格的出现导致tokenize后的长度不一致
                #text=看一下2019年有哪些好看的动作片, seq_in=看 一 下 2019 年 有 哪 些 好 看 的 动 作 片
                slot_begin_char=slot_value_tokens[0]
                for i,token in enumerate(seq_in):
                    if token==slot_begin_char:
                        char=''
                        if i+slot_value_length>len(seq_in):
                            print("槽值在句子中的开始位置是%d，槽值的长度是%d，而句子的长度是%d，槽位中槽值的结束位置超出了句子的长度"%(i,slot_value_length,len(seq_in)))
                            print("出现的问题的example是 ",example)
                            break
                        for j in range(i,i+slot_value_length):
                            if seq_in[j][:2]=='##':
                                char+=seq_in[j][2:]#出现英文单词
                            else:
                                char+=seq_in[j]
                        if tokenizer.do_lower_case:
                            slot_value=slot_value.lower()#大小写问题
                        if char==slot_value:
                            slot_begin_pos=i
                            slot_end_pos=i+slot_value_length
                            break
            
            if slot_begin_pos!=-1:
                seq_out[slot_begin_pos]='B-'+slot_name
                for pos in range(slot_begin_pos+1,slot_end_pos):
                    seq_out[pos]='I-'+slot_name
                    
        if multiple_slots_in_text:
            for start_pos,end_pos in zip(multiple_slot_begin_pos,multiple_slot_end_pos):
                seq_out[start_pos]='B-'+slot_name
                for pos in range(start_pos+1,end_pos):
                    seq_out[pos]='I-'+slot_name
            # print("text中出现了多个槽值，标注后的形式为:")
            # print(seq_in)
            # print(seq_out)
            # print(example)
            # print('-'*100)
            need_print=True
    if need_print:
        print(seq_in)
        print(seq_out)
        print(example)
        print('-'*100)
    return seq_in,seq_out

def write_seqIn_and_seqOut_from_original_file(seq_in_path,seq_out_path,file_path,tokenizer):
    examples=[]
    with open(file_path) as f:
        lines=f.readlines()
    for line in lines:
        examples.append(json.loads(line.strip()))
    
    f_seq_in=open(seq_in_path,"w",encoding='utf-8')
    f_seq_out=open(seq_out_path,"w",encoding="utf-8")

    for example in examples:
        seq_in,seq_out=generate_SeqIn_and_SeqOut_based_Examples(example,tokenizer)
        assert len(seq_in)==len(seq_out)
        f_seq_in.write(' '.join(seq_in)+"\n")
        f_seq_out.write(' '.join(seq_out)+'\n')
    f_seq_in.close()
    f_seq_out.close()


def getSeqIn_and_SeqOut(seq_in_path,seq_out_path,intent_path=None):
    def readData(path):
        data=[]
        with open(path,encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                data.append(line.strip().split(' '))
        return data

    seq_in=readData(path=seq_in_path)
    seq_out=readData(path=seq_out_path)
    if intent_path!=None:
        intents=readData(path=intent_path)
        return seq_in,seq_out,intents
    else:
        return seq_in,seq_out
    

def readExample(seq_in_list,seq_out_list=None):
    '''
    传进来的seq_in_list和seq_out_list形如
    ['五', '一', '劳', '动', '节', '是', '农', '历', '几', '时']
    ['B-datetime_date', 'I-datetime_date', 'I-datetime_date', 'I-datetime_date', 'I-datetime_date', 'O', 'O', 'O', 'O', 'O']
    '''
    assert type(seq_in_list)==type(seq_out_list)==list
    examples=[]
    if seq_out_list is not None:
        if type(seq_in_list[0])==list:
            assert type(seq_out_list[0])==list and len(seq_in_list[0])==len(seq_out_list[0])
        else:
            assert len(seq_in_list[0].split(' '))==len(seq_out_list[0].split(' '))

        for seq_in,seq_out in zip(seq_in_list,seq_out_list):
            examples.append(InputExample(seq_in=seq_in,seq_out=seq_out))
    else:
        for seq_in in seq_in_list:
            if type(seq_in)==list:
                seq_out=['O']*len(seq_in)
            else:
                assert isinstance(seq_in)==str
                seq_out=' '.join(['O']*len(seq_in.split(' ')))
            examples.append(InputExample(seq_in=seq_in,seq_out=seq_out))
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_len, label2id):
    '''
    label_ids在CLS、SEP、PAD位置上都是0，代表这些位置都是一个tag
    attention_mask仍然是考虑CLS和SEP的

    example的seq_in和seq_out都是列表
    '''
    features = []
    labels=[]
    special_token_nums=2#[CLS],[SEP]
    assert isinstance(examples[0],InputExample)#强制要求传进来的examples的element必须是InputExample类型

    max_len_this_batch=0
    for example in examples:
        length=len(example.seq_in)
        if length>max_len_this_batch:
            max_len_this_batch=length

    max_seq_len=min(max_seq_len,max_len_this_batch+special_token_nums)

    for example_index, example in enumerate(examples):
        tokens = []
        label_ids = []
        token_type_ids = []

        tokens.append("[CLS]")
        token_type_ids.append(0)
        label_ids.append(label2id["[PAD]"])#需要特别注意，只需要让CLS,SEP,PAD当作一个类别预测即可
        #没必要让[PAD]和[SEP]也作为类别，而且label2id['[PAD]']必须是0
        
        sentence_tokens = example.seq_in
        label_list = example.seq_out
        
        assert len(sentence_tokens)==len(label_list)
        if len(sentence_tokens)>=max_seq_len-special_token_nums:
            sentence_tokens=sentence_tokens[:max_seq_len-special_token_nums]
            label_list=label_list[:max_seq_len-special_token_nums]

        tokens += sentence_tokens
        token_type_ids += [0] * len(sentence_tokens)
        label_ids += [label2id[label] for label in label_list]

        tokens.append('[SEP]')
        token_type_ids.append(0)
        label_ids.append(label2id["[PAD]"])#只需要用一个tag来代表这三个特殊字符即可
        ##没必要让[PAD]和[SEP]也作为类别，而且label2id['[PAD]']必须是0

        seq_len = len(tokens)
        pad_len = max_seq_len - seq_len
        # print(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids += [0] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
        token_type_ids += [0] * pad_len
        label_ids += [label2id["[PAD]"]] * pad_len

        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == max_seq_len == len(label_ids)

        features.append(InputFeatures(input_ids=input_ids, 
                                    token_type_ids=token_type_ids, 
                                    attention_mask=attention_mask,
                                    label_ids=label_ids))
        # if example_index < 5:
        #     logging.info("*** Example ***")
        #     logging.info("example_index: %s" % (example_index))
        #     logging.info("tokens: %s" % " ".join(tokens))
        #     logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logging.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logging.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     logging.info("label_ids : %s" % " ".join([str(example.label)]))
    
    features=convert_to_tensor(features=features)
    return features
