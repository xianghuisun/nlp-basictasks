from typing import Text, Union, List
import numpy as np
import random,sys,os,json
import torch
import collections
from tqdm.autonotebook import tqdm,trange


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from log import logging
logger=logging.getLogger(__name__)

from .base_reader import convert_to_tensor,is_chinese_char,is_fuhao,tokenize_chinese_chars,is_whitespace,SPIECE_UNDERLINE

SPECIAL_TOKEN_NUMS=3#[CLS] and [SEP], [SEP]

'''
阅读理解SQUAD数据集格式形如：
{
    'context_id': 'TRAIN_186', 
    'context_text': '范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生；童年时接受良好教育后，被一位越南神父带到河内继续其学业。范廷颂于1940年在河内大修道院完成神学学业。范廷颂于1949年6月6日在河内的主教座堂晋铎；及后被派到圣女小德兰孤儿院服务。1950年代，范廷颂在河内堂区创建移民接待中心以收容到河内避战的难民。1954年，法越战争结束，越南民主共和国建都河内，当时很多天主教神职人员逃至越南的南方，但范廷颂仍然留在河内。翌年管理圣若望小修院；惟在1960年因捍卫修院的自由、自治及拒绝政府在修院设政治课的要求而被捕。1963年4月5日，教宗任命范廷颂为天主教北宁教区主教，同年8月15日就任；其牧铭为「我信天主的爱」。由于范廷颂被越南政府软禁差不多30年，因此他无法到所属堂区进行牧灵工作而专注研读等工作。范廷颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会团体等。1990年，教宗若望保禄二世在同年6月18日擢升范廷颂为天主教河内总教区宗座署理以填补该教区总主教的空缺。1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理；同年11月26日，若望保禄二世擢升范廷颂为枢机。范廷颂在1995年至2001年期间出任天主教越南主教团主席。2003年4月26日，教宗若望保禄二世任命天主教谅山教区兼天主教高平教区吴光杰主教为天主教河内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴光杰同日真除天主教河内总教区总主教职务。范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。', 
    'qas': [
                {'query_id': 'TRAIN_186_QUERY_0', 'query_text': '范廷颂是什么时候被任为主教的？', 'answers': ['1963年']}, 
                {'query_id': 'TRAIN_186_QUERY_1', 'query_text': '1990年，范廷颂担任什么职务？', 'answers': ['1990年被擢升为天主教河内总教区宗座署理']}, 
                {'query_id': 'TRAIN_186_QUERY_2', 'query_text': '范廷颂是于何时何地出生的？', 'answers': ['范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生']}, 
                {'query_id': 'TRAIN_186_QUERY_3', 'query_text': '1994年3月，范廷颂担任什么职务？', 'answers': ['1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理']}, 
                {'query_id': 'TRAIN_186_QUERY_4', 'query_text': '范廷颂是何时去世的？', 'answers': ['范廷颂于2009年2月22日清晨在河内离世']}
            ],
    'title': '范廷颂'
}
'''

class InputExample:
    def __init__(self,guid='',question='',context='',answer='',context_tokens=[],start_position=0,end_position=0):
        '''
        为了确保不出错，question、context、answer要么全是string类型，要么全是tokenize后的list类型
        由于数字空格原因可能导致context.find(answer)对不上，所以要有context_tokes，它是形如tokenizer.tokenize(context)后的结果
        '''
        assert type(question)==type(context)==type(answer)
        self.guid=guid
        self.question=question
        self.context=context
        self.context_tokens=context_tokens
        self.answer=answer
        self.start_position=start_position
        self.end_position=end_position
    
    def __str__(self) -> str:
        return '\n'+'question : '+self.question+'\n'+'context : '+self.context+'\n'+'answer : '+\
                self.answer+'\n'+"start_position : "+str(self.start_position)+"\n"+"end_position : "+\
                str(self.end_position)+'\n'+\
                'context_tokens[{}:{}+1] is {}'.format(self.start_position,self.end_position,''.join(self.context_tokens[self.start_position:self.end_position+1]))+'\n'

class InputFeature:
    def __init__(self, unique_id=0,
                    example_index=0,
                    doc_span_index=0,
                    tokens=[],
                    token_to_orig_map={},
                    token_is_max_context={},
                    input_ids=[],
                    attention_mask=[],
                    token_type_ids=[],
                    start_position=0,
                    end_position=0) -> None:
        '''
        unique_id代表这个feature的id
        example_index代表这个特征是属于examples中的第几个example
        context_span_index代表当前的feature是example经过截断后的第几个span
        token_to_orig_map保存的是一个dict，其中key是当前input_ids中从context开始计数，value对应的是当前context_tokens(maybe split)中的token对应于原始的context_tokens中的
        的第几个token，具体来说：一个example可能由于文本太长被分成三部分(doc_stride=256)，那么
        第一部分形如：example_index : 0,context_span_index : 0,token_to_orig_map : {17: 0, 18: 1, 19: 2, 20: 3, 21: 4, 22: 5, 23: 6, 24: 7...}
        第二部分形如：example_index : 0,context_span_index : 1,token_to_orig_map : {17: 256, 18: 257, 19: 258, 20: 259, 21: 260, 22: 261...}
        第三部分形如：example_index : 0,context_span_index : 2,token_to_orig_map : {17: 512, 18: 513, 19: 514, 20: 515, 21: 516, 22: 517, ...}
        每一部分的query_tokens是相同的，其余的变量都不同
        '''
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_position = start_position
        self.end_position = end_position
    
    def __str__(self) -> str:
        return '\n'+"unique_id : "+str(self.unique_id)+\
               '\n'+"example_index : "+str(self.example_index)+\
               '\n'+'context_span_index : '+str(self.doc_span_index)+\
                '\n'+'context tokens(maybe split) : '+' '.join(self.tokens)+\
                '\n'+'token_to_orig_map : '+str(self.token_to_orig_map)+\
                '\n'+'token_is_max_context : '+str(self.token_is_max_context)+\
                '\n'+'input ids : '+str(self.input_ids)+\
                '\n'+'input_mask : '+str(self.input_mask)+\
                '\n'+"token_type_ids : "+str(self.token_type_ids)+\
                '\n'+"start position : "+str(self.start_position)+\
                '\n'+'end position : '+str(self.end_position)+\
                '\n'+"self.tokens[{}:{}+1] is {}".format(self.start_position,self.end_position,''.join(self.tokens[self.start_position:self.end_position+1]))
        
def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def read_squad_example(input_file):
    '''
    context='范廷颂是越南罗马天主教枢机。1963年被任为主教'
    context_chs='范▁廷▁颂▁是▁越▁南▁罗▁马▁天▁主▁教▁枢▁机▁。▁1963▁年▁被▁任▁为▁主▁教▁'
    char_to_word_offset=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 14, 15, 16, 17, 18, 19, 20]
    可以看到有4个14。
    doc_tokens=['范', '廷', '颂', '是', '越', '南', '罗', '马', '天', '主', '教', '枢', '机', '。', '1963', '年', '被', '任', '为', '主', '教']
    '''
    with open(input_file) as f:
        data=json.load(f)
    
    logger.info("The total amount of data is {}".format(len(data)))
    examples=[]
    for each_example in data:
        context=each_example['context_text']
        qas=each_example['qas']
        context_chs=tokenize_chinese_chars(context)#形如'范▁廷▁颂▁枢▁机▁（▁，▁）▁，▁圣▁名▁保▁禄▁·▁若▁瑟▁（▁）▁，▁是▁越▁南▁罗▁马▁天▁主▁教▁枢▁机▁。▁1963▁年▁被▁任▁为▁主▁教▁'
        doc_tokens=[]
        char_to_word_offset=[]#char_to_word_offset顾名思义就是char对应的word的偏移
        #中文没有子词，所以大多数case下char与word一一对应，但是当遇到数字就不会一一对应了
        prev_is_whitespace = True

        for c in context_chs:
            if is_whitespace(c):
                #if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE 用来代表字与字之间的分割
                prev_is_whitespace=True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1]+=c#比如说1963这个case，context_chs中形如_1963_,所以1963回被看成一个token加入到doc_tokens
                prev_is_whitespace=False
            if c!=SPIECE_UNDERLINE:
                char_to_word_offset.append(len(doc_tokens)-1)
        
        for qa in qas:
            question=qa['query_text']
            answer=qa['answers'][0]#默认只有一个答案
            answer=str(answer)
            answer_index=context.find(answer)
            answer_length=len(answer)
            start_position=char_to_word_offset[answer_index] if answer_index!=-1 else -1
            end_position=char_to_word_offset[answer_index+answer_length-1] if start_position!=-1 else -1
            examples.append(InputExample(question=question,
                                        context=context,
                                        context_tokens=doc_tokens,
                                        answer=answer,
                                        start_position=start_position,
                                        end_position=end_position))
    logger.info("The total amount of examples is {}".format(len(examples)))
    return examples

'''
One example like :
question : 大动脉会分支出什么？
context : 体循环（又称为大循环）是心血管循环系统中，携带充氧血离开心脏，进入身体各部位进行气体交换及运输养分后，将缺氧血带回心脏的部分。相对于体循环的另一种血液循环则称为肺循环（又称为小循环）。其循环式如下：左心室→主动脉→小动脉→组织微血管→小静脉→大静脉（上、下腔静脉）→右心房先由左心室将从肺静脉送回心脏充满营养和氧气的充氧血从大（主）动脉输出至身体各部位组织的微血管进行养分的运输以及气体的交换。由大动脉渐分支出小动脉，再分支出为微血管。在微血管中，血液中的养分以及氧气分子会送至组织细胞中，组织细胞中的二氧化碳分子以及废物则会送至血液中。接下来再将完成交换及运输的减氧血经由上下大静脉送回右心房，而继续进行肺循环。
answer : 由大动脉渐分支出小动脉，再分支出为微血管。
start_position : 197
end_position : 217
context_tokens[197:217+1] is 由大动脉渐分支出小动脉，再分支出为微血管。
'''

def convert_examples_to_features(examples,tokenizer,max_seq_length,max_query_length=64,doc_stride=394,is_training=True,show_progress_bar=True):
    '''
    context='The case written in 2021年用来展示MRC的处理流程'
    context_tokens=['The', 'case', 'written', 'in', '2021', '年', '用', '来', '展', '示', 'MRC', '的', '处', '理', '流', '程']
    context_tokens_after_tokenized=['the', 'case', 'w', '##ri', '##tte', '##n', 'in', '2021', '年', '用', '来', '展', '示', 'mr', '##c', '的', '处', '理', '流', '程']
    orig_to_tok_index记录的是context_tokens中的每一个token在context_tokens_after_tokenized中位置，所以
    orig_to_tok_index=[0,1,2,6,7,8,9,10,11,12,13,15,16,17,18,19]
    tok_to_orig_index记录的是context_tokens_after_tokenized中的每一个subtoken在原来的context_tokens中的位置，所以
    tok_to_orig_index=[0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15]

    可以看出tok_to_orig_index保存的是subtoken对应token的关系，而orig_to_tok_index保存的是token对应subtoken的关系
    '''
    features=[]
    unique_id=10000
    for example_index in trange(len(examples), desc="Evaluating", disable=not show_progress_bar):
        example=examples[example_index]
        query_tokens=tokenizer.tokenize(example.question)
        if len(query_tokens)>max_query_length:
            query_tokens=query_tokens[:max_query_length]
        
        context_tokens_after_tokenized=[]#context_tokens并没有用tokenize，所以tokenize后可能会出现子词
        #所以context_tokens_after_tokenized存储的就是真正要输入给模型的句子，之前的context、context_tokens都不是
        orig_to_tok_index=[]#这个变量保存的是每一个token对应于context_tokens_after_tokenized的位置
        #我们以上面的例子为例，假如答案是2021年，那么context_tokens中start_position=4，orig_to_tok_index[start_position]=7
        #而context_tokens_after_tokenized[7]正是2021，所以也就得到了tokenized后答案的起始位置
        tok_to_orig_index=[]#这个变量保存的是每一个subtoken对应于context_tokens中的第几个单词

        for token_idx,token in enumerate(example.context_tokens):
            orig_to_tok_index.append(len(context_tokens_after_tokenized))
            sub_tokens=tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                context_tokens_after_tokenized.append(sub_token)
                tok_to_orig_index.append(token_idx)
        
        assert len(tok_to_orig_index)==len(context_tokens_after_tokenized) and len(orig_to_tok_index)==len(example.context_tokens)
        #如果不出现子词，也就是没有英文字母出现，那么context_tokens和context_tokens_after_tokenized是一样的，此时start and end position也不会变
        tok_start_position=None
        tok_end_position=None

        if is_training:
            tok_start_position=orig_to_tok_index[example.start_position] if example.start_position!=-1 else -1
            if example.end_position<len(example.context_tokens)-1:
                tok_end_position=orig_to_tok_index[example.end_position+1]-1 if tok_start_position!=-1 else -1
                #这个+1-1操作是为了处理子词的case，比如还是以上面为例，如果答案是written，那么start_position是2
                #orig_to_tok_index[2]也是2，对应context_tokens_after_tokenized中的w
                #而end_position仍然是2，此时orig_to_tok_index[2+1]是6，然后-1=5，此时正好对应context_tokens_after_tokenized中的##n

            else:
                tok_end_position=len(example.context_tokens)-1 if tok_start_position!=-1 else -1
        #上面的代码是由于子词的问题导致答案的位置不对应的问题解决了，此时tok_start_position和tok_end_position代表的就是答案在context_tokens_after_tokenized中的位置
        #context_tokens_after_tokenzied是真正要传给模型的句子
        #下面的代码处理长度太长导致必须切分文档的问题
        max_context_length=max_seq_length-len(query_tokens)-SPECIAL_TOKEN_NUMS
        context_spans=[]
        ContextSpan=collections.namedtuple('ContextSpan',['start','length'])
        start_offset=0
        while start_offset<len(context_tokens_after_tokenized):
            length=len(context_tokens_after_tokenized)-start_offset
            if length>max_context_length:
                length=max_context_length
            context_spans.append(ContextSpan(start=start_offset,length=length))
            if start_offset+length==len(context_tokens_after_tokenized):
                break
            start_offset+=min(length,doc_stride)
        
        for (context_span_index,context_span) in enumerate(context_spans):
            #下面才开始真正的构造的一条feature
            tokens=[]
            token_type_ids=[]
            tokens.append("[CLS]")
            token_type_ids.append(0)

            for token in query_tokens:
                tokens.append(token)
                token_type_ids.append(0)
            
            tokens.append("[SEP]")
            token_type_ids.append(0)

            token_is_max_context = {}
            token_to_orig_map = {}

            for i in range(context_span.length):
                split_token_index=context_span.start+i#split_token_index代表的是当前的这个token在当前这个span的位置

                is_max_context=check_is_max_context(doc_spans=context_spans,cur_span_index=context_span_index,position=split_token_index)
                token_is_max_context[len(tokens)]=is_max_context#用来记录context部分中每一个token是不是有着最大的上下文位置

                token_to_orig_map[len(tokens)]=tok_to_orig_index[split_token_index]
                #记录的是当前tokens中context位置的每一个token对应的在context_tokens中的第几个token
                tokens.append(context_tokens_after_tokenized[split_token_index])
                token_type_ids.append(1)

            tokens.append("[SEP]")
            token_type_ids.append(1)
            
            input_ids=tokenizer.convert_tokens_to_ids(tokens)
            input_mask=[1]*len(input_ids)

            while len(input_ids)<max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                token_type_ids.append(0)

            assert len(input_ids) == len(token_type_ids) == len(input_mask) == max_seq_length
            start_position=None
            end_position=None
            if is_training:
                if tok_start_position==-1 or tok_end_position==-1:
                    start_position=0
                    end_position=0
                else:
                    out_of_span=False
                    context_span_start=context_span.start
                    context_span_end=context_span_start+context_span.length-1

                    if not (tok_start_position>=context_span_start and tok_end_position<=context_span_end):
                        out_of_span=True
                    if out_of_span:
                        start_position=0
                        end_position=0
                    else:
                        _offset=len(query_tokens)+2
                        start_position=tok_start_position-context_span_start+_offset
                        end_position=tok_end_position-context_span_start+_offset

            features.append(InputFeature(unique_id=str(unique_id+1),
                                        example_index=example_index,
                                        doc_span_index=context_span_index,
                                        tokens=tokens,
                                        token_to_orig_map=token_to_orig_map,
                                        token_is_max_context=token_is_max_context,
                                        input_ids=input_ids,
                                        attention_mask=input_mask,
                                        token_type_ids=token_type_ids,
                                        start_position=start_position,
                                        end_position=end_position))
            unique_id+=1
    #logger.info("The total amount of features is {}".format(len(features)))
    return features

    