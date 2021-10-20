import torch
import jieba
import random
SPIECE_UNDERLINE = '▁'


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

def is_chinese_char(cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False

def is_fuhao(c):
    if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
            or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
            or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
            or c == '‘' or c == '’':
        return True
    return False

def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if is_chinese_char(cp) or is_fuhao(char):
            if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                output.append(SPIECE_UNDERLINE)
            output.append(char)
            output.append(SPIECE_UNDERLINE)
        else:
            output.append(char)
    return "".join(output)

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
        return True
    return False