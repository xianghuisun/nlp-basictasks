import logging
import os,sys
import csv
from typing import Dict, List
import numpy as np
from seqeval.metrics import accuracy_score,classification_report,f1_score,precision_score,recall_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from log import logging
#logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

class nerEvaluator:
    """
    This evaluator can be used with the CrossEncoder class.
    It is designed for CrossEncoders with 2 or more outputs. It measure the
    accuracy of the predict class vs. the gold labels.
    """
    def __init__(self, label2id, seq_in: List[str], seq_out: List[str], name: str='', write_csv: bool = True):
        self.label2id=label2id
        self.seq_in = seq_in

        if type(seq_in[0])==list:
            self.input_is_list=True
        else:
            self.input_is_list=False
        if type(seq_in[0])==str:
            self.input_is_string=True
        else:
            self.input_is_string=False

        self.seq_out = seq_out
        self.name = name
        #传进去的seq_in是一个string
        self.csv_file = "nerEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy", "Precision", "Recall", "F1"]
        self.write_csv = write_csv

        logger.info('Total evaluate nums : {}'.format(len(self.seq_in)))
        logger.info("input is string : {}, input is list : {}".format(self.input_is_string,self.input_is_list))
        logger.info("seq in and out like : \n"+str(self.seq_in[0])+"\t"+str(self.seq_out[0]))
        logger.info("In this evaluator, slot contains "+"("+" ".join(list(self.label2id.keys()))+")")
        #self.slotmetric={key:{} for key in self.label2id.keys()}#记录每一个key的precision,recall and f1 score

    @classmethod
    def from_input_string(cls, label2id, seq_in: List[str], seq_out: List[str], **kwargs):
        return cls(label2id,seq_in,seq_out,**kwargs)


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("nerEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_tags = model.predict(self.seq_in, input_is_string=self.input_is_string, input_is_list=self.input_is_list, show_progress_bar=False)##(num_eval_examples,seq_len)
        #每一个的长度是不一样的
        golden_tags = []
        for i,each_predict_tag in enumerate(pred_tags):
            if type(self.seq_out[i])==str:
                each_golden_label=self.seq_out[i].strip().split(' ')
            else:
                each_golden_label=self.seq_out[i]
            #print(each_predict_tag,each_golden_label,len(each_predict_tag),len(each_golden_label))
            assert len(each_predict_tag)==len(each_golden_label)
            golden_tags.append(each_golden_label)

        result=classification_report(y_pred=pred_tags, y_true=golden_tags, digits=4)
        f1=f1_score(y_pred=pred_tags, y_true=golden_tags)
        acc=accuracy_score(y_pred=pred_tags, y_true=golden_tags)
        precision=precision_score(y_pred=pred_tags, y_true=golden_tags)
        recall=recall_score(y_pred=pred_tags, y_true=golden_tags)

        print(result)
        assert len(self.seq_in)==len(pred_tags)
        with open(os.path.join(output_path,"predict_tags.txt"),"w",encoding="utf-8") as f:
            for text,predicts in zip(self.seq_in,pred_tags):
                if type(text)==list:
                    text=' '.join(text)
                f.write(text+"\t"+" ".join(predicts)+"\n")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc, precision, recall, f1])

        return f1