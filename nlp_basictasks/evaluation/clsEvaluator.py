import logging
import os
import csv
from typing import List
from sklearn import metrics
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from log import logging
#logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

class clsEvaluator:
    """
    This evaluator can be used with the CrossEncoder class.
    It is designed for CrossEncoders with 2 or more outputs. It measure the
    accuracy of the predict class vs. the gold labels.
    """
    def __init__(self, sentences, labels, name: str='', write_csv: bool = True):
        #sentences可能是单句子分类，也可能是双句子
        self.sentences = sentences
        self.labels = labels
        self.name = name

        self.csv_file = "ClsEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy"]
        self.write_csv = write_csv

        logging.info("Evalautor sentence like : \n")
        for i in range(5):
            logging.info(self.sentences[i]+"\t"+str(self.labels[i])+"\n")

    def __call__(self, model, label2id={'0':0,'1':1}, batch_size=32, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        '''
        model.predict根据传进去的列表得到dataloader，然后dataloader.collate_fn==smart_collate_fn
        而在smart_collate_fn中做的是convert_examples_to_examples，
        convert_examples_to_ids中，如果examples中的元素不是InputExample类型，那么会转换为Example类型
        '''
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("ClsEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        pred_scores = model.predict(is_pairs=False, dataloader=self.sentences, batch_size=batch_size)#(num_eval_examples,num_classes)
        pred_labels = np.argmax(pred_scores, axis=1)
        assert len(pred_labels) == len(self.labels)
        acc = np.sum(pred_labels == self.labels) / len(self.labels)

        logging.info("Accuracy: {:.3f}".format(acc))
        if len(label2id)==2:
            assert label2id=={"0":0,"1":1}
            y=np.array([label2id[tag] if type(tag)==str else tag for tag in self.labels])
            pred=pred_scores[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
            auc_val=metrics.auc(fpr, tpr)
            logging.info("AUC: {:.3f}".format(auc_val))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            if len(label2id)==2 and "Auc" not in self.csv_headers:
                self.csv_headers.append("Auc")
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                if len(label2id)==2:
                    writer.writerow([epoch, steps, acc, auc_val])
                else:
                    writer.writerow([epoch, steps, acc])#AUC不能作为多分类的指标

        if len(label2id)==2:
            return auc_val
        else:
            return acc

    @classmethod
    def getAcc(self,predictions,labels):
        return sum(predictions==labels)/len(predictions)