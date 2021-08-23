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
    def __init__(self, sentences, label_ids, label2id, name: str='', write_csv: bool = True):
        self.sentences = sentences
        if type(label_ids[0])==str:
            logger.info("input label is str type, converting to id!")
            label_ids=[label2id[label_id] for label_id in label_ids]
        self.label_ids = label_ids
        self.label2id=label2id
        self.id2label={id_:tag for tag,id_ in self.label2id.items()}
        logger.info('label2id like : {}'.format(self.label2id))
        self.name = name
        self.label2metrics={}
        for key in self.label2id.keys():
            self.label2metrics[key]={'golden_num':0,'predict_num':0,'predict_correct_num':0}
            #标签是key的真实样本的数量、预测key的数量、预测正确的样本数量
        for tag_id in self.label_ids:
            tag=self.id2label[tag_id]
            self.label2metrics[tag]['golden_num']+=1

        for tag in self.label2metrics.keys():
            logger.info('The number of '+tag+' in dataset is {}'.format(self.label2metrics[tag]['golden_num']))

        self.csv_file = "ClsEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy"]
        self.write_csv = write_csv

        logging.info("Evalautor sentence like : \n")
        for i in range(5):
            logging.info(self.sentences[i]+"\t"+str(self.label_ids[i])+"\n")

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
        pred_label_ids = np.argmax(pred_scores, axis=1)
        assert len(pred_label_ids) == len(self.label_ids)
        acc = np.sum(pred_label_ids == self.label_ids) / len(self.label_ids)

        logging.info("Accuracy: {:.3f}".format(acc))

        for i,pred_tag_id in enumerate(pred_label_ids):
            tag=self.id2label[pred_tag_id]
            self.label2metrics[tag]['predict_num']+=1
            if pred_tag_id==self.label_ids[i]:
                self.label2metrics[tag]['predict_correct_num']+=1

        for label,values in self.label2metrics.items():
            precision=values['predict_correct_num']/(values['predict_num']+0.01)
            recall=values['predict_correct_num']/(values['golden_num']+0.01)
            f1_score=2*precision*recall/(precision+recall+0.01)
            logger.info(label+'\t'+"precision : %f, recall : %f,  f1 score : %f"%(precision,recall,f1_score))
            self.label2metrics[label]['predict_correct_num']=0
            self.label2metrics[label]['predict_num']=0
            #不归0就变成累加了
        
        if len(label2id)==2:
            assert label2id=={"0":0,"1":1}
            y=np.array([label2id[tag] if type(tag)==str else tag for tag in self.label_ids])
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