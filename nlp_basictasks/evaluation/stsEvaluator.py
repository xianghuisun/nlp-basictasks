import os
import csv
from typing import List
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from log import logging
#logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr

from enum import Enum

class SimilarityFunction(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2
    DOT_PRODUCT = 3

class stsEvaluator:
    """
    This evaluator can be used with the CrossEncoder class.
    It is designed for CrossEncoders with 2 or more outputs. It measure the
    accuracy of the predict class vs. the gold labels.
    """
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], batch_size: int = 16, main_similarity: SimilarityFunction = SimilarityFunction.COSINE, name: str = '', show_progress_bar: bool = True, write_csv: bool = True ,
                    whitening_W = None, whitening_mu = None):
        """
        Constructs an evaluator based for the dataset
        The labels need to indicate the similarity between the sentences.
        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]

        #whitening
        self.whitening_W=whitening_W
        self.whitening_mu=whitening_mu
        if self.whitening_W is not None:
            logging.info("Using whitening to smooth bert sentence embeddings")
            self.n_components=self.whitening_W.shape[1]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        labels = self.scores
        bsz,dim=embeddings1.shape
        #embeddings1.shape==embeddings2.shape==(bsz,768)
        if self.whitening_W is not None:
            #W.shape==(768,n_components), mu.shape==(1,768)
            embeddings1=(embeddings1-self.whitening_mu).dot(self.whitening_W)#(batch_size,n_components)
            embeddings2=(embeddings2-self.whitening_mu).dot(self.whitening_W)#(batch_size,n_components)
            assert embeddings1.shape==embeddings2.shape==(bsz,self.n_components)
        #self.scores就是example.label，这个label通常是1或者0，也可以是浮点数，代表的含义是两个句子的相关度分数
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))#cosine_scores.size()==(batch_size),代表的是两行对应的每一个样本的cosine距离，注意这是距离不是score，所以要1-distance
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]
        #manhattan_distances and euclidean_distances both is (batch_size,)，come along with scores to compute each metrics

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)#second item is p-value
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        logging.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        logging.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
        logging.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))
        logging.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_dot, eval_spearman_dot))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_euclidean,
                                 eval_spearman_euclidean, eval_pearson_manhattan, eval_spearman_manhattan, eval_pearson_dot, eval_spearman_dot])


        if self.main_similarity == SimilarityFunction.COSINE:
            #default is cosine，也就是说计算各个指标的时候是需要传进去scores的，而目前的scores都是类别1和0代替
            return eval_spearman_cosine
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return eval_spearman_euclidean
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return eval_spearman_manhattan
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot)
        else:
            raise ValueError("Unknown main_similarity value")