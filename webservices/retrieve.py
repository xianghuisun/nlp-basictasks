import json,math,os,torch
import numpy as np
from tqdm import tqdm
import faiss
from .log import logging
logger=logging.getLogger(__name__)

from nlp_basictasks.tasks.sts import SimilarityRetrieve

class RetrieveModel:
    def __init__(self,
                    model_name_or_path,
                    encode_dim=512,
                    device='cpu'):
        '''
        model_name_or_path has two cases, if model_name_or_path contains 'BERT' and 'Pooling', then 
        model_name_or_path is a fintuned model path
        else model_name_or_path only has pytorch_model.bin, you need to create pooling config
        '''
        if os.path.exists(os.path.join(model_name_or_path,'BERT')):
            logger.info("Loading BERT model from {}".format(os.path.join(model_name_or_path,'BERT')))
            bert_model_path=os.path.join(model_name_or_path,'BERT')
            pooling_model_path=os.path.join(model_name_or_path,'Pooling')
            if os.path.exists(os.path.join(model_name_or_path,'MLP')):
                mlp_model_path=os.path.join(model_name_or_path,'BERT')
            else:
                mlp_model_path=None
            self.model=SimilarityRetrieve(bert_model_path=bert_model_path,
                                        pooling_model_path=pooling_model_path,
                                        mlp_model_path=mlp_model_path,
                                        max_seq_length=128,
                                        device=device)
        else:
            assert os.path.exists(os.path.join(model_name_or_path,'pytorch_model.bin'))
            logger.info("Loading BERT model from {}".format(model_name_or_path))
            logger.info("In this case, no Pooling and MLP model will be used!")
            self.model=SimilarityRetrieve(bert_model_path=model_name_or_path,
                                        pooling_model_path=None,
                                        mlp_model_path=None,
                                        max_seq_length=128,
                                        device=device)   
        self.encode_dim=encode_dim
    
    def createIndex(self,sentences,
                        batch_size=128,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        convert_to_tensor=False,
                        normalize_embeddings=False):
        logger.info("The number of sentences is {}".format(len(sentences)))
        retrievalbase_embeddings=self.model.encode(sentences=sentences,
                                                batch_size=batch_size,
                                                show_progress_bar=show_progress_bar,
                                                convert_to_numpy=convert_to_numpy,
                                                convert_to_tensor=convert_to_tensor,
                                                normalize_embeddings=normalize_embeddings)
        logger.info("embeddings.shape : {}".format(retrievalbase_embeddings.shape))
        if not normalize_embeddings:
            faiss.normalize_l2(retrievalbase_embeddings)
        nums,dim=retrievalbase_embeddings.shape
        # assert nums==self.pointer_add_pos
        # faiss建立索引
        assert self.encode_dim==dim
        index_flat = faiss.IndexFlatIP(self.encode_dim)
        index_flat.add(retrievalbase_embeddings)

        return index_flat,retrievalbase_embeddings
    
    def createIVFIndex(self,sentences,
                        nlist=100,
                        nprobe=10,
                        batch_size=128,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        convert_to_tensor=False,
                        normalize_embeddings=False):
        index_flat,retrievalbase_embeddings=self.createIndex(sentences,
                                                            batch_size=batch_size,
                                                            show_progress_bar=show_progress_bar,
                                                            convert_to_numpy=convert_to_numpy,
                                                            convert_to_tensor=convert_to_tensor,
                                                            normalize_embeddings=normalize_embeddings)
        nlist=max(nlist,int(8*math.sqrt(len(sentences))))
        index_ivf = faiss.IndexIVFFlat(index_flat,self.encode_dim,nlist,faiss.METRIC_INNER_PRODUCT)
        index_ivf.nprobe=nprobe
        logger.info("The nlist of IVFIndex is {}, nprobe is {}".format(nlist,nprobe))
        ## train IVFFlat index
        logging.info("Training the index_ivf...")
        index_ivf.train(retrievalbase_embeddings)
        assert index_ivf.is_trained

        ## add embeddings to construct index
        insert_ids=np.arange(self.pointer_add_pos)
        index_ivf.add_with_ids(retrievalbase_embeddings,ids=insert_ids)

        return index_ivf

    def get_cos_score(self,sentences1,sentences2):
        input_is_string=False
        if isinstance(sentences1,str):
            assert isinstance(sentences2,str)
            sentences1=[sentences1]
            sentences2=[sentences2]
            input_is_string=True

        ret1 = self.model.encode(sentences1,normalize_embeddings=True,convert_to_tensor=True)
        ret2 = self.model.encode(sentences2,normalize_embeddings=True,convert_to_tensor=True)
        #(batch_size,dim)
        cos_score = torch.sum(ret1*ret2,dim=1).numpy().tolist()#(batch_size,batch_size)
        if input_is_string:
            cos_score=cos_score[0]
        #cos_score的数值是float类型，可以dumps，如果是numpy.float则不能dumps
        return json.dumps(cos_score,ensure_ascii=False)