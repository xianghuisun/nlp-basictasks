import json,math,os,torch
from logging import LogRecord
import numpy as np
from tqdm import tqdm
import faiss
from .log import logging
logger=logging.getLogger(__name__)

class RetrieveModel:
    def __init__(self,
                    save_index_path,
                    save_query2id_path,
                    encode_dim=512,
                    sentence_transformers_model=None,
                    model_name_or_path=None,
                    device='cpu',
                    use_IVF_num=10000):
        '''
        model_name_or_path代表加载BERT模型的路径，model_name_or_path与sentence_transformers_model只能有一个是None
        save_index_path代表保存索引的路径
        save_query2id_path代表保存query2id的路径
        索引中的每一个vector的id与query2id中每一个query的id必须是一一对应的
        '''
        if sentence_transformers_model is not None:
            #这是因为sentence_transformers保存的模型与原生的BertModel.from_pretrained的keys对不上
            #所以如果制定了用sentence_transformers，那么就直接调用sentence_transformers模型
            self.model=sentence_transformers_model
        else:
            #如果不使用sentence_transformers模型
            from nlp_basictasks.tasks.sts import SimilarityRetrieve
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
        logger.info("Necessary model has been successfully loaded!")
        self.use_IVF_num=use_IVF_num#当数据量达到这个数值后，使用IVFFlat索引，这种索引先聚类再检索，速度更快

        self.index=None
        self.is_IVF=False#is_IVF用来指示当前索引的类型是否是IVFFlat类型
        self.query2id={}
        self.query_num=len(self.query2id)
        self.save_index_path=save_index_path
        self.save_query2id_path=save_query2id_path

        if os.path.exists(self.save_index_path):
            #说明当前已经保存了索引，那么直接加载索引和query2id
            assert os.path.exists(self.save_query2id_path)
            self.index=faiss.read_index(self.save_index_path)
            logger.info('Index has been loaded from {}'.format(self.save_index_path))
            if self.index.ntotal>=self.use_IVF_num:
                self.is_IVF=True
                logger.info("Index type is IVFFlat!")
            with open(self.save_query2id_path,encoding='utf-8') as f:
                self.query2id=json.load(f)
                logger.info("query2id has been loaded from {}".format(self.save_query2id_path))
            self.id2query = {id_:query for query,id_ in self.query2id.items()}
            self.pointer_add_pos=list(self.id2query.keys())[-1]
            #之所以不能以len(id2query)作为基准，是因为id2query中的id可能因为删除操作导致不是连续的，所以
            #必须以最后一个问题对应的id为基准，从这个id后面开始插入问题
        else:
            logger.info("Index is none, you need call createIndex method to build the index")

        self.config_keys=['encode_dim','query_num','pointer_add_pos','use_IVF_num','save_index_path',
                        'is_IVF','save_query2id_path']

    def create_query2id(self,sentences):
        query2id={}
        for sentence in sentences:
            if sentence not in query2id:
                query2id[sentence]=len(query2id)
        
        return query2id

    def update_query2id(self):
        '''
        当经过插入或者删除index中的vector后，对于FlatIndex类型的索引，需要及时更新对应的query2id，因为
        index上的插入和删除导致了index中vector的id发生了变化
        '''
        if self.is_IVF==False:
            new_query2id={}
            for key,_ in self.query2id.items():
                new_query2id[key]=len(new_query2id)
            self.query2id=new_query2id
            logging.info("Reset the query2id.")

        with open(self.save_query2id_path,'w',encoding='utf-8') as f:
            json.dump(self.query2id,f)#Remember that after updating query2id, it is necessary to write in file
        self.id2query = {id_:query for query,id_ in self.query2id.items()}

    def createIndex(self,sentences,
                        batch_size=128,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        convert_to_tensor=False,
                        normalize_embeddings=False,
                        nlist=100,
                        nprobe=10):
        self.query2id=self.create_query2id(sentences)
        self.query_num=len(self.query2id)
        if self.query_num<=1:
            raise Exception('The number of sentences only has {}, is too less to create index'.format(self.query_num))
        with open(self.save_query2id_path,'w',encoding='utf-8') as f:
            json.dump(obj=self.query2id,fp=f,ensure_ascii=False)
        logger.info('query2id has been created and saved in {}, which has {} sentences'.format(self.save_query2id_path,self.query_num))
        
        if self.query_num!=len(sentences):
            #如果不相等，说明有重复的问题
            assert self.query_num<len(sentences)
            logger.info("There are {} repeated sentences".format(len(sentences)-self.query_num))
            sentences=list(self.query2id.keys())

        self.pointer_add_pos=self.query_num
        #pointer_add_pos用来指示插入位置，对于IVFFlat类型的索引，删除一个问题之后，其余问题在index中的id是不变的
        #因此当再次插入问题时，不能以index中ntotal为开始位置插入，否则就会出现覆盖的现象

        if self.query_num<self.use_IVF_num:
            logger.info('The total query num in base is {}, less than use_IVF_nums {}, using Flat index'.format(self.query_num,self.use_IVF_num))
            self.index,_=self.createFlatIndex(sentences=sentences,
                                            batch_size=batch_size,
                                            show_progress_bar=show_progress_bar,
                                            convert_to_numpy=convert_to_numpy,
                                            convert_to_tensor=convert_to_tensor,
                                            normalize_embeddings=normalize_embeddings)
        else:
            logger.info('The total query num in base is {}, more than use_IVF_nums {}, using IVFFlat index'.format(self.query_num,self.use_IVF_num))
            self.index=self.createIVFIndex(sentences=sentences,
                                            nlist=nlist,
                                            nprobe=nprobe,
                                            batch_size=batch_size,
                                            show_progress_bar=show_progress_bar,
                                            convert_to_numpy=convert_to_numpy,
                                            convert_to_tensor=convert_to_tensor,
                                            normalize_embeddings=normalize_embeddings)
            self.is_IVF=True

        faiss.write_index(self.index,self.save_index_path)
        logger.info('Index has been build and saved in {}'.format(self.save_index_path))

    def createFlatIndex(self,sentences,
                        batch_size=128,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        convert_to_tensor=False,
                        normalize_embeddings=False):
        retrievalbase_embeddings=self.model.encode(sentences=sentences,
                                                batch_size=batch_size,
                                                show_progress_bar=show_progress_bar,
                                                convert_to_numpy=convert_to_numpy,
                                                convert_to_tensor=convert_to_tensor,
                                                normalize_embeddings=normalize_embeddings)
        logger.info("embeddings.shape : {}".format(retrievalbase_embeddings.shape))
        if not normalize_embeddings:
            faiss.normalize_L2(retrievalbase_embeddings)
        nums,dim=retrievalbase_embeddings.shape
        # assert nums==self.pointer_add_pos
        # faiss建立索引
        if self.encode_dim!=dim:
            raise Exception("The assigned encode_dim is {} and model encode dim is {},  which is mismatch!".format(self.encode_dim,dim))

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
        index_flat,retrievalbase_embeddings=self.createFlatIndex(sentences=sentences,
                                                                batch_size=batch_size,
                                                                show_progress_bar=show_progress_bar,
                                                                convert_to_numpy=convert_to_numpy,
                                                                convert_to_tensor=convert_to_tensor,
                                                                normalize_embeddings=normalize_embeddings)
        nlist=max(nlist,int(8*math.sqrt(len(self.query2id))))
        index_ivf = faiss.IndexIVFFlat(index_flat,self.encode_dim,nlist,faiss.METRIC_INNER_PRODUCT)
        index_ivf.nprobe=nprobe
        logger.info("The nlist of IVFIndex is {}, nprobe is {}".format(nlist,nprobe))
        ## train IVFFlat index
        logger.info("Training the index_ivf...")
        index_ivf.train(retrievalbase_embeddings)
        assert index_ivf.is_trained

        ## add embeddings to construct index
        insert_ids=np.arange(self.pointer_add_pos)
        index_ivf.add_with_ids(retrievalbase_embeddings,ids=insert_ids)

        return index_ivf

    def add_To_Index(self,sentences,insert_ids):
        '''
        add vectors to self.index
        '''
        assert type(sentences)==list
        sentences_embeddings=self.model.encode(sentences,convert_to_numpy=True,normalize_embeddings=False)
        faiss.normalize_L2(sentences_embeddings)
        if insert_ids==[] and self.is_IVF==True:
            raise Exception("For IVFFlay type index, it must provide insert ids")
        if isinstance(insert_ids,list):
            insert_ids=np.array(insert_ids)
        if self.is_IVF:
            self.index.add_with_ids(x=sentences_embeddings,ids=insert_ids)
        else:
            self.index.add(sentences_embeddings)
        faiss.write_index(self.index,self.save_index_path)#一定要将插入后的索引再次写入
        #self.index=faiss.read_index(self.index_path)
        #logger.info("Updated index has been written in {} ".format(self.index_path))

    def add_Sentences(self,sentences):
        if len(self.query2id)==0:
            logger.info("There is no query in query2id, so the input sentences will be used to create index and query2id")
            self.createIndex(sentences=sentences)
            return "Added {} sentences to the index".format(len(self.query2id))

        added_sentences=[]
        added_sentences_ids=[]
        insert_start_id_pos=self.pointer_add_pos
        if isinstance(sentences,str):
            sentences=[sentences]
        logger.info("The number of sentences in current query2id before insert sentences: {}".format(len(self.query2id)))

        if self.is_IVF==False:
            #对于FlatIndex类型的索引，每一次删除和插入向量后，都会重新更新query2id，而且index中的id时连续的
            logger.info("Add vectors to Flat type index")
            for sentence in sentences:
                if sentence not in self.query2id:
                    self.query2id[sentence]=len(self.query2id)
                    added_sentences.append(sentence)
                else:
                    logger.info("sentence {} has already in self.query2id, which id is {}".format(sentence,self.query2id[sentence]))
        else:
            logger.info("Add vectors to IVFFlat type index, pay attention to the coverage situation, current pointer_add_pos is {}, vectors will be inserted into the index staring from this id, similarly for query2id".format(self.pointer_add_pos))
            for sentence in sentences:
                if sentence not in self.query2id:
                    self.query2id[sentence]=insert_start_id_pos
                    added_sentences.append(sentence)
                    added_sentences_ids.append(self.pointer_add_pos)
                    self.pointer_add_pos+=1
                else:
                    logger.info("sentence {} has already in self.query2id, which id is {}".format(sentence,self.query2id[sentence]))
        
        self.update_query2id()#write the query2id after added sentences to save_query2id_path
        logger.info("The number of sentences in current query2id after insert sentences: {}".format(len(self.query2id)))
        ################################Insert the corresponding vectors to index@#######################
        insert_ids=np.arange(insert_start_id_pos,self.pointer_add_pos)# for IVFFlat type index to avoid coverage situation
        logger.info("The index ntotal is {} before adding vectors".format(self.index.ntotal))
        self.add_To_Index(sentences=added_sentences,insert_ids=insert_ids)
        logger.info("The index ntotal is {} after adding vectors".format(self.index.ntotal))
        logger.info("updated index and query2id have been saved in {} and {}".format(self.save_index_path,self.save_query2id_path))
        return "Added {} sentences to the index".format(len(added_sentences))

    def delete_Sentences(self,sentences):
        '''
        删除操作对于Flat和IVF两种类型的索引是一样的，只需要删除query2id中的query，取出对应的id
        然后删除id在索引中对应的vector即可
        '''
        if len(self.query2id)==0:
            raise Exception("There are no sentence in query2id, you can not delete any sentences")

        deleted_sentences=[]
        deleted_sentences_ids=[]
        if isinstance(sentences,str):
            sentences=[sentences]
        
        logger.info("The number of sentences in current query2id before delete sentences: {}".format(len(self.query2id)))
        for sentence in sentences:
            if sentence in self.query2id:
                deleted_sentences.append(sentence)
                deleted_sentences_ids.append(self.query2id[sentence])
                del self.query2id[sentence]
        
        logger.info("The number of sentences in current query2id after delete sentences: {}".format(len(self.query2id)))
        self.update_query2id()
        #根据query2id中的query对应的id删除在索引中对应的id的vectpr
        logger.info("The index ntotal is {} before delete vectors".format(self.index.ntotal))
        deleted_sentences_ids=np.array(deleted_sentences_ids)
        self.index.remove_ids(deleted_sentences_ids)
        logger.info("The index ntotal is {} after delete vectors".format(self.index.ntotal))
        faiss.write_index(self.index,self.save_index_path)
        logger.info("updated index and query2id have been saved in {} and {}".format(self.save_index_path,self.save_query2id_path))
        return "Deleted {} sentences to the index".format(len(deleted_sentences))

    def retrieval(self,sentence, topk):
        '''
        返回一个list，长度是topk，每一个元素是一个tuple，包含检索问题和对应的余弦分数
        '''
        input_is_string=False
        if type(sentence)==str:
            input_is_string=True
            sentence=[sentence]

        sentence_embeddings=self.model.encode(sentence,convert_to_numpy=True,normalize_embeddings=False,show_progress_bar=False)
        faiss.normalize_L2(sentence_embeddings)
        D,I=self.index.search(sentence_embeddings,topk)
        if input_is_string:
            D=D[0]
            I=I[0]
        D=D.tolist()
        I=I.tolist()
        #D和I都是长度为topk的list，D代表的是topk个余弦分数,I代表的是topk个检索问题的id

        result=[]
        for id_,score in zip(I,D):
            retrieval_query=self.id2query[id_]
            cos_score=float(score)
            result.append((retrieval_query,cos_score))
        
        return result


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