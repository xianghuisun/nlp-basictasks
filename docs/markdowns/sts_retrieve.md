sts_retrieve.py主要实现检索模型

# 逐个方法分析

## __init__
- sentence_transformers_model代表传进来的模型是不是sentence-transformers框架写的模型，此时检索模型就是sentence-transformers框架写的模型SentenceTransformer
- model_name_or_path代表如果不使用sentence-transformers框架写的模型，那么需要给出模型路径，此时检索模型会利用nlp-basictasks框架从model_name_or_path下加载
- use_IVF_num代表使用Faiss建立IVFFlat类型的索引需要的问题数量。(目前只支持Flat和IVFFlat两个类型，后续会添加其他类型的索引)
- save_index_path代表保存索引的路径，save_query2id_path代表保存query2id的路径。索引中的每一个vector的id与query2id中每一个query的id必须是一一对应的
- save_index_path和save_query2id_path必须同时存在，同时为空。

```python
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
else:
    logger.info("Index is none, you need call createIndex method to build the index")
```

上面的代码含义是如果save_index_path下存在文件，那么save_query2id_path也必然存在文件，此时检索模型需要的索引index以及query2id直接从这两个路径下加载。否则打印日志"Index is none, you need call createIndex method to build the index"，表明此时索引是空的，没有向量和句子在里面，需要外部显示的调用createIndex来创建索引和query2id。

## createIndex

该方法会根据传进来的所有句子，先创建query2id，然后根据句子数量选择性的创建索引，如果句子数量少于指定的use_IVF_num时，就会创建Flat索引，否则创建IVFFlat索引，并且会将query2id以及创建好的索引写在save_query2id_path和save_index_path中。

## retrieval
该方法就是根据传入的句子返回检索的topk个问题

## add_Sentences

该方法将传入的sentences同时插入到index和query2id中



