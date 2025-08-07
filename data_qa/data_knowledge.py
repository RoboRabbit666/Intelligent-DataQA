from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

import traceback
from czce_ai.utils.log import logger
from czce_ai.document.reader import JSONReader
from czce_ai.document.reader.text_reader import TextReader
from czce_ai.document.parser import ParserConfig, SQLSchemaParser
from app.core.components import mxbai_reranker, embedder, tokenizer, sql_kb, minio, document_kb

from czce_ai.document.reader import DocxReader
from czce_ai.document.parser import DocumentParser

bucket_name = "czce-ai-dev"
url = 'http://10.251.146.131:19530'

data_collection_name = "hybrid_sql"
data_knowledge_id = "3cc33ed2-21fb-4452-9e10-528867bd5f99"

domain_collection_name = "domain_kl"
domain_knowledge_id = "3cc33ed2-21fb-4452-910-528867bd5f95"

#数据问答知识库
class DataKnowledge:
    def __init__(self):
        self.knowledge_id = data_knowledge_id
        self.bucket_name = bucket_name
        self.collection_name = data_collection_name
        #self.sql_kl = SQLSchemaKnowledge(tokenizer, embedder, url, mxbai_reranker)
        self.url = url
        self.reader = JSONReader()
        self.config = ParserConfig(
            self.knowledge_id,
            self.collection_name,
            sql_kb,
            bucket_name,
            embedder,
            minio,
            self.reader
        )
        self.parser = SQLSchemaParser(self.config)

    def doc2k1(
        self,
        doc_id: list,
        doc_name: list
    ):
        for i in range(len(doc_id)):
            try:
                self.parser.process_and_insert(doc_id[i], doc_name[i])
            except Exception as e:
                logger.error(f"Processing Document doc_id: {doc_id[i]}, doc_name: {doc_name[i]} Insert into KnowledgeBase Error:{e}")
                traceback.print_exc()
                raise e

# 业务知识库
class DomainKnowledge:
    def __init__(self):
        self.knowledge_id = domain_knowledge_id
        self.bucket_name = bucket_name
        self.collection_name = domain_collection_name
        self.url = url
        self.reader = TextReader()
        self.config = ParserConfig(
            self.knowledge_id,
            self.collection_name,
            document_kb,
            bucket_name,
            embedder,
            minio,
            self.reader
        )
        self.parser = DocumentParser(self.config)

    def doc2k1(
        self,
        doc_id: list,
        doc_name: list
    ):
        for i in range(len(doc_id)):
            try:
                self.parser.process_and_insert(doc_id[i], doc_name[i])
            except Exception as e:
                logger.error(f"Processing Document doc_id: {doc_id[i]}, doc_name: {doc_name[i]} Insert into KnowledgeBase Error:{e}")
                traceback.print_exc()
                raise e

data = DataKnowledge()

doc_id_list=[
    "04d020e9-3685-512d-b1b0-9fc7b2d172f4",
    "2fdc3975-eb12-580c-be65-c45c165b07ac",
    "6805cc0c-59da-5689-b555-af4960aa0786",
    "ae91a4ca-6dbf-5307-9d6b-71aad4ea4af5",
    "24817b79-799a-56ab-a9e7-e883a788d17e",
    "bifeaca1-2e6a-5937-9389-dfc72e83f206",
    "1755c0df-2f6d-56b7-af1e-5bdb9ea75b24",
    "f2f5bafa-0d4f-58f5-b27d-dbbedb28005b",
    "e013717c-7a9a-5eb5-87c2-faa53c089366",
    "3ec28ffb-55c0-571e-b553-f10008527c7d",
    "a0706249-8a56-5215-966b-a178c0153896",
    "e9cec1ac-a68c-568b-ba80-d9537853d31d"
]

doc_name_list = [
    "交易日历信息表",
    "境内期货市场仓库仓单量统计信息表",
    "境内期货市场品种信息表",
    "期货公司利润及客户统计表",
    "外盘期货行情基本信息表",
    "郑商所仓库基本信息表",
    "郑商所仓库品种信息表",
    "郑商所合约信息表",
    "郑商所会员持仓交易及客户统计信息表",
    "郑商所会员分支机构信息表",
    "郑商所会员基本信息表",
    "郑商所品种信息表",
    "郑商所合约基本信息表",
    "郑商所会员持仓交易统计信息表",
    "郑商所会员持仓交易及客户统计信息表",
]
#data.doc2k1(doc_id=doc_id_list, doc_name=doc_name_list)

# 业务知识库
domain = DomainKnowledge()
domain_id_list = ["8cd91343-0d9e-5b74-836f-74dc4d5c7ca7"]
domain_name_list = ["indicator"]
domain.doc2k1(doc_id=domain_id_list, doc_name=domain_name_list)