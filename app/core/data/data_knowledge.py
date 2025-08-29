from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent.parent))
sys.path.append(str(Path.cwd().parent.parent.parent))

import os
import re
import traceback
from czce_ai.utils.log import logger
from czce_ai.document.reader.json_reader import JSONReader
from czce_ai.document.reader.text_reader import TextReader
from czce_ai.document.parser.base import ParserConfig
from app.core.components.components import mxbai_reranker, tokenizer, sql_kb, minio, document_kb, embedder
from czce_ai.document.parser.sql_schema_parser import SQLSchemaParser
from czce_ai.document.reader import DocxReader
from czce_ai.document.parser import DocumentParser
from czce_ai.utils import getUUID
from czce_ai.document import Chunk, DocumentChunkData, QAPairChunkData, ApiChunkData
from czce_ai.knowledge.document import DocumentKnowledge
from czce_ai.knowledge.qa_pair import QAPairKnowledge
from czce_ai.knowledge.api_data import ApiDataKnowledge

bucket_name = "czce-ai-dev"
url = 'http://10.251.146.131:19530'
data_collection_name = "hybrid_sql"
data_knowledge_id = "3cc33ed2-21fb-4452-9e10-528867bd5f99"
domain_collection_name = "domain_k1"
domain_knowledge_id = "3cc33ed2-21fb-4452-9e10-528867bd5f95"
sqlqa_collection_name = "sql_qa"
sqlqa_knowledge_id = "3cc33ed2-21fb-4452-9e10-528867bd5f96"
api_collection_name = 'api_k1'
api_knowledge_id = '3cc33ed2-21fb-4452-9e10-528867bd5f97'


"""
def create_knowledge():
    # 创建数据问答知识库
    collection = "api_k1"
    document_k1 = ApiDataKnowledge(tokenizer, embedder, url, mxbai_reranker)
    document_k1.create_collection(collection, force=True)
create_knowledge()
"""

# 数据问答知识库
class DataKnowledge:
    def __init__(self):
        self.knowledge_id = data_knowledge_id
        self.bucket_name = bucket_name
        self.collection_name = data_collection_name
        self.url = url
        self.reader = JSONReader()
        self.config = ParserConfig(
            knowledge_id=self.knowledge_id,
            collection_name=self.collection_name,
            knowledge=sql_kb,
            bucket_name=bucket_name,
            embedder=embedder,
            minio_client=minio,
            reader=self.reader,
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
                logger.error(
                    f"Processing Document doc_id: {doc_id[i]}, doc_name: {doc_name[i]} Insert into KnowledgeBase Error: {e}")
                traceback.print_exc()
                raise e
"""
# 生成数据表知识库
# 共有12张数据表
data = DataKnowledge()
doc_id_list = [
    "c685c881-6c64-57a5-a34c-2ef38fa3b524",
    "cd8bafcb-d61b-5c93-a05b-aa83d847d8d6",
    "5666b0f7-3fe7-5783-9dc6-b01f7e66f9ed",
    "28ce3a6a-462b-5704-b42d-b69cca9d35de",
    "ce4da5ad-f57d-5299-b8c8-a763220773e4",
    "cef4d44c-6542-5a1a-a9a1-14cd024f472e",
    "e4324094-61e2-5659-88d3-bbf573c9f7ba",
    "51b915fb-803f-5071-a546-0674f18fc79e",
    "2a94722c-8858-5473-9cfc-92e61bae6f8a",
    "ff3f9e35-ed09-57b0-b4c1-66cee074e1aa",
    "9abcd9b6-1133-5342-9110-3596385eba2a",
    "52430df7-3aad-5269-8152-1cc995312956"
]
doc_name_list = [
    "境内期货市场仓库仓单统计表",
    "境内期货市场品种信息表",
    "期货公司利润及客户数统计表_月",
    "郑商所仓库基本信息表",
    "郑商所仓库品种信息表",
    "郑商所合约信息统计表",
    "郑商所会员分支机构信息表",
    "郑商所会员基本信息表",
    "郑商所会员品种成交持仓及客户数统计表",
    "郑商所品种信息统计表",
    "交易日历信息表",
    "外盘期货行情表",
]
data.doc2kl(doc_id=doc_id_list, doc_name=doc_name_list)
"""

# 业务知识库
class DomainKnowledge:
    def __init__(self):
        self.knowledge_id = domain_knowledge_id
        self.bucket_name = bucket_name
        self.collection_name = domain_collection_name
        self.url = url
        self.reader = TextReader()

    def doc2k1(
        self,
        doc_id_list: list,
        doc_name_list: list
    ):
        for i in range(len(doc_id_list)):
            try:
                doc_id = doc_id_list[i]
                doc_name = doc_name_list[i]
                file_obj = minio.get_object(bucket_name=self.bucket_name, object_name=doc_id)
                doc = self.reader.read(file=file_obj, doc_id=doc_id, doc_name=doc_name)
                for line in doc.content.splitlines():
                    print(line.strip())
                    document_data = DocumentChunkData(
                        content=line.strip(),
                        embed_rerank_content=line.strip()
                    )
                    chunk = Chunk(
                        chunk_id=getUUID(""),
                        data=document_data,
                        doc_id=doc_id,
                        embedder=embedder
                    )
                    chunk.embed()
                    chunk.knowledge_id = self.knowledge_id
                    doc_kl = DocumentKnowledge(tokenizer, embedder, self.url, mxbai_reranker)
                    doc_kl.insert(collection=self.collection_name, chunks=[chunk])
                logger.info(f"Document doc_id: {doc_id}, doc_name: {doc_name} Insert into KnowledgeBase Success")
            except Exception as e:
                logger.error(
                    f"Processing Document doc_id: {doc_id[i]}, doc_name: {doc_name[i]} Insert into KnowledgeBase Error:{e}")
                traceback.print_exc()
                raise e

"""
#生成业务知识库
domain = DomainKnowledge()
domain_id_list = ["c68dd67a-ed8c-52f7-8e0e-1f77de244e13"]
domain_name_list = ["indicator"]
domain.doc2kl(doc_id_list=domain_id_list, doc_name_list=domain_name_list)
"""

# 数据API知识库
class APIDataKnowledge:
    def __init__(self):
        self.knowledge_id = api_knowledge_id
        self.bucket_name = bucket_name
        self.collection_name = api_collection_name
        self.url = url
        self.reader = JSONReader()

    def doc2k1(
        self,
        doc_id_list: list,
        doc_name_list: list
    ):
        for i in range(len(doc_id_list)):
            try:
                doc_id = doc_id_list[i]
                doc_name = doc_name_list[i]
                file_obj = minio.get_object(bucket_name=self.bucket_name, object_name=doc_id)
                doc = self.reader.read(file=file_obj, doc_id=doc_id, doc_name=doc_name)
                for api in doc.content:
                    print("api:", api)
                    api_info = "\n".join([
                        f"Name: {api.get('api_name', '')}",
                        f"Description: {api.get('api_description', '')}",
                        f"Request: {api.get('api_request', '')}",
                        f"Response: {api.get('api_response', '')}"
                    ])
                    document_data = ApiChunkData(
                        api_id=api.get("api_id", ""),
                        api_name=api.get("api_name", ""),
                        api_description=api.get("api_description", ""),
                        api_request=api.get("api_request", ""),
                        api_response=api.get("api_response", ""),
                        api_info=api_info
                    )
                    chunk = Chunk(
                        chunk_id=getUUID(""),
                        data=document_data,
                        doc_id=doc_id,
                        embedder=embedder
                    )
                    chunk.embed()
                    chunk.knowledge_id = self.knowledge_id
                    doc_kl = ApiDataKnowledge(tokenizer, embedder, self.url, mxbai_reranker)
                    doc_kl.insert(collection=self.collection_name, chunks=[chunk])
                logger.info(f"Document doc_id: {doc_id}, doc_name: {doc_name} Insert into KnowledgeBase Success")
            except Exception as e:
                logger.error(
                    f"Processing Document doc_id: {doc_id[i]}, doc_name: {doc_name[i]} Insert into KnowledgeBase Error:{e}")
                traceback.print_exc()
                raise e

#生成数据API知识库
api_data = APIDataKnowledge()
api_data_id_list = ["a2c15c75-12ca-5e6a-a5fb-9517f0f51488"]
api_data_name_list = ["api_data"]
api_data.doc2k1(doc_id_list=api_data_id_list, doc_name_list=api_data_name_list)

# SQL QA知识库库
class SqlQaKnowledge:
    def __init__(self):
        self.knowledge_id = sqlqa_knowledge_id
        self.bucket_name = bucket_name
        self.collection_name = sqlqa_collection_name
        self.url = url
        self.reader = TextReader()

    def doc2k1(
        self,
        doc_id_list: list,
        doc_name_list: list
    ):
        for i in range(len(doc_id_list)):
            try:
                doc_id = doc_id_list[i]
                doc_name = doc_name_list[i]
                file_obj = minio.get_object(bucket_name=self.bucket_name, object_name=doc_id)
                doc = self.reader.read(file=file_obj, doc_id=doc_id, doc_name=doc_name)
                logger.info(f"doc_id: {doc_id}, doc_name: {doc_name}")
                pattern = r'问题:(.*?)\n(.*?) (?=\n问题:|\Z)'
                matches = re.findall(pattern, doc.content, re.DOTALL)
                for question, sql in matches:
                    # 清理SQL中的多余空格和换行
                    cleaned_sql = re.sub(r'\n\s+', '\n', sql.strip())
                    qa_pair_data = QAPairChunkData(
                        question=question,
                        answer=cleaned_sql,
                        embed_rerank_content=question
                    )
                    chunk = Chunk(
                        chunk_id=getUUID(""),
                        knowledge_id=getUUID(""),
                        data=qa_pair_data,
                        doc_id=getUUID(""),
                        embedder=embedder
                    )
                    chunk.embed()
                    chunk.knowledge_id = self.knowledge_id
                    doc_kl = QAPairKnowledge(tokenizer, embedder, self.url, mxbai_reranker)
                    doc_kl.insert(collection=self.collection_name, chunks=[chunk])
                logger.info(f"Document doc_id: {doc_id}, doc_name: {doc_name} Insert into KnowledgeBase Success")
            except Exception as e:
                logger.error(
                    f"Processing Document doc_id: {doc_id[i]}, doc_name: {doc_name[i]} Insert into KnowledgeBase Error:{e}")
                traceback.print_exc()
                raise e
"""
sqlqa = SqlQaKnowledge()
sqlqa_id_list = [
    "0aa28673-dcd9-5453-bccc-a6bb16c14164",
    "0a864d11-d7b4-54b4-b428-47ec6cc5468f",
    "c407d549-2f50-5644-9032-f81db74cc974",
    "042c3737-4143-5d2a-823b-0cf6f9739849",
    "51c9f250-7e59-5fb6-b142-f79021d60ebd",
    "2546115c-beda-53c9-943c-529799b8d782",
    "9b2ba6d4-d270-53fe-bbd9-f4e68874f0cb",
    "9fbf7ca0-32f2-5b6b-97cf-c5765c25126c",
    "27acd8ed-e935-52c5-b0c0-77fee4a3a724",
    "f6b29834-02bb-5ecc-8907-738d6720e010",
    "2f1860b2-c7f8-5bd3-9c95-a76a0cfc89fb"
]
sqlqa_name_list = [
    "郑商所合约信息统计表sql知识库",
    "郑商所仓库品种信息表sql知识库",
    "境内期货市场仓库仓单统计表sql知识库",
    "境内期货市场品种信息表sql知识库",
    "期货公司利润及客户数统计表_月sql知识库",
    "外盘期货行情表sql知识库",
    "郑商所仓库基本信息表_sql知识库",
    "郑商所会员分支机构信息表 sql知识库",
    "郑商所会员基本信息表sql知识库",
    "郑商所会员品种成交持仓及客户数统计表sql知识库",
    "郑商所品种信息统计表sql知识库"
]
sqlqa.doc2k1(doc_id_list=sqlqa_id_list, doc_name_list=sqlqa_name_list)
"""