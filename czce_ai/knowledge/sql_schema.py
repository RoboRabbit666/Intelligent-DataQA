from typing import Any, Dict

from czce_ai.document import Chunk, SQLSchemaChunkData

from .base import BaseKnowledge


class SQLSchemaKnowledge(BaseKnowledge):
    """Knowledge base implementation specialized for document chunks"""

    collection_type = "sql_schema"

    def _convert_record_to_chunk(self, entity: Dict) -> SQLSchemaChunkData:
        """根据Milvus返回结果构造SQLSchemaChunkData对象"""
        return SQLSchemaChunkData(
            table_name=entity.get("table_name"),
            table_info=entity.get("table_info"),
            embed_rerank_content=entity.get("embed_rerank_content"),
        )

    def _convert_chunk_to_record(self, chunk: Chunk) -> Dict[str, Any]:
        """将Chunk对象转换为Milvus兼容的字典格式"""
        if not isinstance(chunk.data, SQLSchemaChunkData):
            raise ValueError("Chunk data must be of type SQLSchemaChunkData")
        data = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "knowledge_id": chunk.knowledge_id,
            "table_name": chunk.data.table_name or "",
            "table_schema": chunk.data.table_schema,
            "table_info": chunk.data.table_info,
            "qa_scenarios": chunk.data.qa_scenarios,
            "embed_rerank_content": chunk.data.embed_rerank_content,
            "meta_data": chunk.meta_data,
        }
        dense_fields = self.config.get("search_params")["dense_fields"]
        embedding_map = {
            "table_schema": chunk.data.table_schema_embedding,
            "qa_scenarios": chunk.data.qa_scenarios_embedding,
        }
        for field in dense_fields:
            for key, embedding in embedding_map.items():
                if key in field and embedding is not None:
                    data[field] = embedding
                    break  # 一个字段只填一次

        data["table_schema_tks"] = " ".join(
            self.tokenizer.tokenize(data["table_schema"], rmSW=True)
        )
        data["qa_scenarios_tks"] = " ".join(
            self.tokenizer.tokenize(data["qa_scenarios"], rmSW=True)
        )
        return data