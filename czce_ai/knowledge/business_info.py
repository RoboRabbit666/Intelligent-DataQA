from typing import Any, Dict

from czce_ai.document import BusinessInfoChunkData, Chunk
from czce_ai.knowledge.base import BaseKnowledge


class BusinessInfoKnowledge(BaseKnowledge):
    """Knowledge base implementation specialized for document chunks"""

    collection_type = "business_info"

    def _convert_record_to_chunk(self, entity: Dict) -> BusinessInfoChunkData:
        """根据Milvus返回结果构造BusinessInfoChunkData对象"""
        return BusinessInfoChunkData(
            business_desc=entity.get("business_desc", ""),
            info_type=entity.get("info_type", ""),
            embed_rerank_content=entity.get("embed_rerank_content", ""),
        )

    def _convert_chunk_to_record(self, chunk: Chunk) -> Dict[str, Any]:
        """将Chunk对象转换为Milvus兼容的字典格式"""
        if not isinstance(chunk.data, BusinessInfoChunkData):
            raise ValueError("Chunk data must be of type BusinessInfoChunkData")
        data = {
            "chunk_id": chunk.chunk_id,
            "knowledge_id": chunk.knowledge_id,
            "business_desc": chunk.data.business_desc or "",
            "info_type": chunk.data.info_type,
            "embed_rerank_content": chunk.data.embed_rerank_content
            or chunk.data.business_desc,
            "meta_data": chunk.meta_data,
        }
        dense_fields = self.config.get("search_params")["dense_fields"]
        embedding_map = {
            "business_desc": chunk.data.business_desc_embedding,
        }
        for field in dense_fields:
            for key, embedding in embedding_map.items():
                if key in field and embedding is not None:
                    data[field] = embedding
                    break  # 一个字段只填一次

        data["business_desc_tks"] = " ".join(
            self.tokenizer.tokenize(data["business_desc"], rmSW=True)
        )
        return data