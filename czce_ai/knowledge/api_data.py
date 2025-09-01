from typing import Any, Dict

from czce_ai.document import ApiChunkData, Chunk
from czce_ai.knowledge.base import BaseKnowledge


class ApiDataKnowledge(BaseKnowledge):
    """Knowledge base implementation specialized for API data chunks"""

    collection_type = "api_data"

    def _convert_record_to_chunk(self, entity: Dict) -> ApiChunkData:
        """根据Milvus返回结果构造ApiChunkData对象"""
        return ApiChunkData(
            api_id=entity.get("api_id", ""),
            api_name=entity.get("api_name", ""),
            api_description=entity.get("api_description", ""),
            api_request=entity.get("request_parameters", ""),
            api_response=entity.get("response_parameters", ""),
            api_info=entity.get("api_info", ""),
            embed_rerank_content=entity.get("embed_rerank_content"),
        )

    def _convert_chunk_to_record(self, chunk: Chunk) -> Dict[str, Any]:
        """将Chunk对象转换为Milvus兼容的字典格式"""
        if not isinstance(chunk.data, ApiChunkData):
            raise ValueError("Chunk data must be of type ApiChunkData")
        data = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "knowledge_id": chunk.knowledge_id,
            "api_id": chunk.data.api_id,
            "api_name": chunk.data.api_name,
            "api_description": chunk.data.api_description,
            "api_request": chunk.data.api_request,
            "api_response": chunk.data.api_response,
            "api_info": chunk.data.api_info,
            "embed_rerank_content": chunk.data.embed_rerank_content
            or chunk.data.api_info,
            "meta_data": chunk.meta_data,
        }

        dense_fields = self.config.get("search_params")["dense_fields"]
        embedding_map = {
            "api_info": chunk.data.api_info_embedding,
        }

        for field in dense_fields:
            for key, embedding in embedding_map.items():
                if key in field and embedding is not None:
                    data[field] = embedding
                    break  # 一个字段只填一次

        data["api_info_tks"] = " ".join(
            self.tokenizer.tokenize(data["api_info"], rmSW=True)
        )
        return data