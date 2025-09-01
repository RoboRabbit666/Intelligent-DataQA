from typing import Any, Dict

from czce_ai.document import Chunk, DocumentChunkData
from .base import BaseKnowledge


class DocumentKnowledge(BaseKnowledge):
    """Knowledge base implementation specialized for document chunks"""

    collection_type = "document"

    def _convert_record_to_chunk(self, entity: Dict) -> DocumentChunkData:
        """根据Milvus返回结果构造DocumentChunkData对象"""
        return DocumentChunkData(
            content=entity.get("content"),
            title=entity.get("title"),
            bag_tks=(
                entity.get("bag_tks").split(" ") if entity.get("bag_tks") else []
            ),
            embed_rerank_content=entity.get("embed_rerank_content"),
        )

    def _convert_chunk_to_record(self, chunk: Chunk) -> Dict[str, Any]:
        """将Chunk对象转换为Milvus兼容的字典格式"""
        if not isinstance(chunk.data, DocumentChunkData):
            raise ValueError("Chunk data must be of type DocumentChunkData")

        data = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "knowledge_id": chunk.knowledge_id,
            "content": chunk.data.content,
            "title": chunk.data.title or "",
            "bag_tks": " ".join(chunk.data.bag_tks) if chunk.data.bag_tks else "",
            "embed_rerank_content": chunk.data.embed_rerank_content
            or chunk.data.content,
            "content_dense_1024": chunk.data.embedding,
            "meta_data": chunk.meta_data,
        }
        data["content_tks"] = " ".join(
            self.tokenizer.tokenize(data["embed_rerank_content"], rmSW=True)
        )
        data["title_tks"] = " ".join(
            self.tokenizer.tokenize(data["title"], for_search=True, rmSW=True)
        )
        return data