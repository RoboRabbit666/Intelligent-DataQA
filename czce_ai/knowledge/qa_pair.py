from typing import Any, Dict

from czce_ai.document import Chunk, QAPairChunkData
from czce_ai.knowledge.base import BaseKnowledge


class QAPairKnowledge(BaseKnowledge):
    """Knowledge base implementation specialized for document chunks"""

    collection_type = "qa_pair"

    def _convert_record_to_chunk(self, entity: Dict) -> QAPairChunkData:
        """根据Milvus返回结果构造QAPairChunkData对象"""
        return QAPairChunkData(
            question=entity.get("question", ""),
            answer=entity.get("answer", ""),
            embed_rerank_content=entity.get("embed_rerank_content"),
        )

    def _convert_chunk_to_record(self, chunk: Chunk) -> Dict[str, Any]:
        """将Chunk对象转换为Milvus兼容的字典格式"""
        if not isinstance(chunk.data, QAPairChunkData):
            raise ValueError("Chunk data must be of type QAPairChunkData")
        data = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "knowledge_id": chunk.knowledge_id,
            "question": chunk.data.question,
            "answer": chunk.data.answer,
            "embed_rerank_content": chunk.data.embed_rerank_content
            or chunk.data.question,
            "meta_data": chunk.meta_data,
        }
        dense_fields = self.config.get("search_params")["dense_fields"]
        embedding_map = {
            "question": chunk.data.question_embedding,
        }

        for field in dense_fields:
            for key, embedding in embedding_map.items():
                if key in field and embedding is not None:
                    data[field] = embedding
                    break  # 一个字段只填一次

        data["question_tks"] = " ".join(
            self.tokenizer.tokenize(data["question"], rmSW=True)
        )
        return data