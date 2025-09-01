# coding=utf-8
from app.core.components import document_kb, embedder
from app.models import InsertChunkRequest
from czce_ai.document import Chunk, DocumentChunkData


async def upsert_chunk(request: InsertChunkRequest):
    """更新插入chunk"""
    chunk_data = DocumentChunkData(
        content=request.content,
        title=request.title,
        kwd_tks=request.kwd_tks,
        embed_rerank_content=request.embed_rerank_content,
    )
    chunk = Chunk(
        chunk_id=request.chunk_id,
        data=chunk_data,
        doc_id=request.doc_id,
        knowledge_id=request.knowledge_id,
        meta_data=request.meta_data,
    )
    #生成embedding
    await chunk.aembed(embedder)
    # 异步插入
    await document_kb.aupsert(
        collection=request.collection, chunks=[chunk], batch_size=1
    )