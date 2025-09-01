#coding utf-8
from typing import List

from czce_ai.document.parser import DocumentParser, ParserConfig
from czce_ai.document.reader import ReaderRegistry
from app.models import DocumentChunkPublic, InsertDocumentKB, SplitterInfo
from czce_ai.document.splitter import SplitterRegister, SplitterType
from app.core.components import document_kb, embedder, minio, qwen3_moe_llm


def insert_and_chunk(req: InsertDocumentKB):
    """解析文档并插入知识库,返回生成的chunks

    Args:
        req (InsertDocumentKB):请求参数
    """
    try:
        doc_name, doc_type = req.doc_original_name.rsplit(".", 1)
    except ValueError:
        raise ValueError(f"无效的文件名格式:{req.doc_original_name}")
    # 创建阅读器实例
    doc_reader = ReaderRegistry.get_reader(doc_type.lower())
    if req.splitter_info is None:
        req.splitter_info = SplitterInfo(
            split_type=SplitterType.Recursive, chunk_size=1024
        )
    # 创建分割器实例
    splitter = SplitterRegister.get_splitter(
        splitter_type=req.splitter_info.split_type,
        chunk_size=req.splitter_info.chunk_size,
        separators=req.splitter_info.separators,
    )
    # 创建解析配置
    llm = None
    if req.enable_table2nl:
        llm = qwen3_moe_llm
    config = ParserConfig(
        knowledge_id=req.kb_uuid,
        collection_name=req.kb_collection,
        bucket_name=req.doc_bucket,
        embedder=embedder,
        minio_client=minio,
        llm=llm,
        reader=doc_reader,
        chunk_strategy=splitter,
        knowledge=document_kb,
    )
    # 根据文件类型选择合适的解析器
    parser = DocumentParser(config)
    parser.process_and_insert(req.doc_uuid, doc_name)
    chunks: List[DocumentChunkPublic] = []
    for c in parser.chunks:
        chunk: DocumentChunkPublic = DocumentChunkPublic(
            uuid=c.chunk_id,
            doc_uuid=req.doc_uuid,
            content=c.data.content,
            kwd_tks=c.data.kwd_tks,
            title=c.data.title,
            embed_rerank_content=c.data.embed_rerank_content,
        )
        chunks.append(chunk)
    return chunks