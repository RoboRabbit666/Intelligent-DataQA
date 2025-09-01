# coding=utf-8
from typing import List
from fastapi import APIRouter, HTTPException

from app.core.components import document_kb, sql_kb
from app.core.knowledge.chunk import upsert_chunk
from app.core.knowledge.document import insert_and_chunk
from app.models import (
    CreateKBCollection,
    DeleteChunkRequest,
    DeleteDocumentKB,
    DeleteKnowledgeBase,
    DocumentChunkPublic,
    InsertChunkRequest,
    InsertDocumentKB,
    KBCollectionPublic,
    KBType,
)
from app.utils.log import logger


router = APIRouter(prefix="/knowledge")


@router.post("/collection", response_model=KBCollectionPublic)
async def create_collection(request: CreateKBCollection):
    """document类型知识库

    Args:
        request (CreateDocumentKB):请求参数

    Raises:
        HTTPException: 500 失败异常
    """
    if request.kb_type is KBType.DOCUMENT:
        document_kb.create_collection(
            collection=request.collection,
        )
    elif request.kb_type is KBType.SQL:
        sql_kb.create_collection(
            collection=request.collection,
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid KBType")
    return KBCollectionPublic(
        collection=request.collection,
        kb_type=request.kb_type,
    )


@router.post("/document/insert", response_model=List[DocumentChunkPublic])
async def insert_document_kb(request: InsertDocumentKB):
    """将minio中的文档 插入 document 类型的知识库

    Args:
        request (InsertDocumentKB):请求参数

    Raises:
        HTTPException: 500 失败异常
    """
    try:
        response = insert_and_chunk(request)
        return response
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document/delete", status_code=204)
async def delete_document_kb(request: DeleteDocumentKB):
    """删除document类型知识库中的文档

    Args:
        request (DeleteDocumentKB):请求参数

    Raises:
        HTTPException: 500 失败异常
    """
    try:
        document_kb.delete_with_ids(
            doc_id=request.doc_uuid,
            knowledge_id=request.kb_uuid,
            collection=request.collection,
        )
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete", status_code=204)
async def delete_knowledge_base(request: DeleteKnowledgeBase):
    """删除知识库

    Args:
        request (DeleteKnowledgeBase):请求参数

    Raises:
        HTTPException: 500 失败异常
    """
    try:
        document_kb.delete_with_ids(
            knowledge_id=request.kb_uuid,
            collection=request.collection,
        )
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chunk/upsert", status_code=204)
async def upsert_document_chunk(request: InsertChunkRequest):
    """更新或插入指定chunk"""
    try:
        await upsert_chunk(request=request)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chunk/delete", status_code=204)
async def delete_document_chunk(request: DeleteChunkRequest):
    """删除指定chunk

    Args:
        request (DeleteChunkRequest): _description_

    Raises:
        HTTPException: _description_
    """
    try:
        await document_kb.adelete_with_ids(
            collection=request.collection,
            chunk_id=request.chunk_id,
            knowledge_ids=[request.knowledge_id],
        )
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))