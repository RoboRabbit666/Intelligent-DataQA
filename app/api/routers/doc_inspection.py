#!/usr/bin/env python
# coding: utf-8
from fastapi import APIRouter, HTTPException
from app.core.components import minio
from app.core.doc_inspection.doc_inspection import DocInspection
from app.models import DocInspectionGenerateRequest, DocInspectionResponse
from app.utils.log import logger
from czce_ai.document.reader import ReaderRegistry


router = APIRouter(prefix="/doc_inspection")
doc_inspection = DocInspection()


@router.post(
    "/generate",
    response_model=DocInspectionResponse,
    summary="doc_inspection completions",
)
async def generate(request: DocInspectionGenerateRequest):
    """doc_inspection对外服务api实现

    Args:
        request:输入的request包体

    Returns:
        response:输出的response
    """
    logger.debug(f"doc_inspection request: {request}".format(request=request))

    if request.doc_uuid is not None:
        if request.doc_bucket is None:
            raise HTTPException(status_code=400, detail="doc_bucket 不能为空")
        if request.doc_original_name is None:
            raise HTTPException(status_code=400, detail="doc_original_name 不能为空")
        try:
            doc_name, doc_type = request.doc_original_name.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError(f"无效的文件名格式:{request.doc_original_name}") from exc
        file_obj = minio.get_object(request.doc_bucket, request.doc_uuid)
    try:
        response = doc_inspection.do_generate(
            input_doc_name=doc_name,
            doc_object=file_obj,
            thinking=request.thinking,
        )
        if response == -1:
            raise HTTPException(status_code=400, detail="article content cannot be empty")
        return response
    except Exception as e:
        logger.error("Error occurred: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e