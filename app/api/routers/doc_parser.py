#!/usr/bin/env python
# coding: utf-8

from fastapi import APIRouter

from app.core.doc_parser import parse as parse_doc
from app.models import DocParserRequest, DocParserResponse
from app.utils.log import logger


router = APIRouter(prefix="/doc_parser")


@router.post(
    "/parse",
    response_model=DocParserResponse,
    summary="Parse document and extract information",
)
async def parse(request: DocParserRequest):
    logger.debug(f"DocParserRequest: {request}".format(request=request))

    content = parse_doc(
        doc_uuid=request.doc_uuid,
        doc_bucket=request.doc_bucket,
        doc_original_name=request.doc_original_name,
    )
    return DocParserResponse(content=content)