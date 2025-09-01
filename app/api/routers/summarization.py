#!/usr/bin/env python
# coding: utf-8
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.doc_parser import parse
from app.core.summarization.summarization import Summarization
from app.models import (
    ChatCompletionResponse,
    SummaryCompletionRequest,
    SummaryGenerateRequest,
    SummaryGenerateResponse,
)
from app.utils.log import logger


router = APIRouter(prefix="/summarization")
summary = Summarization()


@router.post(
    "/generate", response_model=SummaryGenerateResponse, summary="summary generate"
)
async def generate(request: SummaryGenerateRequest):
    logger.debug(f"SummaryGenerateRequest: {request.json()}")
    if request.doc_uuid is not None:
        if request.doc_bucket is None:
            raise HTTPException(status_code=400, detail="doc_bucket 不能为空")
        if request.doc_original_name is None:
            raise HTTPException(status_code=400, detail="doc_original_name 不能为空")
        article_content = parse(
            doc_uuid=request.doc_uuid,
            doc_bucket=request.doc_bucket,
            doc_original_name=request.doc_original_name,
        )
    else:
        if request.article_content is None or request.article_content.strip() == "":
            raise HTTPException(status_code=400, detail="article_content 不能为空")
        article_content = request.article_content
    try:
        response = summary.do_generate(
            article_content=article_content,
            thinking=False,
            summary_length=request.summary_length,
        )
        summary_text = response.choices[0].message.content
        return SummaryGenerateResponse(
            summary=summary_text,
        )
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="summary completions",
)
async def chat_completion(request: SummaryCompletionRequest):
    """summarization对外服务api实现

    Args:
        request:输入的request包体

    Returns:
        response:输出的response
    """
    logger.debug(f"Summarization request: {request}".format(request=request))
    if request.doc_uuid is not None:
        if request.doc_bucket is None:
            raise HTTPException(status_code=400, detail="doc_bucket 不能为空")
        if request.doc_original_name is None:
            raise HTTPException(status_code=400, detail="doc_original_name 不能为空")
        article_content = parse(
            doc_uuid=request.doc_uuid,
            bucket=request.doc_bucket,
            doc_original_name=request.doc_original_name,
        )
    else:
        if len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="messages 不能为空")
        if request.messages[-1].content is None:
            raise HTTPException(status_code=400, detail="message content 不能为空")
        if request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="message role 必须为user")
        # 取用户message 作为 article content
        article_content = request.messages[-1].content
    try:
        if article_content == "":
            raise HTTPException(status_code=400, detail="article content cannot be empty")
        if request.stream:
            return StreamingResponse(
                summary.do_stream(
                    article_content=article_content,
                    thinking=request.thinking,
                    summary_length=request.summary_length,
                ),
                media_type="text/event-stream",
                headers={"Cache-control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            response = summary.do_generate(
                article_content=article_content,
                thinking=request.thinking,
                summary_length=request.summary_length,
            )
            return response
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))