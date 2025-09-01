#!/usr/bin/env python
# coding: utf-8

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config.config import settings
from app.core.rag.workflow import RagWorkflow
from app.utils.log import logger

from app.core.components import qwen3_llm, qwen3_thinking_llm
from app.models import ChatCompletionResponse, KnoqCompletionRequest, RerankerInfo


router = APIRouter(prefix=settings.rag_workflow.router_prefix)
rag = RagWorkflow(
    ans_llm=qwen3_llm,
    ans_thinking_llm=qwen3_thinking_llm,
    query_llm=qwen3_llm,
    reranking_threshold=settings.rag_workflow.reranking_threshold,
)


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Knoq completions",
)
async def chat_completion(request: KnoqCompletionRequest):
    """rag_workflow对外服务api实现

    Args:
        request:输入的request包体

    Returns:
        response:输出的response
    """
    try:
        if request.reranker_info is None:
            request.reranker_info = RerankerInfo()
        if request.stream:
            return StreamingResponse(
                rag.do_stream(
                    input_messages=request.messages,
                    collection_name=request.collection_name,
                    use_reranker=request.use_reranker,
                    reranker_info=request.reranker_info,
                    knowledge_base_ids=request.knowledge_base_ids,
                    thinking=request.thinking,
                ),
                media_type="text/event-stream",
                headers={"Cache-control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            response = await rag.ado_generate(
                input_messages=request.messages,
                collection_name=request.collection_name,
                use_reranker=request.use_reranker,
                reranker_info=request.reranker_info,
                knowledge_base_ids=request.knowledge_base_ids,
                thinking=request.thinking,
            )
            return response
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))