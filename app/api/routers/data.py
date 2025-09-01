#!/usr/bin/env python
#coding: utf-8

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.components import qwen3_llm, qwen3_thinking_llm
from app.core.data.workflow import DataQaWorkflow
from app.models import DataQAChatCompletionResponse, DataQACompletionRequest, RerankerInfo
from app.utils.log import logger

router = APIRouter(prefix=settings.dataqa_workflow.router_prefix)

dataqa = DataQaWorkflow(
    ans_llm=qwen3_llm,
    ans_thinking_llm=qwen3_thinking_llm,
    query_llm=qwen3_llm,
    reranking_threshold=settings.dataqa_workflow.reranking_threshold,
    collection=settings.dataqa_workflow.milvus_collection,
)

@router.post(
    "/dataqa",
    response_model=DataQAChatCompletionResponse,
    summary="DataQa completions",
)
async def chat_completion(request: DataQAChatCompletionResponse):
    """dataqa_workflow对外服务api实现
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
                dataqa.do_stream(
                    input_messages=request.messages,
                    use_reranker=request.use_reranker,
                    reranker_info=request.reranker_info,
                    knowledge_base_ids=request.knowledge_base_ids,
                    thinking=request.thinking,
                ),
                media_type="text/event-stream",
                headers={"Cache-control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            response = dataqa.do_generate(
                input_messages=request.messages,
                use_reranker=request.use_reranker,
                reranker_info=request.reranker_info,
                knowledge_base_ids=request.knowledge_base_ids,
                thinking=request.thinking,
            )
            return response
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))