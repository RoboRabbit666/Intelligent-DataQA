#!/usr/bin/env python
# coding: utf-8
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.chat.prompt import chat_system_prompt
from app.core.components import get_llm
from app.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
)


router = APIRouter(prefix="/chat")
support_llm_models = ["Qwen3"]


@router.post(
    "/completions", response_model=ChatCompletionResponse, summary="Chat completions"
)
async def completion(request: ChatCompletionRequest):
    if request.model not in support_llm_models:
        raise HTTPException(status_code=400, detail="Unsupported model")
    llm = get_llm(thinking=request.thinking, model=request.model)
    system_msg = ChatMessage(
        role="system",
        content=chat_system_prompt,
    )
    messages = [system_msg] + request.messages[:]
    # 处理流式请求
    if request.stream:
        async def event_generator():
            for chunk in llm.invoke_stream(messages=messages):
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"  # 发送结束信号

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        # 处理非流式请求
        response = llm.invoke(messages=messages)
        usage = ChatUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        choices = list(
            map(
                lambda x: ChatCompletionChoice(
                    finish_reason=x.finish_reason,
                    index=x.index,
                    message=ChatMessage(
                        role=x.message.role,
                        content=x.message.content,
                        reasoning_content=x.message.reasoning_content,
                    ),
                ),
                response.choices,
            )
        )
        return ChatCompletionResponse(
            id=response.id,
            model=response.model,
            created=response.created,
            choices=choices,
            usage=usage,
        )