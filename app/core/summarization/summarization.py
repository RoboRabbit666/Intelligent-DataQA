#coding utf-8
import traceback

from app.core.components import get_llm, qwen3_moe_llm
from app.core.summarization.prompt import summarization_system_prompt
from app.models import ChatCompletionChoice, ChatCompletionResponse, ChatUsage
from czce_ai.llm.message import Message as ChatMessage
from czce_ai.utils.log import logger


class Summarization:
    def __init__(self):
        pass

    def do_generate(
        self, article_content: str, summary_length: int = 250, thinking: bool = False
    ) -> ChatMessage:
        try:
            system_msg = ChatMessage(
                role="system",
                content=summarization_system_prompt.format(
                    paragraph=article_content,
                    summary_length=(
                        250 if summary_length is None else summary_length
                    ), #使用 summary_length 参数,如果为 None 则默认为 250
                ),
            )
            llm = qwen3_moe_llm
            response = llm.invoke(messages=[system_msg])
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
        except Exception as e:
            logger.error(f"Modify query Error:{e}")
            traceback.print_exc()
            raise e

    async def do_stream(
        self, article_content: str, summary_length: int = 250, thinking: bool = False
    ):
        system_msg = ChatMessage(
            role="system",
            content=summarization_system_prompt.format(
                paragraph=article_content, summary_length=summary_length
            ),
        )
        llm = get_llm(thinking=thinking)
        response = llm.invoke_stream(messages=[system_msg])
        #发送回答数据
        for chunk in response:
            yield f"data: {chunk.model_dump_json()}\n\n"