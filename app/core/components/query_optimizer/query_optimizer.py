# coding=utf-8
import json
import re
from datetime import datetime
from typing import List, Optional

from czce_ai.llm.chat import LLMChat as LLMModel
from czce_ai.llm.message import Message as ChatMessage
from czce_ai.utils.log import logger

from .entities import OptimizedQuery, QueryOptimizationType
from .prompt import query_optimization_prompt_template_mapping

class QueryOptimizer:
    def __init__(self, llm: LLMModel):
        self.llm = llm
        self.cur_date_str = datetime.now().strftime("%Y年%m月%d日")

    def _format_chat_history(
        self, chat_history: List[ChatMessage], max_messages: int = 6
    ) -> str:
        """处理聊天上下文"""
        if not chat_history:
            return "(无)"
        
        trimmed_history = chat_history[-max_messages:]
        formatted = []
        role_mapping = {"user": "用户", "assistant": "助手"}
        
        for msg in trimmed_history:
            if msg.role in role_mapping:
                prefix = role_mapping.get(msg.role, f"{msg.role}")
                formatted.append(f"[{prefix}]:{msg.content.strip()}")
        return "\n".join(formatted)

    def _build_system_message(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        optimization_type: QueryOptimizationType = QueryOptimizationType.NORMAL,
    ) -> ChatMessage:
        """构造用于查询优化的system prompt"""
        return ChatMessage(
            role="system",
            content=query_optimization_prompt_template_mapping[
                optimization_type
            ].format(
                cur_date_str=self.cur_date_str,
                original_query=query,
                chat_history=self._format_chat_history(chat_history),
            )
        )

    def generate_optimized_query(
        self,
        query: str,
        chat_history: List[ChatMessage],
        optimization_type: QueryOptimizationType = QueryOptimizationType.NORMAL,
    ) -> OptimizedQuery:
        try:
            system_msg = self._build_system_message(
                query, chat_history, optimization_type
            )
            # print(system_msg)
            response = self.llm.invoke(messages=[system_msg])
            if optimization_type == QueryOptimizationType.DECOMPOSE:
                data = self.parse(response.choices[0].message.content)
            else:
                data = {"rewritten_query": response.choices[0].message.content}
            
            if not data.get("rewritten_query"):
                data["rewritten_query"] = query
            return OptimizedQuery.model_validate(data)
        except Exception as e:
            logger.error(f"query optimization Error: {e}")
            raise e

    async def stream_optimized_query(self, query: str, chat_history: List[ChatMessage]):
        """不建议使用"""
        system_msg = self._build_system_message(query, chat_history)
        response = self.llm.invoke_stream(messages=[system_msg])
        # 发送回答数据
        for chunk in response:
            yield f"data: {chunk.model_dump_json()}\n\n"

    def parse(self, raw_text: str):
        try:
            parsed = json.loads(raw_text)
            result = {
                "step_back_query": parsed.get("step_back_query", "").strip(),
                "sub_queries": [s.strip() for s in parsed.get("sub_queries", [])],
                "rewritten_query": parsed.get("rewritten_query", "").strip(),
            }
            return result
        except Exception as e:
            return self._regex_fallback(raw_text)

    def _regex_fallback(self, text: str):
        """正则提取构建json"""
        result = {
            "step_back_query": self._extract_single_field(text, "step_back_query"),
            "sub_queries": self._extract_sub_queries(text),
            "rewritten_query": self._extract_single_field(text, "rewritten_query"),
        }
        return result

    def _extract_single_field(self, text: str, field: str) -> str:
        pattern = re.compile(rf'"{field}"\s*:\s*"([^"]+)"', re.DOTALL)
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    def _extract_sub_queries(self, text: str) -> List[str]:
        pattern = re.compile(
            r'"sub_queries"\s*:\s*\[\s*((?:"[^"]*"\s*,?\s*)+)\]', re.DOTALL
        )
        match = pattern.search(text)
        if not match:
            return []
        item_text = match.group(1)
        items = re.findall(r'"([^"]+)"', item_text)
        return [
            item.strip() for item in items if item.strip() and item != "rewritten_query"
        ]