#coding utf-8
import ast
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from czce_ai.llm.chat import LLMChat as LLMModel
from czce_ai.llm.message import Message as ChatMessage

from .entities import (
    BaseOptimizedQuery,
    DataQAOptimizedQuery,
    QueryOptimizationType,
    RAGOptimizedQuery,
)
from .prompt import query_optimization_prompt_template_mapping


class BaseOptimizationStrategy:
    """问题优化策略基类,提供共性逻辑"""

    # 由子类定义结果模型。
    result_model: Type[BaseOptimizedQuery]

    def __init__(self, llm: LLMModel, optimization_type: QueryOptimizationType):
        self.llm = llm
        self.optimization_type = optimization_type

    @property
    def prompt_template(self) -> str:
        """未来使用配置中心支持prompt热加载 TODO"""
        return query_optimization_prompt_template_mapping[self.optimization_type]

    @property
    def cur_date(self):
        """动态获取时间"""
        weekday_map = [
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "星期六",
            "星期日",
        ]
        today = datetime.now()
        weekday_name = weekday_map[today.weekday()]
        return today.strftime("%Y年%m月%d日") + " " + weekday_name

    def build_system_message(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs,
    ) -> ChatMessage:
        """构建系统消息,所有策略公用,子策略可以重写独有逻辑"""
        formatted = self._format_chat_history(chat_history)
        content = self.prompt_template.format(
            cur_date=self.cur_date, original_query=query, chat_history=formatted
        )
        return ChatMessage(role="system", content=content)

    def _format_chat_history(
        self, chat_history: List[ChatMessage], max_messages: int = 8
    ) -> str:
        """格式化聊天历史信息"""
        if not chat_history:
            return "(无)"
        trimmed = chat_history[-max_messages:]
        role_map = {"user": "用户", "assistant": "助手"}
        return "\n".join(
            f"[{role_map.get(m.role, m.role)}]:{m.content.strip()}" for m in trimmed
        )

    def parse_response(self, text: str) -> Dict[str, Any]:
        """解析LLM响应,默认JSON解析+正则回退,子类可重写"""
        try:
            return json.loads(text)
        except Exception:
            return self._regex_fallback(text)

    def _regex_fallback(self, text: str) -> Dict[str, Any]:
        """正则提取"""
        result = {}
        for field in ["rewritten_query", "step_back_query"]:
            match = re.search(rf'"{field}"\s*:\s*"([^"\n]*)"', text)
            if match:
                result[field] = match.group(1).strip()
        match = re.search(r'"sub_queries"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if match:
            result["sub_queries"] = [
                s.strip('"') for s in re.findall(r'"(.*?)"', match.group(1))
            ]
        return result

    def generate(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs,
    ) -> BaseOptimizedQuery:
        """生成优化结果"""
        self.original_query = query
        system_msg = self.build_system_message(
            query=query, chat_history=chat_history, **kwargs
        )
        response = self.llm.invoke(messages=[system_msg])
        data = self.parse_response(response.choices[0].message.content)
        data["original_query"] = query
        return self.result_model.model_validate(data)

    async def agenerate(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs,
    ) -> BaseOptimizedQuery:
        self.original_query = query
        system_msg = self.build_system_message(
            query=query, chat_history=chat_history, **kwargs
        )
        response = await self.llm.invoke(messages=[system_msg])
        data = self.parse_response(response.choices[0].message.content)
        data["original_query"] = query
        return self.result_model.model_validate(data)


class RAGStrategy(BaseOptimizationStrategy):
    def __init__(self, llm: LLMModel, optimization_type: QueryOptimizationType):
        super().__init__(llm, optimization_type)
        if self.optimization_type not in [
            QueryOptimizationType.BACK,
            QueryOptimizationType.DECOMPOSE,
            QueryOptimizationType.REWRITE,
            QueryOptimizationType.UNIFIED,
        ]:
            raise ValueError(
                f"optimization_type({optimization_type}) is not in [back, decompose, rewrite, unified]"
            )

    def parse_response(self, text: str) -> Dict[str, Any]:
        if self.optimization_type == QueryOptimizationType.UNIFIED:
            data = super().parse_response(text)
            # 确保rewritten_query有默认值
            data["rewritten_query"] = data.get("rewritten_query") or self.original_query
            return data
        elif self.optimization_type == QueryOptimizationType.REWRITE:
            return {"rewritten_query": text.strip() or self.original_query}
        elif self.optimization_type == QueryOptimizationType.DECOMPOSE:
            try:
                sub_queries = [s.strip() for s in ast.literal_eval(text) if s.strip()]
                return {"sub_queries": sub_queries}
            except Exception as e:
                return {"sub_queries": [text]}
        elif self.optimization_type == QueryOptimizationType.BACK:
            return {"step_back_query": text.strip() or self.original_query}

    @property
    def result_model(self):
        return RAGOptimizedQuery


class DataQAStrategy(BaseOptimizationStrategy):
    """数据问答优化策略类"""

    def __init__(self, llm: LLMModel, optimization_type: QueryOptimizationType):
        super().__init__(llm, optimization_type)
        if self.optimization_type not in [
            QueryOptimizationType.DATAQA,
            QueryOptimizationType.FOLLOWUP,
        ]:
            raise ValueError(
                f"optimization_type({optimization_type}) is not in [dataqa, follow-up]"
            )

    @property
    def result_model(self):
        return DataQAOptimizedQuery

    def parse_response(self, text: str) -> Dict[str, Any]:
        if self.optimization_type == QueryOptimizationType.FOLLOWUP:
            data = super().parse_response(text)
            # 确保rewritten_query有默认值
            data["rewritten_query"] = data.get("rewritten_query") or self.original_query
            return data
        elif self.optimization_type == QueryOptimizationType.DATAQA:
            return {"rewritten_query": text.strip() or self.original_query}

    def build_system_message(
        self,
        query: str,
        chat_history: List[ChatMessage],
        **kwargs,
    ) -> ChatMessage:
        """重建消息构建"""
        formatted = {}
        if self.optimization_type == QueryOptimizationType.DATAQA:
            formatted["chat_history"] = self._format_chat_history(chat_history)
            formatted["original_query"] = query
        elif self.optimization_type == QueryOptimizationType.FOLLOWUP:
            # 这里的逻辑是将用户的最新输入重新放入chat_history,然后找出真正的query
            cur_msg = ChatMessage(
                role="user", content=query, reasoning_content=None, is_follow_up=None
            )
            chat_messages = (
                [cur_msg] if chat_history is None else chat_history + [cur_msg]
            )
            # 定位本轮数据问答的开始位置
            idx = -1
            for i in range(len(chat_messages) - 1, -1, -1):
                if (
                    chat_messages[i].role == "assistant"
                    and not chat_messages[i].is_follow_up
                ):
                    idx = i
                    break
            
            chat_history = [] if idx < 0 else chat_messages[: idx + 1]
            formatted["chat_history"] = self._format_chat_history(chat_history)
            formatted["original_query"] = chat_messages[idx + 1].content
            formatted["follow_up_history"] = self._format_follow_up(
                chat_messages[idx + 2 :]
            )
        
        content = self.prompt_template.format(cur_date=self.cur_date, **formatted)
        return ChatMessage(role="system", content=content)

    def _format_follow_up(self, follow_up_messages: List[ChatMessage]) -> str:
        if not follow_up_messages:
            return "(无)"
        role_map = {"user": "[用户][补充]", "assistant": "[助手][追问]"}
        return "\n".join(
            [
                f"{role_map[msg.role]}: {msg.content.strip()}"
                for msg in follow_up_messages
            ]
        )

    def _regex_fallback(self, text: str) -> Dict[str, Any]:
        """重写函数,补充is_sufficient正则提取"""
        data = super()._regex_fallback(text)
        if not data.get("rewritten_query"):
            match = re.search(rf'"rewritten_query"\s*:\s*"([^"\n]*)"', text)
            if match:
                data["rewritten_query"] = match.group(1).strip()
        
        match = re.search(
            r'"is_sufficient"\s*:\s*(true|false)', text, re.IGNORECASE
        )
        if match:
            data["is_sufficient"] = match.group(1).lower() == "true"
        return data


class OptimizationStrategyFactory:
    """优化策略工厂类,统一管理策略创建和配置"""

    # 策略类型映射
    _strategy_mapping: Dict[QueryOptimizationType, Type[BaseOptimizationStrategy]] = {
        QueryOptimizationType.BACK: RAGStrategy,
        QueryOptimizationType.DECOMPOSE: RAGStrategy,
        QueryOptimizationType.REWRITE: RAGStrategy,
        QueryOptimizationType.UNIFIED: RAGStrategy,
        QueryOptimizationType.DATAQA: DataQAStrategy,
        QueryOptimizationType.FOLLOWUP: DataQAStrategy,
    }

    @classmethod
    def create_strategy(
        cls,
        llm: LLMModel,
        optimization_type: QueryOptimizationType,
    ) -> BaseOptimizationStrategy:
        """创建优化策略实例

        Args:
            llm (LLMModel): llm模型实例
            optimization_type (QueryOptimizationType): 优化类型

        Raises:
            ValueError: 不支持的优化类型

        Returns:
            BaseOptimizationStrategy: 策略实例
        """
        if optimization_type not in cls._strategy_mapping:
            raise ValueError(
                f"Unsupported optimization_type: {optimization_type}. "
                f"Supported types: {list(cls._strategy_mapping.keys())}"
            )
        strategy_class = cls._strategy_mapping[optimization_type]
        return strategy_class(llm, optimization_type)

    @classmethod
    def get_supported_types(cls) -> List[QueryOptimizationType]:
        """获取支持的优化类型列表"""
        return list(cls._strategy_mapping.keys())