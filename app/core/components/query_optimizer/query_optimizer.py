#coding utf-8
from typing import Dict, List, Optional

from czce_ai.llm.chat import LLMChat as LLMModel
from czce_ai.llm.message import Message as ChatMessage
from czce_ai.utils.log import logger
from .entities import BaseOptimizedQuery, QueryOptimizationType
from .strategy import (
    BaseOptimizationStrategy,
    DataQAStrategy,
    OptimizationStrategyFactory,
    RAGStrategy,
)


class QueryOptimizer:
    """优化器入口类"""

    def __init__(self, llm: LLMModel):
        self.llm = llm
        # 策略缓存
        self._strategies_cached: Dict[
            QueryOptimizationType, BaseOptimizationStrategy
        ] = {}

    def _get_strategy(
        self, optimization_type: QueryOptimizationType
    ) -> BaseOptimizationStrategy:
        """获取策略"""
        if optimization_type not in self._strategies_cached:
            try:
                self._strategies_cached[optimization_type] = (
                    OptimizationStrategyFactory.create_strategy(
                        self.llm, optimization_type
                    )
                )
            except Exception as e:
                raise e
        return self._strategies_cached[optimization_type]

    def generate_optimized_query(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        optimization_type: QueryOptimizationType = QueryOptimizationType.REWRITE,
    ) -> BaseOptimizedQuery:
        """生成优化查询

        Args:
            query (str): 原始查询
            chat_history (Optional[List[ChatMessage]], optional): 聊天历史. Defaults to None.
            optimization_type (QueryOptimizationType, optional): 优化类型. Defaults to QueryOptimizationType.REWRITE.

        Raises:
            e: _description_

        Returns:
            BaseOptimizedQuery: 优化后的查询
        """
        try:
            strategy = self._get_strategy(optimization_type)
            return strategy.generate(query=query, chat_history=chat_history)
        except Exception as e:
            logger.error(f"Query optimization error: {e}")
            raise e

    async def agenerate_optimized_query(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        optimization_type: QueryOptimizationType = QueryOptimizationType.REWRITE,
    ) -> BaseOptimizedQuery:
        """与同步调用相同"""
        try:
            strategy = self._get_strategy(optimization_type)
            return await strategy.agenerate(query=query, chat_history=chat_history)
        except Exception as e:
            logger.error(f"Async query optimization error: {e}")
            raise e