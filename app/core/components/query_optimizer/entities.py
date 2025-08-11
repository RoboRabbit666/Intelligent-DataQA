#coding=utf-8
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class QueryOptimizationType(str, Enum):
    #退阶修改
    BACK = "back"
    # 问题分解
    DECOMPOSE = "decompose"
    # 问题重写
    REWRITE = "rewrite"
    #三种统一优化
    UNIFIED = "unified"
    # 数据查询问题改写
    DATAQA = "dataqa"
    # 数据查询问题追问
    FOLLOWUP = "follow-up"


class BaseOptimizedQuery(BaseModel):

    original_query: str = Field(description="输入的原始查询")
    rewritten_query: Optional[str] = Field(
        default=None, description="原始查询优化后的表述方式,更适合信息检索的优化问题"
    )


class RAGOptimizedQuery(BaseOptimizedQuery):

    step_back_query: Optional[str] = Field(
        default=None, description="更宏观的背景性问题"
    )
    sub_queries: Optional[List[str]] = Field(
        default=None, description="拆解的1~4个具体子问题"
    )


class DataQAOptimizedQuery(BaseOptimizedQuery):
    is_sufficient: Optional[bool] = Field(
        default=True, description="问题是否完整(False表示补充信息)"
    )