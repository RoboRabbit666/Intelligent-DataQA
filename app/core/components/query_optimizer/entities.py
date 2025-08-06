# coding=utf-8
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

class QueryOptimizationType(str, Enum):
    NORMAL = "normal"
    DECOMPOSE = "decompose"
    DATAQA = "dataqa"

# 用于Query优化的自定义模型
class OptimizedQuery(BaseModel):
    step_back_query: Optional[str] = None
    sub_queries: Optional[List[str]] = None
    rewritten_query: str