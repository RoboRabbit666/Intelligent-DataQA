from enum import Enum


class SearchType(str, Enum):
    # 语义检索
    dense = "dense"
    # 全文检索
    sparse = "sparse"
    # 混合检索,结合了语义和全文检索的优势,通常用于更复杂的查询场景。
    hybrid = "hybrid"