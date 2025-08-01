from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class Embedder:
    """"Base class for managing embedders"""

    dimensions: Optional[int] = 1536

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入表示"""
        raise NotImplementedError

    # 获取文本的嵌入表示及其使用情况 （如token数量等， eg{'prompt_tokens': 4, 'total_tokens': 4, 'completion_tokens': 0, 'prompt_tokens_details': None}）
    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        """获取文本的嵌入向量表示及其使用情况"""
        raise NotImplementedError