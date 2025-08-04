# coding=utf-8
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from czce_ai.document import Chunk

@dataclass
class Reranker(ABC):
    base_url: str
    model: str
    batch_size: int = 32

    @abstractmethod
    def prepare_inputs(self, query: str, documents: List[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def invoke(self, data: Union[List[str], Dict[str, Any]]):
        """获取rerank分数"""

        raise NotImplementedError