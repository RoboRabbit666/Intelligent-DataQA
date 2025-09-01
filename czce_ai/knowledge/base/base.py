# coding=utf-8
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pymilvus import AsyncMilvusClient, MilvusClient

from czce_ai.document import BaseChunkData, Chunk
from czce_ai.embedder import Embedder
from czce_ai.nlp import NLPToolkit
from czce_ai.reranker import Reranker

from resources import KNOWLEDGE_CONF

from ..entities import RankerType, SearchType
from .collection_manager import CollectionManager
from .config import ConfigManager
from .data_service import DataService
from .search_service import SearchService


class BaseKnowledge(ABC):
    """基础知识库类- 使用组合模式整合各个服务组件"""

    # 定义知识库类型,类属性
    collection_type: str
    config_path: Path = KNOWLEDGE_CONF

    def __init__(
        self,
        tokenizer: NLPToolkit,
        embedder: Embedder,
        uri: str,
        reranker: Optional[Reranker] = None,
        token: Optional[str] = None,
    ):
        """初始化知识库管理器

        Args:
            tokenizer:分词器实例
            embedder:嵌入模型实例
            uri: Milvus连接URI
            reranker:重排序模型(可选)
            token: Milvus认证token
        """
        if not hasattr(self, "collection_type") or not self.collection_type:
            raise ValueError("子类必须定义 `collection_type`")

        # 初始化基础组件
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.reranker = reranker

        # 加载配置
        self.config_manager = ConfigManager(self.config_path)
        self.config = self.config_manager.load_config(
            self.collection_type, self.config_path
        )
        # 初始化Milvus客户端
        self.client = MilvusClient(uri=uri, token=token)
        self.async_client = AsyncMilvusClient(uri=uri, token=token)
        self._init_services()

    def _init_services(self):
        """初始化各个服务组件"""
        self.collection_manager = CollectionManager(
            client=self.client,
            collection_type=self.collection_type,
            config_manager=self.config_manager,
        )
        self.search_service = SearchService(
            self.client, self.async_client, self.embedder, self.tokenizer, self.reranker
        )
        self.data_service = DataService(self.client, self.async_client)

    def create_collection(self, collection: str, force: bool = False):
        """创建集合

        Args:
            collection: Collection 名称
            force:是否强制重建
        """
        self.collection_manager.create_collection(collection, self.config, force)

    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        limit: int = 20,
        search_type: SearchType = SearchType.hybrid,
        ranker_type: RankerType = RankerType.WEIGHTED,
        dense_weight: float = 0.6,
        use_reranker: bool = True,
        score_fusion: bool = True,
        rrf_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        dense_sub_weights: Optional[List[float]] = None,
        sparse_sub_weights: Optional[List[float]] = None,
        knowledge_ids: Optional[List[str]] = None,
        filters: Optional[str] = None,
        post_process: Optional[Callable[[Chunk], Chunk]] = None,
    ) -> List[Chunk]:
        """在向量数据库中检索chunk数据

        Args:
            collection:集合名称
            query: 搜索的query
            top_k:返回的结果数量
            limit:候选结果数量
            search_type: 检索类型,全文检索、向量检索和混合检索
            dense_weight: 向量检索权重
            use_reranker:是否使用重排序
            rerank_top_k: 参与重排序的检索结果数量
            dense_sub_weights: 多路向量检索权重
            sparse_sub_weights: 多路全文检索权重
            knowledge_ids:检索知识库的范围
            filters:过滤条件
            post_process:后处理函数

        Returns:
            List[Chunk]:搜索结果
        """
        return self.search_service.search(
            collection=collection,
            query=query,
            config=self.config,
            chunk_data_constructor=self._convert_record_to_chunk,
            top_k=top_k,
            limit=limit,
            search_type=search_type,
            ranker_type=ranker_type,
            dense_weight=dense_weight,
            use_reranker=use_reranker,
            score_fusion=score_fusion,
            rrf_k=rrf_k,
            rerank_top_k=rerank_top_k,
            dense_sub_weights=dense_sub_weights,
            sparse_sub_weights=sparse_sub_weights,
            knowledge_ids=knowledge_ids,
            filters=filters,
            post_process=post_process,
        )

    async def asearch(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        limit: int = 20,
        search_type: SearchType = SearchType.hybrid,
        ranker_type: RankerType = RankerType.WEIGHTED,
        dense_weight: float = 0.6,
        use_reranker: bool = True,
        score_fusion: bool = True,
        rrf_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        dense_sub_weights: Optional[List[float]] = None,
        sparse_sub_weights: Optional[List[float]] = None,
        knowledge_ids: Optional[List[str]] = None,
        filters: Optional[str] = None,
        post_process: Optional[Callable[[Chunk], Chunk]] = None,
    ) -> List[Chunk]:
        """异步搜索"""
        return await self.search_service.asearch(
            collection=collection,
            query=query,
            config=self.config,
            chunk_data_constructor=self._convert_record_to_chunk,
            top_k=top_k,
            limit=limit,
            search_type=search_type,
            ranker_type=ranker_type,
            dense_weight=dense_weight,
            use_reranker=use_reranker,
            score_fusion=score_fusion,
            rrf_k=rrf_k,
            rerank_top_k=rerank_top_k,
            dense_sub_weights=dense_sub_weights,
            sparse_sub_weights=sparse_sub_weights,
            knowledge_ids=knowledge_ids,
            filters=filters,
            post_process=post_process,
        )

    def insert(
        self,
        collection: str,
        chunks: List[Chunk],
        batch_size: int = 32,
        max_workers: int = 4,
    ) -> None:
        """插入数据
        语义上表示新增数据, 但底层会智能处理knowledge_ids合并。
        如果chunk_id已存在, 会合并knowledge_ids而不是覆盖。
        Args:
            collection:集合名称
            chunks:需要插入的chunk
            batch_size:每次插入的数量
            max_workers: 最大工作线程数
        """
        self.data_service.insert(
            collection, chunks, self._convert_chunk_to_record, batch_size, max_workers
        )

    def upsert(
        self,
        collection: str,
        chunks: List[Chunk],
        batch_size: int = 32,
        max_workers: int = 4,
    ) -> None:
        """更新插入数据
        语义上表示更新或插入数据,底层会智能处理knowledge_ids合并。
        如果chunk_id已存在,会合并knowledge_ids 如果不存在,则新增记录。
        Args:
            collection: 集合名称
            chunks:需要插入的chunk
            batch_size:每次插入的数量
            max_workers: 最大工作线程数
        """
        self.data_service.upsert(
            collection, chunks, self._convert_chunk_to_record, batch_size, max_workers
        )

    async def ainsert(self, collection: str, chunks: List[Chunk], batch_size: int = 32):
        """异步插入数据
        语义上表示新增数据
        """
        await self.data_service.ainsert(
            collection, chunks, self._convert_chunk_to_record, batch_size
        )

    async def aupsert(self, collection: str, chunks: List[Chunk], batch_size: int = 32):
        """异步更新插入数据
        语义上表示更新或插入数据
        """
        await self.data_service.aupsert(
            collection, chunks, self._convert_chunk_to_record, batch_size
        )

    def delete_with_ids(
        self,
        collection: str,
        chunk_id: Optional[Union[str, List[str]]] = None,
        doc_id: Optional[Union[str, List[str]]] = None,
        knowledge_ids: Optional[List[str]] = None,
    ):
        """根据条件删除数据"""
        self.data_service.delete_with_ids(collection, chunk_id, doc_id, knowledge_ids)

    async def adelete_with_ids(
        self,
        collection: str,
        chunk_id: Optional[Union[str, List[str]]] = None,
        doc_id: Optional[Union[str, List[str]]] = None,
        knowledge_ids: Optional[List[str]] = None,
    ) -> None:
        """异步根据条件删除数据"""
        await self.data_service.adelete_with_ids(
            collection, chunk_id, doc_id, knowledge_ids
        )

    def get_by_ids(
        self,
        collection: str,
        chunk_id: Optional[Union[str, List[str]]] = None,
        doc_id: Optional[Union[str, List[str]]] = None,
        knowledge_ids: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> Optional[List[Chunk]]:
        """根据条件查询数据"""
        if output_fields is None:
            output_fields = self.config["search_params"]["output_fields"]
        return self.data_service.get_by_ids(
            collection,
            self._convert_record_to_chunk,
            chunk_id,
            doc_id,
            knowledge_ids,
            output_fields,
        )

    def doc_id_exists(self, collection: str, doc_id: str) -> bool:
        """检查文档是否存在

        Args:
            collection: 集合名称
            doc_id: 文档ID

        Returns:
            bool:文档是否存在
        """
        return self.data_service.doc_id_exists(collection, doc_id)

    def get_count(self, collection: str) -> int:
        """获取集合中的数据数量

        Args:
            collection:集合名称

        Returns:
            int:数据数量
        """
        stats = self.collection_manager.get_collection_stats(collection)
        return int(stats["row_count"])

    @abstractmethod
    def _convert_record_to_chunk(self, entity: Dict) -> BaseChunkData:
        """根据类型构造对应的ChunkData对象

        Args:
            entity:实体数据

        Returns:
            BaseChunkData: 构造的ChunkData对象
        """
        pass

    @abstractmethod
    def _convert_chunk_to_record(self, chunk: Chunk) -> Dict[str, Any]:
        """转化为schema对应的数据

        Args:
            chunk: chunk数据

        Returns:
            Dict[str, Any]: schema格式数据
        """
        pass