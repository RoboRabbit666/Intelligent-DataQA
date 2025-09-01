# coding=utf-8
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from czce_ai.embedder import Embedder


@dataclass
class DocTable:
    content: str
    title: str = None
    embedding_content: str = None


@dataclass
class Document:
    """Dataclass for managing a document"""

    content: Optional[Union[str, Dict[Any, Any], List[Any]]] = None
    tables: Optional[List[DocTable]] = None
    id: Optional[str] = None
    name: Optional[str] = None
    meta_data: Dict[str, Any] = field(default_factory=dict)
    kwd_tks: Optional[List[str]] = None
    usage: Optional[Dict[str, Any]] = None
    reranking_score: Optional[float] = None


class ChunkType(Enum):
    DOCUMENT = "document"
    SQL_SCHEMA = "sql_schema"
    QA_PAIR = "qa_pair"
    API_DATA = "api_data"
    BUSINESS_INFO = "business_info"


@dataclass
class BaseChunkData:
    """所有chunk类型的base class"""
    pass


@dataclass
class DocumentChunkData(BaseChunkData):
    """document chunk data - 使用通用的content和embedding字段"""
    content: str
    chunk_type: ChunkType = field(default=ChunkType.DOCUMENT, init=False)
    title: Optional[str] = None
    embedding: Optional[List[float]] = None
    kwd_tks: Optional[List[str]] = None
    embed_rerank_content: Optional[str] = None  # 用于embedding或者rerank的原始內容

    def embed(self, embedder: Embedder) -> None:
        self.embedding = embedder.get_embedding(
            self.embed_rerank_content or self.content
        )

    async def aembed(self, embedder: Embedder) -> None:
        self.embedding = await embedder.aget_embedding(
            self.embed_rerank_content or self.content
        )


@dataclass
class SQLSchemaChunkData(BaseChunkData):
    """SQL Schema chunk data 自定义多个embedding字段"""
    table_name: Optional[str] = None
    table_schema: Optional[str] = None
    table_info: Optional[str] = None
    chunk_type: ChunkType = field(default=ChunkType.SQL_SCHEMA, init=False)
    qa_scenarios: Optional[str] = None
    table_schema_embedding: Optional[List[float]] = None
    qa_scenarios_embedding: Optional[List[float]] = None
    embed_rerank_content: Optional[str] = None  # 用于embedding或者rerank的原始内容

    def embed(self, embedder: Embedder) -> None:
        if self.table_schema:
            self.table_schema_embedding = embedder.get_embedding(self.table_schema)
        if self.qa_scenarios:
            self.qa_scenarios_embedding = embedder.get_embedding(self.qa_scenarios)

    async def aembed(self, embedder: Embedder) -> None:
        if self.table_schema:
            self.table_schema_embedding = await embedder.aget_embedding(
                self.table_schema
            )
        if self.qa_scenarios:
            self.qa_scenarios_embedding = await embedder.aget_embedding(
                self.qa_scenarios
            )


@dataclass
class QAPairChunkData(BaseChunkData):
    """QA类型知识库的数据chunk"""
    question: Optional[str] = None
    answer: Optional[str] = None
    chunk_type: ChunkType = field(default=ChunkType.QA_PAIR, init=False)
    question_embedding: Optional[List[float]] = None
    embed_rerank_content: Optional[str] = None  # 用于embedding或者rerank的原始内容

    def embed(self, embedder: Embedder) -> None:
        if self.question:
            self.question_embedding = embedder.get_embedding(
                self.embed_rerank_content or self.question
            )

    async def aembed(self, embedder: Embedder) -> None:
        if self.question:
            self.question_embedding = await embedder.aget_embedding(
                self.embed_rerank_content or self.question
            )


@dataclass
class ApiChunkData(BaseChunkData):
    """API类型知识库的数据chunk"""
    api_id: Optional[str] = None  # Assuming there's an api_id field in the entity. Adjust accordingly.
    api_name: Optional[str] = None
    api_description: Optional[str] = None
    api_request: Optional[str] = None
    api_response: Optional[str] = None
    api_info: Optional[str] = None
    api_info_embedding: Optional[List[float]] = None
    chunk_type: ChunkType = field(default=ChunkType.API_DATA, init=False)
    embed_rerank_content: Optional[str] = None

    def embed(self, embedder: Embedder) -> None:
        if self.api_info:
            self.api_info_embedding = embedder.get_embedding(
                self.embed_rerank_content or self.api_info
            )

    async def aembed(self, embedder: Embedder) -> None:
        if self.api_info:
            self.api_info_embedding = await embedder.aget_embedding(
                self.embed_rerank_content or self.api_info
            )


ChunkDataType = Union[DocumentChunkData, SQLSchemaChunkData, QAPairChunkData, ApiChunkData]


@dataclass
class BusinessInfoChunkData(BaseChunkData):
    """业务知识类型的知识库的数据chunk"""
    business_desc: Optional[str] = None
    info_type: Optional[str] = None
    chunk_type: ChunkType = field(default=ChunkType.BUSINESS_INFO, init=False)
    business_desc_embedding: Optional[List[float]] = None
    embed_rerank_content: Optional[str] = None  # 用于embedding rerank的原始内容

    def embed(self, embedder: Embedder) -> None:
        if self.business_desc:
            self.business_desc_embedding = embedder.get_embedding(
                self.embed_rerank_content or self.business_desc
            )

    async def aembed(self, embedder: Embedder) -> None:
        if self.business_desc:
            self.business_desc_embedding = await embedder.aget_embedding(
                self.embed_rerank_content or self.business_desc
            )


ChunkDataType = Union[
    DocumentChunkData, SQLSchemaChunkData, QAPairChunkData, BusinessInfoChunkData
]


@dataclass
class Chunk:
    """管理通用元数据"""
    chunk_id: str
    data: ChunkDataType
    doc_id: Optional[str] = None
    meta_data: Dict[str, Any] = field(default_factory=dict)
    embedder: Optional[Embedder] = None
    knowledge_id: Optional[str] = None
    reranking_score: Optional[float] = None

    def embed(self, embedder: Optional[Embedder] = None) -> None:
        self.data.embed(embedder or self.embedder)

    async def aembed(self, embedder: Optional[Embedder] = None) -> None:
        await self.data.aembed(embedder or self.embedder)