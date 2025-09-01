#coding=utf-8
import time
import uuid
from enum import Enum
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from czce_ai.document.splitter import SplitterType
from czce_ai.llm.message import Message as ChatMessage

class SplitterInfo(BaseModel):
    split_type: SplitterType = SplitterType.Recursive
    chunk_size: int = 1024
    separators: Optional[Union[List[Tuple], List[str], str]] = None

class RerankerInfo(BaseModel):
    #是否启用重排与检索分数融合
    score_fusion: bool = True
    #重排分数丢弃阈值
    threshold: float = 0.1
    # 分数融合系数
    alpha: float = 0.3

class KBType(str, Enum):
    """知识库类型"""
    #文档知识库
    DOCUMENT = "document"
    # SQL Schema知识库
    SQL_SCHEMA = "sql_schema"

class InsertDocumentKB(BaseModel):
    """向文档类型知识库中插入文档"""
    kb_uuid: str
    kb_collection: str
    doc_uuid: str
    doc_original_name: str
    doc_bucket: str
    splitter_info: Optional[SplitterInfo] = None

class DocumentChunkPublic(BaseModel):
    """数据库文档分块信息"""
    uuid: str
    doc_uuid: str
    content: str
    kwd_tks: Optional[List[str]] = None

class CreateKBCollection(BaseModel):
    """创建知识库collection"""
    kb_type: KBType
    collection: str

class KBCollectionPublic(BaseModel):
    """知识库collection返回"""
    kb_type: KBType
    collection: str

class ChatCompletionRequest(BaseModel):
    """聊天补全请求"""
    messages: List[ChatMessage]
    model: str = "QWen3"
    stream: Optional[bool] = False
    #是否开启深度思考
    thinking: Optional[bool] = True

class KnoqCompletionRequest(ChatCompletionRequest):
    """知识问答请求"""
    knowledge_base_ids: List[str]
    use_reranker: bool = False
    reranker_info: Optional[RerankerInfo] = None

class SummaryCompletionRequest(ChatCompletionRequest):
    """总结请求"""
    doc_uuid: Optional[str] = None
    doc_bucket: Optional[str] = None
    doc_original_name: Optional[str] = None
    summary_length: Optional[int] = None

#标准 OpenAI Usage 模型
class ChatUsage(BaseModel):
    prompt_tokens: int = Field(..., description="提示词的token数量")
    completion_tokens: int = Field(..., description="生成词的token数量")
    total_tokens: int = Field(..., description="总的token数量")

# 用于 RAG 引用的自定义模型
class ChatReference(BaseModel):
    chunk_uuid: str = Field(..., description="引用源文档中的块的唯一标识符")
    doc_uuid: str = Field(..., description="引用源文档的唯一标识符")
    title: Optional[str] = Field(None, description="引用源文档的标题")
    content: Optional[str] = Field(None, description="引用的原文片段内容")
    reranking_score: Optional[float] = Field(None, description="重排序分数,用于排序引用结果")

#步骤
class ChatStep(BaseModel):
    key: str = Field(..., description="步骤key")
    name: str = Field(..., description="步骤名称")
    number: int = Field(..., description="步骤编号")
    references: Optional[List[ChatReference]] = Field(None, description="步骤中的引用信息")
    prompt: Optional[str] = Field(None, description="步骤中的提示信息")
    finished: Optional[bool] = Field(False, description="步骤是否完成")
    object: str = Field("chat.completion.step", description="对象类型")
    created: int = Field(default_factory=lambda: int(time.time()))

class ChatCompletionChoice(BaseModel):
    index: int = Field(..., description="回复选项的索引")
    message: ChatMessage = Field(..., description="回复消息")
    finish_reason: Optional[str] = Field(None, description="回复结束的原因")

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., description="使用的模型名称")
    choices: List[ChatCompletionChoice] = Field(..., description="回复选项列表")
    usage: Optional[ChatUsage] = Field(None, description="使用情况统计")
    steps: Optional[List[ChatStep]] = Field(None, description="生成回复过程中涉及的步骤信息列表")

class DocInspectionCompletionRequest(BaseModel):
    """文档审核请求"""
    doc_uuid: Optional[str] = None
    doc_bucket: Optional[str] = None
    doc_original_name: Optional[str] = None
    stream: Optional[bool] = False
    thinking: Optional[bool] = False

class ChunkModify(BaseModel):
    """文档审核单chunk修改信息"""
    origin: str
    advise: str
    revise_para: str

class DocInspectionResponse(BaseModel):
    """文档审核返回"""
    modifies: List[ChunkModify]
    bucket_name: str
    object_name: str

class DataQACompletionRequest(ChatCompletionRequest):
    """数据问答请求"""
    knowledge_base_ids: List[str]
    use_reranker: bool = True
    follow_up_num: Optional[int] = Field(default=0, description="记录数据问答中,问题追问的轮数")
    reranker_info: Optional[RerankerInfo] = None

class DataQAChatCompletionResponse(ChatCompletionResponse):
    """数据问答返回"""
    follow_up_num: Optional[int] = Field(default=0, description="记录数据问答中,问题追问的轮数")