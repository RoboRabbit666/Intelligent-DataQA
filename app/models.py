# coding=utf-8
import time
import uuid
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from czce_ai.document.splitter import SplitterType
from czce_ai.llm.message import Message as ChatMessage


class SplitterInfo(BaseModel):
    split_type: SplitterType = SplitterType.RECURSIVE
    chunk_size: int = 1024
    separators: Optional[Union[List[Tuple], List[str], str]] = None

class KBType(str, Enum):
    """知识库类型"""
    # 文档知识库
    DOCUMENT = "document"
    # SQL Schema知识库
    SQL_SCHEMA = "sql_schema"

class InsertDocumentKB(BaseModel):
    """向文档类型知识库中插入文档"""
    kb_uuid: str
    kb_collection: str
    doc_uuid: str
    doc_bucket: str
    doc_original_name: str
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
    # 是否开启深度思考
    thinking: Optional[bool] = True

class KnogCompletionRequest(ChatCompletionRequest):
    """知识问答请求"""
    knowledge_base_ids: List[str]

# 标准 OpenAI Usage 模型
class ChatUsage(BaseModel):
    prompt_tokens: int = Field(..., description="提示词的token数量")
    completion_tokens: int = Field(..., description="生成词的token数量")
    total_tokens: int = Field(..., description="总的token数量")

# 用于RAG 引用的自定义模型
class ChatReference(BaseModel):
    chunk_uuid: str = Field(..., description="引用源文档中的块的唯一标识符")
    doc_uuid: str = Field(..., description="引用源文档的唯一标识符")
    title: Optional[str] = Field(None, description="引用源文档的标题")
    content: Optional[str] = Field(None, description="引用的原文片段内容")
    reranking_score: Optional[float] = Field(
        None, description="重排序分数,用于排序引用结果"
    )

# 步骤
class ChatStep(BaseModel):
    key: str = Field(..., description="步骤key")
    name: str = Field(..., description="步骤名称")
    number: int = Field(..., description="步骤编号")
    references: Optional[List[ChatReference]] = Field(
        None, description="步骤中的引用信息"
    )
    prompt: Optional[Any] = Field(None, description="步骤中的提示信息")
    finished: Optional[bool] = Field(False, description="步骤是否完成")
    object: str = Field("chat.completion.step", description="对象类型")
    created: int = Field(default_factory=lambda: int(time.time()), description="创建时间戳")

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
    steps: Optional[List[ChatStep]] = Field(
        None, description="生成回复过程中涉及的步骤信息列表"
    )

# --- DataQA Specific Models (reconstructed from workflow.py) ---

class RerankerInfo(BaseModel):
    """重排器信息"""
    provider: Optional[str] = None
    model: Optional[str] = None

class DataQaCompletionChoice(BaseModel):
    """DataQA 工作流中的回复选项（通常不直接使用）"""
    index: int = Field(..., description="回复选项的索引")
    message: ChatMessage = Field(..., description="回复消息")
    finish_reason: Optional[str] = Field(None, description="回复结束的原因")

class DataQACompletionRequest(BaseModel):
    """DataQA 工作流的补全请求（包含追问计数）"""
    messages: List[ChatMessage]
    model: str = "QWen3"
    stream: Optional[bool] = False
    thinking: Optional[bool] = True
    # 用于控制和记录追问轮次
    follow_up_num: int = 0


class DataQAChatCompletionResponse(BaseModel):
    """DataQA 工作流响应（包含追问计数）"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., description="使用的模型名称")
    # 为兼容当前实现，复用 ChatCompletionChoice
    choices: List[ChatCompletionChoice] = Field(..., description="回复选项列表")
    usage: Optional[ChatUsage] = Field(None, description="使用情况统计")
    steps: Optional[List[ChatStep]] = Field(
        None, description="生成回复过程中涉及的步骤信息列表"
    )
    # 便于前端和测试读取追问次数
    follow_up_num: int = 0