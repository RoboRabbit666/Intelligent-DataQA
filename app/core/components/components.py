#!/usr/bin/env python
# coding: utf-8

from app.config.config import settings

from czce_ai.llm.chat import LLMChat
from czce_ai.embedder import BgeM3Embedder
from czce_ai.knowledge import DocumentKnowledge, SQLSchemaKnowledge
from czce_ai.nlp import NLPToolkit
from czce_ai.reranker import MxbaiReranker
from czce_ai.vectordb.minio import MinioClient
from resources import (
    STOP_WORDS_PATH,
    SYNONYM_DICT_PATH,
    USER_DICT_PATH,
    NER_PATTERNs_PATH,
)

tokenizer = NLPToolkit(
    user_dict_path=USER_DICT_PATH,
    syn_dict_path=SYNONYM_DICT_PATH,
    stop_words_path=STOP_WORDS_PATH,
    patterns_path=NER_PATTERNs_PATH,
)

embedder = BgeM3Embedder(
    base_url=settings.embedder.base_url, api_key=settings.embedder.api_key
)
mxbai_reranker = MxbaiReranker(base_url=settings.mxbai_reranker.base_url)

# 初始化Document的知识库
document_kb = DocumentKnowledge(
    tokenizer=tokenizer,
    embedder=embedder,
    uri=settings.milvus.uri,
    reranker=mxbai_reranker,
)

# 初始化SQLSchema的知识库
sql_kb = SQLSchemaKnowledge(
    tokenizer=tokenizer,
    embedder=embedder,
    uri=settings.milvus.uri,
    reranker=mxbai_reranker,
)

# 初始化 qwq_llm
qwq_llm = LLMChat(
    name="QwQ",
    base_url=settings.qwq_llm.base_url,
    model=settings.qwq_llm.model_name,
    api_key=settings.qwq_llm.api_key,
    temperature=settings.qwq_llm.temperature,
)

# 初始化 qwen3_llm
qwen3_thinking_llm = LLMChat(
    name="Qwen3-Thinking",
    base_url=settings.qwen3_llm.base_url,
    model=settings.qwen3_llm.model_name,
    api_key=settings.qwen3_llm.api_key,
    temperature=settings.qwen3_llm.temperature,
    # top_k=20,
    top_p=0.8,
    max_tokens=8192,
    presence_penalty=1.5,
)

# 不带思考的qwen3
qwen3_llm = LLMChat(
    name="Qwen3",
    base_url=settings.qwen3_llm.base_url,
    model=settings.qwen3_llm.model_name,
    api_key=settings.qwen3_llm.api_key,
    temperature=settings.qwen3_llm.temperature,
    # top_k=20,
    top_p=0.8,
    max_tokens=8192,
    presence_penalty=1.5,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

# 初始化 minIO
minio = MinioClient(
    endpoint=settings.minio.endpoint,
    access_key=settings.minio.access_key,
    secret_key=settings.minio.secret_key,
    secure=False,
)