#!/usr/bin/env python
# coding: utf-8

from resources import USER_DICT_PATH, SYNONYM_DICT_PATH, STOP_WORDS_PATH
from czce.ai.tokenizer import JiebaTokenizer
from czce.ai.embedder import BgeM3Embedder
from czce.ai.reranker import MxbaiReranker
from czce.ai.vectordb.milvus import Milvus
from czce.ai.vectordb.minio import MinioClient
from app.config import settings
from czce.ai.llm.chat import LLMChat

tokenizer = JiebaTokenizer(
    user_dict_path=USER_DICT_PATH,
    synonym_dict_path=SYNONYM_DICT_PATH,
    stop_words_path=STOP_WORDS_PATH,
)

embedder = BgeM3Embedder(
    base_url=settings.embedder.base_url, api_key=settings.embedder.api_key
)
mxbai_reranker = MxbaiReranker(base_url=settings.mxbai_reranker.base_url)

# 初始化milvus
milvus = Milvus(
    tokenizer=tokenizer,
    embedder=embedder,
    uri=settings.milvus.uri,
    reranker=mxbai_reranker,
)

# 初始化 qwq_llm
qwq_llm = LLMChat(
    name="qwq",
    base_url=settings.qwq_llm.base_url,
    model=settings.qwq_llm.model_name,
    api_key=settings.qwq_llm.api_key,
    temperature=settings.qwq_llm.temperature,
)

# 初始化 qwen3_llm
qwen3_thinking_llm = LLMChat(
    name="qwen3-thinking",
    base_url=settings.qwen3_llm.base_url,
    model=settings.qwen3_llm.model_name,
    api_key=settings.qwen3_llm.api_key,
    temperature=settings.qwen3_llm.temperature,
    top_k=20,
    top_p=0.8,
    max_tokens=8192,
    presence_penalty=1.5,
)
# 不带思考的qwen3
qwen3_llm = LLMChat(
    name="qwen3",
    base_url=settings.qwen3_llm.base_url,
    model=settings.qwen3_llm.model_name,
    api_key=settings.qwen3_llm.api_key,
    temperature=settings.qwen3_llm.temperature,
    top_k=20,
    top_p=0.8,
    max_tokens=8192,
    presence_penalty=1.5,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

# 初始化 minio
minio = MinioClient(
    endpoint=settings.minio.endpoint,
    access_key=settings.minio.access_key,
    secret_key=settings.minio.secret_key,
    secure=False,
)