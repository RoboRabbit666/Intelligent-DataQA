# coding: utf-8
import copy
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import re

from app.core.components import mxbai_reranker, sql_kb, tokenizer, minio, embedder, document_kb
from app.config.config import settings
from czce_ai.embedder import BgeM3Embedder
from app.core.components.query_optimizer import (
    DataQAOptimizedQuery,
    QueryOptimizationType,
    QueryOptimizer,
)
from app.models import (
    ChatCompletionChoice,
    ChatReference,
    ChatStep,
    ChatUsage,
    DataQAChatCompletionResponse,
    DataQACompletionRequest,
    RerankerInfo,
)

from czce_ai.llm.chat import LLMChat as LLMModel
from czce_ai.llm.message import Message as ChatMessage
from czce_ai.utils.log import logger

from .entities import WorkflowStepType
from .prompt import dataga_prompt

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

@dataclass
class WorkflowConfig:
    """工作流配置类-统一管理参数、魔法数字等"""
    history_round: int = 1
    follow_up_round: int = 1
    reranking_threshold: float = 0.2
    collection: Optional[str] = "hybrid_sql"
    domain_collection: Optional[str] = 'domain_kl'
    max_table_results: int = 3
    enable_entity_recognition: bool = True
    enable_reranker: bool = True

    def __post_init__(self):
        if self.history_round < self.follow_up_round:
            raise ValueError(
                f"history_round({self.history_round}) must be >= follow_up_round({self.follow_up_round})"
            )

class DataQaWorkflow:
    def __init__(
        self,
        ans_llm: LLMModel,
        ans_thinking_llm: LLMModel,
        query_llm: LLMModel,
        config: Optional[WorkflowConfig] = None,
    ):
        self.ans_client = ans_llm
        self.ans_thinking_client = ans_thinking_llm
        self.query_optimizer = QueryOptimizer(query_llm)
        self.config = config or WorkflowConfig()

    def _create_step(
        self, step_type: WorkflowStepType, number: int, prompt: Any
    ) -> ChatStep:
        """创建工作流步骤"""
        step_names = {
            WorkflowStepType.FOLLOW_UP: "问题追问",
            WorkflowStepType.MODIFY_QUERY: "问题改写",
            WorkflowStepType.ENTITY_RECOGNITION: "问题实体识别",
            WorkflowStepType.LOCATE_TABLE: "表格定位",
            WorkflowStepType.GENERATE_PROMPT: "上下文工程",
            WorkflowStepType.GENERATE_SQL: "SQL生成",
        }
        return ChatStep(
            key=step_type.value,
            name=step_names[step_type],
            number=number,
            prompt=prompt,
            finished=True,
        )

    def _extract_input_messages(
        self, request: DataQACompletionRequest
    ) -> List[ChatMessage]:
        """提取输入信息列表"""
        return request.messages[-self.config.history_round * 2:]

    def entity_recognition(self, query: str):
        """实体识别
        Args:
            query: 本轮问题
        Returns:
            query: 增加实体识别后的query
        """
        if not self.config.enable_entity_recognition:
            return query
        try:
            entity_list = tokenizer.recognize(query)
            for entity in entity_list:
                if entity["id"] != "":
                    # 仅对给定的实体进行识别
                    substring = entity["id"] + "(" + entity["label"] + ")"
                    query = query.replace(entity["text"], substring)
            return query
        except Exception as e:
            logger.error(f"Entity recognition Error: {e}")
            # 实体识别失败返回原查询,不中断流程
            return query

    def modify_query(
        self,
        input_messages: List[ChatMessage],
        enable_follow_up: bool,
    ) -> DataQAOptimizedQuery:
        """问题改写
        Args:
            input_messages (List[ChatMessage]): 输入的消息列表
            enable_follow_up (bool): 是否启用追问功能
        Returns:
            优化后的查询对象
        """
        try:
            input_messages_copy = copy.deepcopy(input_messages)
            optimization_type = (
                QueryOptimizationType.FOLLOWUP
                if enable_follow_up
                else QueryOptimizationType.DATAQA
            )
            optimized_query = self.query_optimizer.generate_optimized_query(
                query=input_messages_copy[-1].content,
                chat_history=input_messages_copy[:-1],
                optimization_type=optimization_type,
            )
            return optimized_query
        except Exception as e:
            logger.error(f"Modify query Error:{e}")
            traceback.print_exc()
            raise e

    # 补充业务知识
    def domain_knowledge_search(
        self, query: str, document_id: Optional[str] = None
    ) -> str:
        """搜索领域知识
        Args:
            query: 查询内容
            document_id: 查询的文档ID
        Returns:
            query+业务知识
        """
        try:
            embedder = BgeM3Embedder(base_url=settings.embedder.base_url, api_key=settings.embedder.api_key)
            query_embedding = embedder.get_embedding(query)
            domain_doc = document_kb.get_by_ids(collection=self.config.domain_collection, doc_id=document_id)
            domains = domain_doc[0].data.content
            sentences = [s.strip() for s in domains.split("\n") if s.strip()]
            sentence_embeddings = [embedder.get_embedding(s) for s in sentences]
            # 计算 query 和每个句子的相似度
            similarities = [cosine_similarity(query_embedding, sent_emb) for sent_emb in sentence_embeddings]
            # 找出最相似的句子
            most_similar_idx = np.argmax(similarities)
            most_similar_sentence = sentences[most_similar_idx]
            similarity_score = similarities[most_similar_idx]
            return most_similar_sentence, similarity_score
        except Exception as e:
            logger.error(f"Domain knowledge search Error:{e}")
            traceback.print_exc()
            raise e

    # 定位表
    def locate_table(
        self, query: str, knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """定位表格
        Args:
            query: 查询内容
            knowledge_base_ids: 查询的知识库ID
        Returns:
            表格信息列表
        """
        try:
            ranked_tables = sql_kb.search(
                self.config.collection,
                query,
                knowledge_ids=knowledge_base_ids,
                top_k=self.config.max_table_results,
                use_reranker=self.config.enable_reranker,
            )
            tables = [
                {
                    "chunk_uuid": table.chunk_id,
                    "table_name": table.data.table_name,
                    "table_info": table.data.table_info,
                    "score": table.reranking_score,
                }
                for table in ranked_tables
            ]
            return tables
        except Exception as e:
            logger.error(f"Locate table Error: {e}")
            return []

    # 生成单表查询的prompt
    def generate_single_table_prompt(self, tables: List[Dict[str, Any]]) -> str:
        """生成单表查询的prompt
        Args:
            tables: 表格信息列表
        Returns:
            表格提示词
        """
        # 从数据库中获取该表的字段信息,该字段信息需要包含完整的字段信息(包括英文名称、中文名称)
        # 用作sql生成的prompt的一部分
        table_info = tables[0].get("table_info", "")
        # 生成prompt
        table_prompt = f"已知如下数据表信息:\n{table_info}\n"
        return table_prompt

    def generate_sql(
        self,
        table_schema: str,
        input_messages: List[ChatMessage],
        thinking: Optional[bool] = False,
    ):
        """生成SQL
        Args:
            table_schema (str): _description_
            input_messages (List[ChatMessage]): _description_
            thinking (Optional[bool], optional): _description_. Defaults to False.
        Returns:
            _type_: _description_
        """
        query = input_messages[-1].content
        content = dataga_prompt.format(table_schema=table_schema, question=query)
        # print(content)
        system_msg = ChatMessage(
            role="system",
            content=content,
        )
        if thinking:
            response = self.ans_thinking_client.invoke(
                messages=[system_msg] + input_messages[:]
            )
        else:
            response = self.ans_client.invoke(messages=[system_msg] + input_messages[:])
        return response

    def _handle_follow_up(
        self,
        optimized_query: DataQAOptimizedQuery,
        request: DataQACompletionRequest,
    ) -> Tuple[Optional[DataQAChatCompletionResponse], DataQAOptimizedQuery]:
        """处理追问逻辑
        Args:
            optimized_query (DataQAOptimizedQuery): 优化后的查询对象
            request (DataQACompletionRequest): 数据问答请求对象
        Returns:
            Tuple[追问响应(如果需要), 更新后的查询对象]
        """
        # 计算追问轮数
        follow_up_num = (
            request.follow_up_num + 1 if not optimized_query.is_sufficient else 0
        )
        # 需要追问且未超过最大追问轮数
        if (
            not optimized_query.is_sufficient
            and follow_up_num <= self.config.follow_up_round
        ):
            step = self._create_step(
                step_type=WorkflowStepType.FOLLOW_UP,
                number=0,
                prompt=optimized_query.rewritten_query,
            )
            choices = [
                ChatCompletionChoice(
                    finish_reason="stop",
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=optimized_query.rewritten_query,
                        reasoning_content=None,
                        is_follow_up=True,
                    ),
                )
            ]
            return (
                DataQAChatCompletionResponse(
                    choices=choices,
                    steps=[step],
                    follow_up_num=follow_up_num,
                ),
                optimized_query,
            )
        elif follow_up_num > self.config.follow_up_round:
            # 超过追问轮数,使用本轮数据问答的会话内容进行检索
            input_messages = self._extract_input_messages(request)
            recent_messages = input_messages[-self.config.follow_up_round * 2:]
            manual_query = " ".join(
                [msg.content for msg in recent_messages if msg.role == "user"]
            )
            return None, DataQAOptimizedQuery(
                original_query=optimized_query.original_query,
                rewritten_query=manual_query or optimized_query.rewritten_query,
                is_sufficient=True,  # 超过轮数后认为查询已足够
            )
        # 不需要追问
        return None, optimized_query

    def do_generate(
        self,
        request: DataQACompletionRequest,
        enable_follow_up: bool = True,
        knowledge_base_ids: Optional[List[str]] = None,
        thinking: Optional[bool] = False,
    ) -> DataQAChatCompletionResponse:
        """生成回答
        Args:
            input_messages (List[ChatMessage]): 输入信息列表
            enable_follow_up (bool): 是否启用追问功能
            knowledge_base_ids (Optional[List[str]]): 知识库ID
            thinking (bool): 是否开启thinking模式, 可选, default: False
        """
        # 提取输入信息
        input_messages = self._extract_input_messages(request)
        
        # Step1: modify_query
        optimized_query = self.modify_query(
            input_messages=input_messages,
            enable_follow_up=enable_follow_up,
        )
        
        # 处理追问逻辑
        if enable_follow_up:
            follow_up_response, optimized_query = self._handle_follow_up(
                optimized_query=optimized_query, request=request
            )
            if follow_up_response:
                return follow_up_response
        
        step1 = self._create_step(
            WorkflowStepType.MODIFY_QUERY, 1, optimized_query.rewritten_query
        )
        
        # Step2: query entity recognition
        query = optimized_query.rewritten_query
        entity_enriched_query = self.entity_recognition(query)
        step2 = self._create_step(
            WorkflowStepType.ENTITY_RECOGNITION, 2, entity_enriched_query
        )
        
        # Step3: locate table
        located_table = self.locate_table(entity_enriched_query, knowledge_base_ids)
        step3 = self._create_step(WorkflowStepType.LOCATE_TABLE, 3, located_table)
        
        # Step4: generate single table prompt
        # TODO 未发现使用位置
        single_table_prompt = self.generate_single_table_prompt(located_table)
        step4 = self._create_step(
            WorkflowStepType.GENERATE_PROMPT, 4, single_table_prompt
        )
        
        # Step5: generate_sql
        # TODO 未修改
        response = self.generate_sql(
            query=entity_enriched_query,
            located_table=located_table,
            thinking=thinking,
        )
        step5 = self.create_step(WorkflowStepType.GENERATE_SQL, 5, response)
        
        usage = ChatUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        
        choices = list(
            map(
                lambda x: ChatCompletionChoice(
                    finish_reason=x.finish_reason,
                    index=x.index,
                    message=ChatMessage(
                        role=x.message.role,
                        content=x.message.content,
                        reasoning_content=x.message.reasoning_content,
                        is_follow_up=False,
                    ),
                ),
                response.choices,
            )
        )
        
        return DataQAChatCompletionResponse(
            id=response.id,
            model=response.model,
            created=response.created,
            choices=choices,
            usage=usage,
            steps=[step1, step2, step3, step4, step5],
        )

# llm:app/core/components.py
# query = "苹果是在哪个交易所上市的"
# dataga = DataQaWorkflow(llm=qwen3_llm, query=query)

# # 测试locate_table函数
# chunk_id, table_name, tabel_score = dataga.locate_table()
# print(chunk_id)
# print(table_name)
# print(tabel_score)

# # 测试generate_single_table_prompt函数
# chunk_id = "1e7fcf17-cac8-5322-912e-b96900beae78"
# prompt = dataga.generate_single_table_prompt(chunk_id)
# analysis_info, sql_code = dataqa.generate_sql_code(prompt, query)
# print(analysis_info)
# print('---------')
# print(sql_code)

# # 测试entity_recgnition函数
# query = '华泰期货在郑商所的持仓量是多少?'
# query = dataqa.entity_recognition(query)
# print(query)

# # 测试domain_knowledge_search函数