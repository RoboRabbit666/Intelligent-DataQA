# coding: utf-8
import copy
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.config.config import settings
from app.core.components import (
    business_info_kb,
    document_kb,
    qa_pair_kb,
    sql_kb,
    tokenizer,
)
from app.core.components.query_optimizer import (
    DataQAOptimizedQuery,
    QueryOptimizationType,
    QueryOptimizer,
)
from app.models import (
    ChatCompletionChoice,
    ChatStep,
    ChatUsage,
    DataQAChatCompletionResponse,
    DataQACompletionRequest,
    RerankerInfo,
)
from czce_ai.document import Chunk
from czce_ai.knowledge import SearchType
from czce_ai.llm.chat import LLMChat as LLMModel
from czce_ai.llm.message import Message as ChatMessage
from czce_ai.utils.log import logger

from .entities import WorkflowStepType
from .prompt import dataqa_prompt


@dataclass
class WorkflowConfig:
    """工作流配置类-统一管理参数、魔法数字等"""

    history_round: int = 3
    follow_up_round: int = 1
    reranking_threshold: float = 0.2
    sql_schema_collection: Optional[str] = "hybrid_sql"
    domain_collection: Optional[str] = "domain_k1"
    api_collection: Optional[str] = "api_k1"
    qa_collection: Optional[str] = "sql_qa"
    max_table_results: int = 3
    enable_entity_recognition: bool = True
    enable_reranker: bool = True
    # FQA相关设置
    enable_fqa_answer: bool = True  # 是否启用FQA作为SQL
    fqa_answer_threshold: float = 0.9  # FQA分数作为结果的阈值
    fqa_sample_num: int = 3  # fqa SQL 样例个数

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
            WorkflowStepType.LOCATE_API: "API定位",
            WorkflowStepType.SEARCH_FAQ: "语义搜索FAQ",
            WorkflowStepType.LOCATE_TABLE: "表格定位",
            WorkflowStepType.GENERATE_SQL: "SQL生成",
        }
        # 转换prompt为字符串(如果不是是字符串的话)
        if not isinstance(prompt, str):
            if isinstance(prompt, list):
                # 对于列表,转换为JSON字符串
                prompt = json.dumps(prompt, ensure_ascii=False, indent=2)
            elif isinstance(prompt, dict):
                # 对于字典,转换为JSON字符串
                prompt = json.dumps(prompt, ensure_ascii=False, indent=2)
            else:
                # 对于其他类型,直接转换为字符串
                prompt = str(prompt)
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
        return request.messages[-self.config.history_round * 2 :]

    def modify_query(
        self,
        input_messages: List[ChatMessage],
        enable_follow_up: bool,
    ) -> DataQAOptimizedQuery:
        """问题改写
        Args:
            input_messages (List[ChatMessage]):输入的消息列表
            enable_follow_up(bool):是否启用追问功能
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
            # 生成优化后的查询对象
            last_user_msg = None
            last_index = None
            for index, message in enumerate(input_messages_copy):
                if message.role == "user":
                    last_user_msg = message
                    last_index = index
            optimized_query = self.query_optimizer.generate(
                query=last_user_msg.content,
                chat_history=input_messages_copy[: last_index + 1],
                optimization_type=optimization_type,
            )
            return optimized_query
        except Exception as e:
            logger.error(f"Modify query Error: {e}")
            traceback.print_exc()
            raise e

    def entity_recognition(self, query: str):
        """实体识别
        Args:
            query:本轮问题
        Returns:
            query:增加实体识别后的query
        """
        if not self.config.enable_entity_recognition:
            return query
        try:
            enhanced_query = query
            entity_list = tokenizer.recognize(query)
            for entity in entity_list:
                if entity["id"] != "" and entity["text"] in enhanced_query:
                    if entity["label"] == "合约":
                        # 标准化合约代码 - 将所有分隔符转换为标准模式
                        normalized_code = re.sub(
                            r"[-_\.\s/]+", "", entity["text"].upper()
                        )
                        substring = f"({normalized_code})(合约)"
                    else:
                        # 其他实体正确格式化
                        substring = f"({entity['text']})({entity['label']})"
                    enhanced_query = enhanced_query.replace(
                        entity["text"], substring, 1
                    )
            logger.info(f"实体识别完成: {query} -> {enhanced_query}")
            return enhanced_query
        except Exception as e:
            logger.error(f"实体识别失败:{e}")
            # 实体识别失败返回原查询,不中断流程
            return query

    def business_info_search(
        self, query: str, top_k: int = 2, knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Chunk]:
        """搜索业务知识
        Args:
            query: 查询内容
            top_k:返回相关信息的个数
        Returns:
            相关业务信息
        """
        try:
            ranked_qas = business_info_kb.search(
                collection=self.config.domain_collection,
                query=query,
                search_type=SearchType.dense,
                top_k=self.config.fqa_sample_num,
                use_reranker=False,
                score_fusion=False,
                knowledge_ids=knowledge_base_ids,
            )
            return ranked_qas
        except Exception as e:
            logger.error(f"QA Search Error: {e}")
            return None

    def locate_table(
        self,
        query: str,
        request: DataQACompletionRequest,
        knowledge_base_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """定位表格
        Args:
            query:查询内容
            knowledge_base_ids: 查询的知识库ID
        Returns:
            表格信息列表
        """
        input_messages = self._extract_input_messages(request)
        search_content = (
            "\n".join([f"{msg.role}: {msg.content}" for msg in input_messages])
            + "\nUser: "
            + query
        )
        logger.info(f"Search content: {search_content}")
        try:
            ranked_tables = sql_kb.search(
                self.config.sql_schema_collection,
                search_content,
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

    def generate_sql(
        self,
        table_schema: str,
        input_messages: List[ChatMessage],
        faq_results: Optional[Dict[str, Any]] = None,
        faq_score: Optional[float] = 0.0,
        thinking: Optional[bool] = False,
    ):
        """生成SQL
        Args:
            table_schema (str): _description_
            input_messages (List[ChatMessage]): _description_
            faq_results (Optional[Dict[str, Any]], optional): _description_. Defaults to None.
            thinking (Optional[bool], optional): _description_. Defaults to False.
        Returns:
            _type_: _description_
        """
        query = input_messages[-1].content
        content = dataqa_prompt.format(
            table_schema=table_schema,
            question=query,
            faq_results=faq_results,
            faq_score=faq_score,
        )
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
            optimized_query (DataQAOptimizedQuery):优化后的查询对象
            request (DataQACompletionRequest):数据问答请求对象
        Returns:
            Tuple[追问响应(如果需要),更新后的查询对象]
        """
        # 计算追问轮数
        logger.info(f"当前追问轮数:{request.follow_up_num}")
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
                    model=request.model or "follow_up",  # 新增缺失的必填字段
                    created=int(datetime.now().timestamp()),  # 新增缺失的必填字段
                    choices=choices,
                    usage=ChatUsage(
                        prompt_tokens=0, completion_tokens=0, total_tokens=0
                    ),
                    steps=[step],
                    follow_up_num=follow_up_num,
                ),
                optimized_query,
            )
        elif follow_up_num > self.config.follow_up_round:
            # 超过追问轮数,使用本轮数据问答的会话内容进行检索
            input_messages = self._extract_input_messages(request)
            recent_messages = input_messages[-self.config.follow_up_round * 2 :]
            manual_query = "".join(
                [msg.content for msg in recent_messages if msg.role == "user"]
            )
            return None, DataQAOptimizedQuery(
                original_query=optimized_query.original_query,
                rewritten_query=manual_query or optimized_query.rewritten_query,
                is_sufficient=True,  # 超过轮数后认为查询已足
            )
        # 不需要追问
        return None, optimized_query

    def locate_api(
        self, query: str, knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """定位数据API
        Args:
            query:查询内容
            knowledge_base_ids:查询的知识库ID
        Returns:
            API列表
        """
        try:
            ranked_api = document_kb.search(
                self.config.api_collection,
                query,
                knowledge_ids=knowledge_base_ids,
                top_k=self.config.max_table_results,
                use_reranker=self.config.enable_reranker,
            )
            api_list = [
                {
                    "chunk_uuid": api.chunk_id,
                    "table_name": api.data.content,
                    "score": api.reranking_score,
                }
                for api in ranked_api
            ]
            return api_list
        except Exception as e:
            logger.error(f"Locate table Error: {e}")
            return []

    def search_qa(self, query: str, knowledge_base_ids: List[str]) -> List[Chunk]:
        try:
            ranked_qas = qa_pair_kb.search(
                collection=self.config.qa_collection,
                query=query,
                top_k=self.config.fqa_sample_num,
                use_reranker=True,
                score_fusion=False,
                knowledge_ids=knowledge_base_ids,
            )
            return ranked_qas
        except Exception as e:
            logger.error(f"QA Search Error: {e}")
            return None

    def do_generate(
        self,
        request: DataQACompletionRequest,
        enable_follow_up: bool = True,
        knowledge_base_ids: Optional[List[str]] = None,
        qa_knowledge_base_ids: Optional[List[str]] = None,
        thinking: Optional[bool] = False,
    ) -> DataQAChatCompletionResponse:
        """生成回答
        Args:
            input_messages (List[ChatMessage]):输入信息列表
            enable_follow_up(bool):是否启用追问功能
            knowledge_base_ids(Optional[List[str]]):知识库ID
            thinking(bool):是否开启thinking模式, 可选 default: False
        """
        # 提取输入信息
        input_messages = self._extract_input_messages(request)
        logger.info(f"input_messages: {input_messages}")
        # Step1: modify_query
        optimized_query = self.modify_query(
            input_messages=input_messages,
            enable_follow_up=enable_follow_up,
        )
        # 处理追问逻辑
        if enable_follow_up:
            logger.info("处理追问逻辑")
            follow_up_response, optimized_query = self._handle_follow_up(
                optimized_query=optimized_query, request=request
            )
            logger.info(f"optimized_query: {optimized_query}")
            logger.info(f"follow_up_response: {follow_up_response}")
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
        # Step3: search FAQ
        faq_results = self.search_qa(entity_enriched_query, qa_knowledge_base_ids)
        if (
            self.config.enable_fqa_answer
            and faq_results
            and faq_results[0].reranking_score >= self.config.fqa_answer_threshold
        ):
            # 命中常用查询,直接使用
            step3 = self._create_step(
                WorkflowStepType.SEARCH_FAQ,
                3,
                f"命中常用查询: {faq_results[0].data.question}",
            )
            # FAQ直接命中,快速构造响应
            response_content = faq_results[0].data.answer
            choices = [
                ChatCompletionChoice(
                    finish_reason="stop",
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_content,
                        reasoning_content=None,
                        is_follow_up=False,
                    ),
                )
            ]
            return DataQAChatCompletionResponse(
                model="human",
                created=int(datetime.now().timestamp()),
                choices=choices,
                usage=ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                steps=[step1, step2, step3],
            )
        else:
            if faq_results:
                step3 = self._create_step(
                    WorkflowStepType.SEARCH_FAQ,
                    3,
                    f"找到相关FAQ 相似度为 {faq_results[0]['similarity']} 但是不满足0.9 阈值",
                )
            else:
                step3 = self._create_step(
                    WorkflowStepType.SEARCH_FAQ, 3, "没有找到相关FAQ"
                )
            faq_results = []
        # Step4: locate table
        located_table = self.locate_table(
            entity_enriched_query, request, knowledge_base_ids
        )
        step4 = self._create_step(WorkflowStepType.LOCATE_TABLE, 4, located_table)
        # Step5: generate_sql (0.7 =< 相似度<0.9,可在generate_sql中添加参考示例)
        # 创建input message的副本并更新最后一条信息为实体识别增强后的的查询
        enhanced_input_messages = copy.deepcopy(input_messages)
        enhanced_input_messages[-1].content = entity_enriched_query
        if (
            faq_results
            and len(faq_results) > 0
            and faq_results[0]["similarity"] >= 0.7
        ):
            # 修改 arguments
            response = self.generate_sql(
                table_schema=(
                    located_table[0] if located_table else ""
                ),
                input_messages=enhanced_input_messages,
                faq_results=faq_results,
                faq_score=faq_results[0]["similarity"],
                thinking=thinking,
            )
            step5 = self._create_step(WorkflowStepType.GENERATE_SQL, 5, response)
        # Step5: generate_sql (FAQ 相似度小于0.7或者没有FAQ,继续走标准流程)
        else:
            response = self.generate_sql(
                table_schema=(
                    located_table[0] if located_table else ""
                ),
                input_messages=enhanced_input_messages,
                faq_results=None,
                thinking=thinking,
            )
            step5 = self._create_step(WorkflowStepType.GENERATE_SQL, 5, response)
        # 构造最终响应
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

    async def amodify_query(
        self,
        input_messages: List[ChatMessage],
        enable_follow_up: bool,
    ) -> DataQAOptimizedQuery:
        """问题改写
        Args:
            input_messages (List[ChatMessage]):输入的消息列表
            enable_follow_up(bool):是否启用追问功能
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
            logger.error(f"Modify query Error: {e}")
            traceback.print_exc()
            raise e