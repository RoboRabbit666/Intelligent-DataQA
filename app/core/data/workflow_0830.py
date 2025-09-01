# coding: utf-8
import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    history_round: int = 3
    follow_up_round: int = 1
    reranking_threshold: float = 0.2
    sql_schema_collection: Optional[str] = "hybrid_sql"
    domain_collection: Optional[str] = "domain_kl"
    api_collection: Optional[str] = "api_kl"
    qa_collection: Optional[str] = "sql_qa"
    max_table_results: int = 3
    enable_entity_recognition: bool = True
    enable_reranker: bool = True
    enable_fqa_answer: bool = True
    fqa_answer_threshold: float = 0.9
    fqa_sample_num: int = 3

    def __post_init__(self):
        if self.history_round < self.follow_up_round:
            raise ValueError("history_round must be >= follow_up_round")


class DataQaWorkflow:
    """
    工作流：
    1 问题改写/追问
    2 实体识别
    3 FAQ 快速命中（高分直接返回）
    4 并行检索：表 / API / 业务知识
    5 SQL 生成（融合多源 + FAQ 样例）
    """

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

    # ---------------- 基础工具 ----------------
    def _create_step(self, step_type: WorkflowStepType, number: int, prompt: Any) -> ChatStep:
        names = {
            WorkflowStepType.FOLLOW_UP: "问题追问",
            WorkflowStepType.MODIFY_QUERY: "问题改写",
            WorkflowStepType.ENTITY_RECOGNITION: "问题实体识别",
            WorkflowStepType.SEARCH_FAQ: "语义搜索FAQ",
            WorkflowStepType.LOCATE_TABLE: "表格定位",
            WorkflowStepType.LOCATE_API: "API定位",
            WorkflowStepType.BUSINESS_INFO: "业务知识检索",
            WorkflowStepType.GENERATE_SQL: "SQL生成",
        }
        if not isinstance(prompt, str):
            if isinstance(prompt, (list, dict)):
                prompt = json.dumps(prompt, ensure_ascii=False, indent=2)
            else:
                prompt = str(prompt)
        return ChatStep(
            key=step_type.value,
            name=names[step_type],
            number=number,
            prompt=prompt,
            finished=True,
        )

    def _extract_input_messages(self, request: DataQACompletionRequest) -> List[ChatMessage]:
        """提取输入信息列表"""
        return request.messages[-self.config.history_round * 2 :]

    # ---------------- Step1 改写/追问 ----------------
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
            optimized_query = self.query_optimizer.generate_optimized_query(
                query=last_user_msg.content,
                chat_history=input_messages_copy[: last_index + 1],
                optimization_type=optimization_type,
            )
            return optimized_query
        except Exception as e:
            logger.error(f"Modify query Error: {e}")
            traceback.print_exc()
            raise e

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

    # ---------------- Step2 实体识别 ----------------
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

    # ---------------- Step3 FAQ 检索 ----------------
    def search_qa(self, query: str, knowledge_base_ids: Optional[List[str]]) -> List[Chunk]:
        try:
            return qa_pair_kb.search(
                collection=self.config.qa_collection,
                query=query,
                top_k=self.config.fqa_sample_num,
                use_reranker=True,
                score_fusion=False,
                knowledge_ids=knowledge_base_ids,
            )
        except Exception as e:
            logger.error(f"FAQ search error: {e}")
            return []

    # ---------------- Step4 多源检索 ----------------
    def locate_table(
        self,
        query: str,
        request: DataQACompletionRequest,
        knowledge_base_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        try:
            ranked = sql_kb.search(
                self.config.sql_schema_collection,
                query,
                knowledge_ids=knowledge_base_ids,
                top_k=self.config.max_table_results,
                use_reranker=self.config.enable_reranker,
            )
            return [
                {
                    "table_name": c.data.table_name,
                    "table_info": c.data.table_info,
                    "table_schema": c.data.table_schema,
                    "score": c.reranking_score,
                }
                for c in ranked
            ]
        except Exception as e:
            logger.error(f"Locate table error: {e}")
            return []

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

    def _parallel_retrieve(
        self, query: str, request: DataQACompletionRequest, knowledge_base_ids: Optional[List[str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        并行检索 表 / API / 业务知识
        """
        result = {"tables": [], "apis": [], "business": []}
        with ThreadPoolExecutor(max_workers=3) as ex:
            tasks = {
                ex.submit(self.locate_table, query, request, knowledge_base_ids): "tables",
                ex.submit(self.locate_api, query, knowledge_base_ids): "apis",
                ex.submit(self.business_info_search, query, 3, knowledge_base_ids): "business",
            }
            for f in as_completed(tasks):
                k = tasks[f]
                try:
                    result[k] = f.result() or []
                except Exception as e:
                    logger.error(f"并行检索失败 {k}: {e}")
        return result

    # ---------------- Step5 SQL 生成 ----------------
    def _format_faq_samples(self, chunks: List[Chunk]) -> Tuple[str, float]:
        if not chunks:
            return "无FAQ样例", 0.0
        top_score = chunks[0].reranking_score
        samples = [c for c in chunks if c.reranking_score >= 0.7]
        if not samples:
            return "无FAQ样例", top_score
        lines = []
        for i, c in enumerate(samples[: self.config.fqa_sample_num], 1):
            q = getattr(c.data, "question", "")
            a = getattr(c.data, "answer", "")
            lines.append(f"[FAQ Sample {i}] score={round(c.reranking_score,4)}\nQ:{q}\nA:{a}")
        return "\n\n".join(lines), top_score

    def _build_context(self, tables: List[Dict[str, Any]], apis: List[Dict[str, Any]], biz: List[Dict[str, Any]]) -> str:
        def fmt_tables():
            if not tables:
                return "无相关表"
            return "\n\n".join(
                f"[Table {i}] name:{t['table_name']}\ninfo:{t['table_info']}\nschema:\n{t['table_schema']}\nscore:{round(t['score'],4)}"
                for i, t in enumerate(tables, 1)
            )

        def fmt_apis():
            if not apis:
                return "无相关API"
            return "\n\n".join(
                f"[API {i}] score:{round(a['score'],4)}\ncontent:{a['api_content']}"
                for i, a in enumerate(apis, 1)
            )

        def fmt_biz():
            if not biz:
                return "无业务背景"
            return "\n\n".join(
                f"[Business {i}] score:{round(b['score'],4)}\ncontent:{b['content']}"
                for i, b in enumerate(biz, 1)
            )

        return f"[TABLES]\n{fmt_tables()}\n\n[APIS]\n{fmt_apis()}\n\n[BUSINESS]\n{fmt_biz()}"

    def generate_sql(
        self,
        table_schema: str,
        input_messages: List[ChatMessage],
        faq_results: Optional[str] = None,
        faq_score: Optional[float] = 0.0,
        thinking: Optional[bool] = False,
    ):
        query = input_messages[-1].content
        content = dataqa_prompt.format(
            table_schema=table_schema,
            question=query,
            faq_results=faq_results or "无FAQ样例",
            faq_score=faq_score or 0.0,
        )
        system_msg = ChatMessage(role="system", content=content)
        client = self.ans_thinking_client if thinking else self.ans_client
        return client.invoke(messages=[system_msg] + input_messages)

    # ---------------- 主流程 ----------------
    def do_generate(
        self,
        request: DataQACompletionRequest,
        enable_follow_up: bool = True,
        knowledge_base_ids: Optional[List[str]] = None,
        qa_knowledge_base_ids: Optional[List[str]] = None,
        thinking: Optional[bool] = False,
    ) -> DataQAChatCompletionResponse:
        messages = self._extract_input_messages(request)

        # Step1 改写/追问
        optimized = self.modify_query(messages, enable_follow_up)
        if enable_follow_up:
            follow_up_resp, optimized = self._handle_follow_up(optimized, request)
            if follow_up_resp:
                return follow_up_resp
        step1 = self._create_step(WorkflowStepType.MODIFY_QUERY, 1, optimized.rewritten_query)

        # Step2 实体识别
        enriched_query = self.entity_recognition(optimized.rewritten_query)
        step2 = self._create_step(WorkflowStepType.ENTITY_RECOGNITION, 2, enriched_query)

        # Step3 FAQ
        faq_chunks = self.search_qa(enriched_query, qa_knowledge_base_ids)
        if (
            self.config.enable_fqa_answer
            and faq_chunks
            and faq_chunks[0].reranking_score >= self.config.fqa_answer_threshold
        ):
            top = faq_chunks[0]
            step3 = self._create_step(
                WorkflowStepType.SEARCH_FAQ,
                3,
                f"命中FAQ直接返回 score={round(top.reranking_score,4)} Q:{getattr(top.data,'question','')}",
            )
            answer = getattr(top.data, "answer", "")
            choice = ChatCompletionChoice(
                finish_reason="stop",
                index=0,
                message=ChatMessage(role="assistant", content=answer, reasoning_content=None, is_follow_up=False),
            )
            return DataQAChatCompletionResponse(
                model=request.model or "faq_direct",
                created=int(datetime.now().timestamp()),
                choices=[choice],
                usage=ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                steps=[step1, step2, step3],
            )
        if faq_chunks:
            step3 = self._create_step(
                WorkflowStepType.SEARCH_FAQ,
                3,
                f"FAQ最高得分 {round(faq_chunks[0].reranking_score,4)} 未达阈值 {self.config.fqa_answer_threshold}",
            )
        else:
            step3 = self._create_step(WorkflowStepType.SEARCH_FAQ, 3, "未检索到FAQ")

        # Step4 并行检索 表 / API / 业务
        retrieval = self._parallel_retrieve(enriched_query, request, knowledge_base_ids)
        tables = retrieval["tables"]
        apis = retrieval["apis"]
        biz = retrieval["business"]
        step4 = self._create_step(WorkflowStepType.LOCATE_TABLE, 4, tables)
        step5 = self._create_step(WorkflowStepType.LOCATE_API, 5, apis)
        step6 = self._create_step(WorkflowStepType.BUSINESS_INFO, 6, biz)

        # Step5 SQL 生成
        enhanced_messages = copy.deepcopy(messages) or [ChatMessage(role="user", content=enriched_query)]
        if enhanced_messages:
            enhanced_messages[-1].content = enriched_query
        faq_block, faq_top = self._format_faq_samples(faq_chunks)
        context_block = self._build_context(tables, apis, biz)
        llm_resp = self.generate_sql(
            table_schema=context_block,
            input_messages=enhanced_messages,
            faq_results=faq_block,
            faq_score=faq_top,
            thinking=thinking,
        )
        step7 = self._create_step(WorkflowStepType.GENERATE_SQL, 7, "SQL生成完成")

        usage = ChatUsage(
            prompt_tokens=llm_resp.usage.prompt_tokens,
            completion_tokens=llm_resp.usage.completion_tokens,
            total_tokens=llm_resp.usage.total_tokens,
        )
        choices = [
            ChatCompletionChoice(
                finish_reason=c.finish_reason,
                index=c.index,
                message=ChatMessage(
                    role=c.message.role,
                    content=c.message.content,
                    reasoning_content=c.message.reasoning_content,
                    is_follow_up=False,
                ),
            )
            for c in llm_resp.choices
        ]
        return DataQAChatCompletionResponse(
            id=llm_resp.id,
            model=llm_resp.model,
            created=llm_resp.created,
            choices=choices,
            usage=usage,
            steps=[step1, step2, step3, step4, step5, step6, step7],
        )

    # ---------------- 异步改写 (保留原接口占位) ----------------
    async def amodify_query(
        self,
        input_messages: List[ChatMessage],
        enable_follow_up: bool,
    ) -> DataQAOptimizedQuery:
        return self.modify_query(input_messages,