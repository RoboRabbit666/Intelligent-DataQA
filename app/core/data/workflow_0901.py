# coding: utf-8
import copy
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator

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
    fqa_reference_threshold: float = 0.7  # FQA作为参考示例的阈值
    fqa_sample_num: int = 3  # fqa SQL 样例个数
    # API相关设置
    enable_api_search: bool = False  # 是否启用API搜索

    def __post_init__(self):
        if self.history_round < self.follow_up_round:
            raise ValueError(
                f"history_round({self.history_round}) must be >= follow_up_round({self.follow_up_round})"
            )


class DataQaWorkflow:
    """数据问答工作流主类"""
    
    def __init__(
        self,
        ans_llm: LLMModel,
        ans_thinking_llm: LLMModel,
        query_llm: LLMModel,
        config: Optional[WorkflowConfig] = None,
        collection: Optional[str] = None,  # 添加collection参数以匹配routers/data.py
        reranking_threshold: Optional[float] = None,  # 添加reranking_threshold参数
    ):
        self.ans_client = ans_llm
        self.ans_thinking_client = ans_thinking_llm
        self.query_optimizer = QueryOptimizer(query_llm)
        self.config = config or WorkflowConfig()
        
        # 如果传入了collection和reranking_threshold，更新配置
        if collection:
            self.config.sql_schema_collection = collection
        if reranking_threshold is not None:
            self.config.reranking_threshold = reranking_threshold

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
        # 转换prompt为字符串
        if not isinstance(prompt, str):
            if isinstance(prompt, list):
                prompt = json.dumps(prompt, ensure_ascii=False, indent=2)
            elif isinstance(prompt, dict):
                prompt = json.dumps(prompt, ensure_ascii=False, indent=2)
            else:
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
            input_messages: 输入的消息列表
            enable_follow_up: 是否启用追问功能
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
            
            # 获取最后一条用户消息
            last_user_msg = None
            last_index = None
            for index, message in enumerate(input_messages_copy):
                if message.role == "user":
                    last_user_msg = message
                    last_index = index
            
            if not last_user_msg:
                # 如果没有用户消息，返回默认查询
                return DataQAOptimizedQuery(
                    original_query="",
                    rewritten_query="",
                    is_sufficient=False
                )
            
            optimized_query = self.query_optimizer.generate_optimized_query(
                query=last_user_msg.content,
                chat_history=input_messages_copy[:last_index] if last_index else [],
                optimization_type=optimization_type,
            )
            return optimized_query
        except Exception as e:
            logger.error(f"Modify query Error: {e}")
            traceback.print_exc()
            raise e

    def entity_recognition(self, query: str) -> str:
        """实体识别增强查询
        Args:
            query: 原始查询
        Returns:
            增强后的查询
        """
        if not self.config.enable_entity_recognition:
            return query
        
        try:
            enhanced_query = query
            entity_list = tokenizer.recognize(query)
            
            for entity in entity_list:
                if entity["id"] and entity["text"] in enhanced_query:
                    if entity["label"] == "合约":
                        # 标准化合约代码
                        normalized_code = re.sub(
                            r"[-_\.\s/]+", "", entity["text"].upper()
                        )
                        substring = f"({normalized_code})(合约)"
                    else:
                        substring = f"({entity['text']})({entity['label']})"
                    enhanced_query = enhanced_query.replace(
                        entity["text"], substring, 1
                    )
            
            logger.info(f"实体识别完成: {query} -> {enhanced_query}")
            return enhanced_query
        except Exception as e:
            logger.error(f"实体识别失败: {e}")
            return query  # 失败时返回原查询

    def search_qa(
        self, query: str, knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Chunk]:
        """搜索FAQ知识库
        Args:
            query: 查询内容
            knowledge_base_ids: 知识库ID列表
        Returns:
            FAQ结果列表
        """
        try:
            ranked_qas = qa_pair_kb.search(
                collection=self.config.qa_collection,
                query=query,
                top_k=self.config.fqa_sample_num,
                use_reranker=self.config.enable_reranker,
                score_fusion=False,
                knowledge_ids=knowledge_base_ids,
            )
            return ranked_qas
        except Exception as e:
            logger.error(f"QA Search Error: {e}")
            return []

    def locate_table(
        self,
        query: str,
        request: DataQACompletionRequest,
        knowledge_base_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """定位数据表
        Args:
            query: 查询内容
            request: 请求对象
            knowledge_base_ids: 知识库ID列表
        Returns:
            表格信息列表
        """
        input_messages = self._extract_input_messages(request)
        # 构建搜索内容，包含历史对话上下文
        search_content = (
            "\n".join([f"{msg.role}: {msg.content}" for msg in input_messages])
            + "\nUser: "
            + query
        )
        logger.info(f"Search content: {search_content}")
        
        try:
            ranked_tables = sql_kb.search(
                collection=self.config.sql_schema_collection,
                query=search_content,
                knowledge_ids=knowledge_base_ids,
                top_k=self.config.max_table_results,
                use_reranker=self.config.enable_reranker,
            )
            
            tables = [
                {
                    "chunk_uuid": table.chunk_id,
                    "table_name": table.data.table_name,
                    "table_info": table.data.table_info,
                    "table_schema": table.data.table_schema,
                    "score": table.reranking_score,
                }
                for table in ranked_tables
            ]
            return tables
        except Exception as e:
            logger.error(f"Locate table Error: {e}")
            return []

    def locate_api(
        self, query: str, knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """定位数据API
        Args:
            query: 查询内容
            knowledge_base_ids: 查询的知识库ID
        Returns:
            API列表
        """
        if not self.config.enable_api_search:
            return []
        
        try:
            ranked_apis = document_kb.search(
                collection=self.config.api_collection,
                query=query,
                knowledge_ids=knowledge_base_ids,
                top_k=self.config.max_table_results,
                use_reranker=self.config.enable_reranker,
            )
            
            api_list = [
                {
                    "chunk_uuid": api.chunk_id,
                    "api_name": api.data.api_name if hasattr(api.data, 'api_name') else api.data.content,
                    "api_description": api.data.api_description if hasattr(api.data, 'api_description') else "",
                    "score": api.reranking_score,
                }
                for api in ranked_apis
            ]
            return api_list
        except Exception as e:
            logger.error(f"Locate API Error: {e}")
            return []

    def generate_sql(
        self,
        table_schema: str,
        input_messages: List[ChatMessage],
        faq_results: Optional[List[Dict[str, Any]]] = None,
        thinking: bool = False,
    ):
        """生成SQL
        Args:
            table_schema: 表结构信息
            input_messages: 输入消息列表
            faq_results: FAQ参考结果
            thinking: 是否开启深度思考
        Returns:
            LLM响应
        """
        query = input_messages[-1].content
        
        # 格式化FAQ结果
        faq_text = ""
        faq_score = 0.0
        if faq_results:
            faq_examples = []
            for faq in faq_results[:self.config.fqa_sample_num]:
                faq_examples.append(f"问题: {faq['question']}\nSQL: {faq['answer']}")
            faq_text = "\n---\n".join(faq_examples)
            faq_score = faq_results[0]['score'] if faq_results else 0.0
        
        content = dataqa_prompt.format(
            table_schema=table_schema,
            question=query,
            faq_results=faq_text,
            faq_score=faq_score,
        )
        
        system_msg = ChatMessage(
            role="system",
            content=content,
        )
        
        llm = self.ans_thinking_client if thinking else self.ans_client
        response = llm.invoke(messages=[system_msg] + input_messages[:])
        return response

    def _handle_follow_up(
        self,
        optimized_query: DataQAOptimizedQuery,
        request: DataQACompletionRequest,
    ) -> Tuple[Optional[DataQAChatCompletionResponse], DataQAOptimizedQuery]:
        """处理追问逻辑
        Args:
            optimized_query: 优化后的查询对象
            request: 请求对象
        Returns:
            (追问响应, 更新后的查询对象)
        """
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
                DataQaChatCompletionResponse(
                    model=request.model or "dataqa",
                    created=int(datetime.now().timestamp()),
                    choices=choices,
                    usage=ChatUsage(
                        prompt_tokens=0, 
                        completion_tokens=0, 
                        total_tokens=0
                    ),
                    steps=[step],
                    follow_up_num=follow_up_num,
                ),
                optimized_query,
            )
        
        # 超过追问轮数，使用合并的查询内容
        elif follow_up_num > self.config.follow_up_round:
            input_messages = self._extract_input_messages(request)
            recent_messages = input_messages[-self.config.follow_up_round * 2 :]
            manual_query = " ".join(
                [msg.content for msg in recent_messages if msg.role == "user"]
            )
            
            return None, DataQAOptimizedQuery(
                original_query=optimized_query.original_query,
                rewritten_query=manual_query or optimized_query.rewritten_query,
                is_sufficient=True,
            )
        
        # 不需要追问
        return None, optimized_query

    def do_generate(
        self,
        input_messages: List[ChatMessage],
        use_reranker: bool = True,
        reranker_info: Optional[RerankerInfo] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        thinking: bool = False,
    ) -> DataQaChatCompletionResponse:
        """生成数据问答响应（完善版本）
        Args:
            input_messages: 输入消息列表
            use_reranker: 是否使用重排序
            reranker_info: 重排序配置
            knowledge_base_ids: 知识库ID列表
            thinking: 是否开启深度思考
        Returns:
            数据问答响应
        """
        # 构造请求对象（为了兼容现有的辅助函数）
        request = DataQACompletionRequest(
            messages=input_messages,
            knowledge_base_ids=knowledge_base_ids or [],
            use_reranker=use_reranker,
            reranker_info=reranker_info,
            follow_up_num=0,
            thinking=thinking,
        )
        
        # 更新配置中的重排序设置
        if reranker_info:
            self.config.enable_reranker = use_reranker
            self.config.reranking_threshold = reranker_info.threshold
        
        # 提取输入信息
        extracted_messages = self._extract_input_messages(request)
        logger.info(f"输入消息: {extracted_messages}")
        
        # Step 1: 问题改写
        optimized_query = self.modify_query(
            input_messages=extracted_messages,
            enable_follow_up=True,
        )
        
        # 处理追问逻辑
        follow_up_response, optimized_query = self._handle_follow_up(
            optimized_query=optimized_query, 
            request=request
        )
        if follow_up_response:
            return follow_up_response
        
        step1 = self._create_step(
            WorkflowStepType.MODIFY_QUERY, 1, optimized_query.rewritten_query
        )
        
        # Step 2: 实体识别
        query = optimized_query.rewritten_query
        entity_enriched_query = self.entity_recognition(query)
        step2 = self._create_step(
            WorkflowStepType.ENTITY_RECOGNITION, 2, entity_enriched_query
        )
        
        # Step 3: 搜索FAQ
        faq_results = self.search_qa(entity_enriched_query, knowledge_base_ids)
        faq_formatted = []  # 格式化FAQ结果
        
        if faq_results and len(faq_results) > 0:
            # 格式化FAQ结果为字典列表
            for faq in faq_results:
                faq_formatted.append({
                    'question': faq.data.question if hasattr(faq.data, 'question') else "",
                    'answer': faq.data.answer if hasattr(faq.data, 'answer') else "",
                    'score': faq.reranking_score
                })
            
            # 检查是否直接使用FAQ答案
            if faq_formatted[0]['score'] >= self.config.fqa_answer_threshold:
                step3 = self._create_step(
                    WorkflowStepType.SEARCH_FAQ,
                    3,
                    f"命中常用查询（相似度: {faq_formatted[0]['score']:.2f}）: {faq_formatted[0]['question']}",
                )
                
                # 直接返回FAQ答案
                response_content = faq_formatted[0]['answer']
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
                
                return DataQaChatCompletionResponse(
                    model="faq",
                    created=int(datetime.now().timestamp()),
                    choices=choices,
                    usage=ChatUsage(
                        prompt_tokens=0, 
                        completion_tokens=0, 
                        total_tokens=0
                    ),
                    steps=[step1, step2, step3],
                )
            else:
                step3 = self._create_step(
                    WorkflowStepType.SEARCH_FAQ,
                    3,
                    f"找到相关FAQ（相似度: {faq_formatted[0]['score']:.2f}），作为参考示例",
                )
        else:
            step3 = self._create_step(
                WorkflowStepType.SEARCH_FAQ, 3, "没有找到相关FAQ"
            )
            faq_formatted = []
        
        # Step 4: 定位表格
        located_tables = self.locate_table(
            entity_enriched_query, request, knowledge_base_ids
        )
        step4 = self._create_step(
            WorkflowStepType.LOCATE_TABLE, 
            4, 
            [{"table_name": t["table_name"], "score": t["score"]} for t in located_tables]
        )
        
        # Step 5: API定位（可选）
        steps = [step1, step2, step3, step4]
        if self.config.enable_api_search:
            located_apis = self.locate_api(entity_enriched_query, knowledge_base_ids)
            step_api = self._create_step(
                WorkflowStepType.LOCATE_API,
                5,
                [{"api_name": a["api_name"], "score": a["score"]} for a in located_apis]
            )
            steps.append(step_api)
        
        # Step 6: 生成SQL
        enhanced_input_messages = copy.deepcopy(extracted_messages)
        enhanced_input_messages[-1].content = entity_enriched_query
        
        # 获取表结构信息
        table_schema = ""
        if located_tables:
            # 组合多个表的schema信息
            schemas = []
            for table in located_tables[:2]:  # 最多使用前2个表
                schemas.append(f"表名: {table['table_name']}\n{table['table_schema']}")
            table_schema = "\n\n".join(schemas)
        
        # 决定是否使用FAQ作为参考
        faq_for_generation = None
        if faq_formatted and faq_formatted[0]['score'] >= self.config.fqa_reference_threshold:
            faq_for_generation = faq_formatted[:self.config.fqa_sample_num]
        
        response = self.generate_sql(
            table_schema=table_schema,
            input_messages=enhanced_input_messages,
            faq_results=faq_for_generation,
            thinking=thinking,
        )
        
        step_sql = self._create_step(
            WorkflowStepType.GENERATE_SQL, 
            len(steps) + 1, 
            "SQL生成完成"
        )
        steps.append(step_sql)
        
        # 构造最终响应
        usage = ChatUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        
        choices = [
            ChatCompletionChoice(
                finish_reason=choice.finish_reason,
                index=choice.index,
                message=ChatMessage(
                    role=choice.message.role,
                    content=choice.message.content,
                    reasoning_content=choice.message.reasoning_content,
                    is_follow_up=False,
                ),
            )
            for choice in response.choices
        ]
        
        return DataQaChatCompletionResponse(
            id=response.id,
            model=response.model,
            created=response.created,
            choices=choices,
            usage=usage,
            steps=steps,
        )

    async def do_stream(
        self,
        input_messages: List[ChatMessage],
        use_reranker: bool = True,
        reranker_info: Optional[RerankerInfo] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        thinking: bool = False,
    ) -> AsyncIterator[str]:
        """流式生成数据问答响应
        Args:
            input_messages: 输入消息列表
            use_reranker: 是否使用重排序
            reranker_info: 重排序配置
            knowledge_base_ids: 知识库ID列表
            thinking: 是否开启深度思考
        Yields:
            SSE格式的响应数据
        """
        try:
            # 执行完整的工作流获取结果（非流式部分）
            response = self.do_generate(
                input_messages=input_messages,
                use_reranker=use_reranker,
                reranker_info=reranker_info,
                knowledge_base_ids=knowledge_base_ids,
                thinking=thinking,
            )
            
            # 如果是追问响应，直接返回
            if response.follow_up_num > 0:
                yield f"data: {response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # 流式输出最终的SQL生成结果
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                # 分块输出内容
                chunk_size = 50  # 每次输出的字符数
                for i in range(0, len(content), chunk_size):
                    chunk_content = content[i:i+chunk_size]
                    chunk_response = {
                        "id": response.id,
                        "object": "chat.completion.chunk",
                        "created": response.created,
                        "model": response.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk_content},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_response, ensure_ascii=False)}\n\n"
                
                # 发送结束信号
                final_chunk = {
                    "id": response.id,
                    "object": "chat.completion.chunk",
                    "created": response.created,
                    "model": response.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }],
                    "usage": response.usage.model_dump() if response.usage else None,
                    "steps": [step.model_dump() for step in response.steps] if response.steps else None
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"流式生成错误: {e}")
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"