# coding: utf-8
import copy
import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor  # 新增：用于并行执行
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
    DataQaChatCompletionResponse,
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
    # 保持不变的配置
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
    
    # FAQ阈值策略
    enable_fqa_answer: bool = True
    fqa_answer_threshold: float = 0.9
    fqa_reference_threshold: float = 0.7
    fqa_sample_num: int = 3
    
    # API相关配置
    enable_api_search: bool = True
    api_threshold: float = 0.8

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
        collection: Optional[str] = None,
        reranking_threshold: Optional[float] = None,
    ):
        # 保持不变：初始化
        self.ans_client = ans_llm
        self.ans_thinking_client = ans_thinking_llm
        self.query_optimizer = QueryOptimizer(query_llm)
        self.config = config or WorkflowConfig()
        
        if collection:
            self.config.sql_schema_collection = collection
        if reranking_threshold is not None:
            self.config.reranking_threshold = reranking_threshold

    def _create_step(
        self, step_type: WorkflowStepType, number: int, prompt: Any
    ) -> ChatStep:
        """创建工作流步骤 - 保持不变"""
        step_names = {
            WorkflowStepType.FOLLOW_UP: "问题追问",
            WorkflowStepType.MODIFY_QUERY: "问题改写",
            WorkflowStepType.ENTITY_RECOGNITION: "问题实体识别",
            WorkflowStepType.LOCATE_API: "API定位",
            WorkflowStepType.SEARCH_FAQ: "语义搜索FAQ",
            WorkflowStepType.LOCATE_TABLE: "表格定位",
            WorkflowStepType.GENERATE_SQL: "SQL生成",
            WorkflowStepType.BUSINESS_INFO: "业务知识搜索",
        }
        
        if not isinstance(prompt, str):
            if isinstance(prompt, list):
                prompt = json.dumps(prompt, ensure_ascii=False, indent=2)
            elif isinstance(prompt, dict):
                prompt = json.dumps(prompt, ensure_ascii=False, indent=2)
            else:
                prompt = str(prompt)
        
        return ChatStep(
            key=step_type.value,
            name=step_names.get(step_type, step_type.value),
            number=number,
            prompt=prompt,
            finished=True,
        )

    def _extract_input_messages(
        self, request: DataQACompletionRequest
    ) -> List[ChatMessage]:
        """提取输入信息列表 - 保持不变"""
        return request.messages[-self.config.history_round * 2 :]

    def modify_query(
        self,
        input_messages: List[ChatMessage],
        enable_follow_up: bool,
    ) -> DataQAOptimizedQuery:
        """问题改写 - 保持不变"""
        try:
            input_messages_copy = copy.deepcopy(input_messages)
            optimization_type = (
                QueryOptimizationType.FOLLOWUP
                if enable_follow_up
                else QueryOptimizationType.DATAQA
            )
            
            last_user_msg = None
            last_index = None
            for index, message in enumerate(input_messages_copy):
                if message.role == "user":
                    last_user_msg = message
                    last_index = index
            
            if not last_user_msg:
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
        """实体识别 - 保持不变"""
        if not self.config.enable_entity_recognition:
            return query
        
        try:
            enhanced_query = query
            entity_list = tokenizer.recognize(query)
            
            for entity in entity_list:
                if entity["id"] and entity["text"] in enhanced_query:
                    if entity["label"] == "合约":
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
            return query

    def business_info_search(
        self, query: str, top_k: int = 2, knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Chunk]:
        """搜索业务知识 - 保持不变"""
        try:
            ranked_qas = business_info_kb.search(
                collection=self.config.domain_collection,
                query=query,
                search_type=SearchType.dense,
                top_k=top_k,
                use_reranker=False,
                score_fusion=False,
                knowledge_ids=knowledge_base_ids,
            )
            return ranked_qas
        except Exception as e:
            logger.error(f"Business info search Error: {e}")
            return []

    def search_qa(
        self, query: str, knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Chunk]:
        """搜索FAQ知识库 - 保持不变"""
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

    def locate_api(
        self, query: str, knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Chunk]:
        """定位数据API - 保持不变"""
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
            return ranked_apis
        except Exception as e:
            logger.error(f"Locate API Error: {e}")
            return []

    def locate_table(
        self,
        query: str,
        request: DataQACompletionRequest,
        knowledge_base_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """定位表格 - 保持不变"""
        input_messages = self._extract_input_messages(request)
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

    def generate_sql(
        self,
        table_schema: str,
        input_messages: List[ChatMessage],
        faq_results: Optional[List[Chunk]] = None,
        business_context: Optional[List[Chunk]] = None,
        thinking: bool = False,
    ):
        """生成SQL - 保持不变"""
        query = input_messages[-1].content
        
        # 处理FAQ结果
        faq_text = ""
        faq_score = 0.0
        if faq_results:
            faq_examples = []
            for faq in faq_results[:self.config.fqa_sample_num]:
                faq_examples.append(
                    f"问题: {faq.data.question}\nSQL: {faq.data.answer}"
                )
            faq_text = "\n---\n".join(faq_examples)
            faq_score = faq_results[0].reranking_score if faq_results else 0.0
        
        # 构建业务上下文
        business_text = ""
        if business_context:
            business_info = []
            for info in business_context[:2]:
                if hasattr(info.data, 'business_desc'):
                    business_info.append(info.data.business_desc)
                elif hasattr(info.data, 'content'):
                    business_info.append(info.data.content)
            if business_info:
                business_text = "\n业务背景知识：\n" + "\n".join(business_info)
        
        content = dataqa_prompt.format(
            table_schema=table_schema + business_text,
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
    ) -> Tuple[Optional[DataQaChatCompletionResponse], DataQAOptimizedQuery]:
        """处理追问逻辑 - 保持不变"""
        follow_up_num = (
            request.follow_up_num + 1 if not optimized_query.is_sufficient else 0
        )
        
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
        
        return None, optimized_query

    # 新增：并行搜索FAQ和API的辅助函数
    def _parallel_search_faq_api(
        self, 
        query: str, 
        knowledge_base_ids: Optional[List[str]] = None
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """并行执行FAQ和API搜索"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交两个任务
            faq_future = executor.submit(self.search_qa, query, knowledge_base_ids)
            api_future = executor.submit(self.locate_api, query, knowledge_base_ids)
            
            # 等待两个任务完成并获取结果
            faq_results = faq_future.result()
            api_results = api_future.result()
            
        return faq_results, api_results

    def do_generate(
        self,
        input_messages: List[ChatMessage],
        use_reranker: bool = True,
        reranker_info: Optional[RerankerInfo] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        thinking: bool = False,
    ) -> DataQaChatCompletionResponse:
        """生成数据问答响应 - 修改：实现真正的并行执行"""
        
        # 构造请求对象 - 保持不变
        request = DataQACompletionRequest(
            messages=input_messages,
            knowledge_base_ids=knowledge_base_ids or [],
            use_reranker=use_reranker,
            reranker_info=reranker_info,
            follow_up_num=0,
            thinking=thinking,
        )
        
        if reranker_info:
            self.config.enable_reranker = use_reranker
            self.config.reranking_threshold = reranker_info.threshold
        
        extracted_messages = self._extract_input_messages(request)
        logger.info(f"输入消息: {extracted_messages}")
        
        # Step 1: 问题改写 - 保持不变
        optimized_query = self.modify_query(
            input_messages=extracted_messages,
            enable_follow_up=True,
        )
        
        follow_up_response, optimized_query = self._handle_follow_up(
            optimized_query=optimized_query,
            request=request
        )
        if follow_up_response:
            return follow_up_response
        
        step1 = self._create_step(
            WorkflowStepType.MODIFY_QUERY, 1, optimized_query.rewritten_query
        )
        
        # Step 2: 实体识别 - 保持不变
        query = optimized_query.rewritten_query
        entity_enriched_query = self.entity_recognition(query)
        step2 = self._create_step(
            WorkflowStepType.ENTITY_RECOGNITION, 2, entity_enriched_query
        )
        
        # Step 3: 业务知识搜索 - 保持不变
        business_context = self.business_info_search(
            entity_enriched_query, 
            top_k=2, 
            knowledge_base_ids=knowledge_base_ids
        )
        step3 = self._create_step(
            WorkflowStepType.BUSINESS_INFO, 
            3, 
            f"找到{len(business_context)}条业务知识"
        )
        
        # Step 4: 真正的并行执行FAQ和API搜索 - 重大修改
        faq_results, api_results = self._parallel_search_faq_api(
            entity_enriched_query, 
            knowledge_base_ids
        )
        
        # 获取分数
        faq_score = faq_results[0].reranking_score if faq_results else 0.0
        api_score = api_results[0].reranking_score if api_results else 0.0
        
        # 判断快速路径 - API优先
        if api_results and api_score >= self.config.api_threshold:
            # API快速路径
            step4 = self._create_step(
                WorkflowStepType.LOCATE_API,
                4,
                f"命中API（相似度: {api_score:.2f}）"
            )
            
            # 构造API响应
            api_data = api_results[0].data
            api_info = {
                "api_id": api_data.api_id if hasattr(api_data, 'api_id') else "",
                "api_name": api_data.api_name if hasattr(api_data, 'api_name') else "",
                "api_description": api_data.api_description if hasattr(api_data, 'api_description') else "",
                "api_request": api_data.api_request if hasattr(api_data, 'api_request') else "",
                "api_response": api_data.api_response if hasattr(api_data, 'api_response') else "",
            }
            
            response_content = f"""找到匹配的API接口：

API名称: {api_info['api_name']}
API描述: {api_info['api_description']}
请求参数: {api_info['api_request']}
响应参数: {api_info['api_response']}

使用此API可以获取所需数据。"""
            
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
                model="api_call",
                created=int(datetime.now().timestamp()),
                choices=choices,
                usage=ChatUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0
                ),
                steps=[step1, step2, step3, step4],
            )
        
        elif faq_results and faq_score >= self.config.fqa_answer_threshold:
            # FAQ快速路径
            step4 = self._create_step(
                WorkflowStepType.SEARCH_FAQ,
                4,
                f"命中常用查询（相似度: {faq_score:.2f}）"
            )
            
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
            
            return DataQaChatCompletionResponse(
                model="faq_sql",
                created=int(datetime.now().timestamp()),
                choices=choices,
                usage=ChatUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0
                ),
                steps=[step1, step2, step3, step4],
            )
        
        # 没有快速路径，继续完整流程
        if faq_results and faq_score >= self.config.fqa_reference_threshold:
            step4 = self._create_step(
                WorkflowStepType.SEARCH_FAQ,
                4,
                f"找到相关FAQ（相似度: {faq_score:.2f}），作为参考"
            )
        else:
            step4 = self._create_step(
                WorkflowStepType.SEARCH_FAQ, 4, "没有找到相关FAQ"
            )
            faq_results = None  # 分数太低，不使用
        
        # Step 5: 表格定位 - 保持不变
        located_tables = self.locate_table(
            entity_enriched_query, request, knowledge_base_ids
        )
        step5 = self._create_step(
            WorkflowStepType.LOCATE_TABLE,
            5,
            [{"table_name": t["table_name"], "score": t["score"]} for t in located_tables] # 创建表格定位步骤，显示表名和分数用于调试
        )
        
        # Step 6: SQL生成 - 保持不变
        enhanced_input_messages = copy.deepcopy(extracted_messages)
        enhanced_input_messages[-1].content = entity_enriched_query
        
        # 构建表结构字符串
        table_schema = ""
        if located_tables:
            schemas = []
            # 只取前两张表，避免过长
            for table in located_tables[:2]:
                schemas.append(f"表名: {table['table_name']}\n{table['table_schema']}") # 提取表名和表结构，用于生成SQL
            table_schema = "\n\n".join(schemas) # 多表之间用双换行分隔
        
        response = self.generate_sql(
            table_schema=table_schema,
            input_messages=enhanced_input_messages,
            faq_results=faq_results,
            business_context=business_context,
            thinking=thinking,
        )
        
        step6 = self._create_step(
            WorkflowStepType.GENERATE_SQL, 6, "SQL生成完成"
        )
        
        # 构造最终响应 - 保持不变
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
            steps=[step1, step2, step3, step4, step5, step6],
        )

    async def do_stream(
        self,
        input_messages: List[ChatMessage],
        use_reranker: bool = True,
        reranker_info: Optional[RerankerInfo] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        thinking: bool = False,
    ):
        """流式生成 - 保持不变"""
        try:
            # 执行完整工作流
            response = self.do_generate(
                input_messages=input_messages,
                use_reranker=use_reranker,
                reranker_info=reranker_info,
                knowledge_base_ids=knowledge_base_ids,
                thinking=thinking,
            )
            
            # 如果是追问，直接返回
            if response.follow_up_num > 0:
                yield f"data: {response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # 流式输出结果
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                # 分块输出
                chunk_size = 50
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
                
                # 发送完成信号
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