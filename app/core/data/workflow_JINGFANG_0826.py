#coding: utf-8
import copy
import re
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
from .prompt import dataqa_prompt

#-------------新增imports------------
from spacy.pipeline import EntityRuler
from datetime import datetime
import numpy as np
import pickle
import os
import json
import spacy

from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

from app.config.config import settings
from czce_ai.embedder import BgeM3Embedder
from app.core.components import mxbai_reranker, sql_kb, tokenizer, minio, embedder, document_kb, qwen3_11m, qwen3_thinking_1lm
from czce_ai.knowledge import SearchType, SQLSchemaKnowledge
from czce_ai.nlp import NLPToolkit
from enum import Enum
from resources import (
    USER_DICT_PATH,
    SYNONYM_DICT_PATH,
    STOP_WORDS_PATH,
    NER_PATTERNs_PATH,
)
#-----------------------------------

#-------新增:余弦相似度函数-----------------
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
#-------新增余弦相似度函数-----------------

@dataclass
class WorkflowConfig:
    """ 工作流配置类-统一管理参数、魔法数字等"""
    history_round: int = 1
    follow_up_round: int = 1
    reranking_threshold: float = 0.2
    collection: Optional[str] = "hybrid_sql"
    domain_collection: Optional[str] = 'domain_kl' #新增领域知识
    api_collection: Optional[str] = 'api_kl' #新增API知识
    sql_schema_collection: Optional[str] = "hybrid_sql" #新增SQL知识
    max_table_results: int = 3
    enable_entity_recognition: bool = True
    enable_reranker: bool = True

    #--------------新增相关配置-------------------
    knowledge_id: str = "3cc33ed2-21fb-4452-9e10-528867bd5f99"
    bucket_name: str = "czce-ai-dev"
    use_cache: bool = True # 是否使用FAQ缓存
    cache_file: Path = Path.cwd().parent / "test_data" / "tables" / "faq_cache.pkl" #缓存FAQ文件路径
    #----------------------------------------------------------------------------------

    def __post_init__(self):
        if self.history_round < self.follow_up_round:
            raise ValueError(
                f"history_round({self.history_round}) must be > follow_up_round({self.follow_up_round})"
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

        #---------------------------新增FAQ相关属性------------------
        self.faq_data = [] #存储FAQ的问题、SQL和嵌入向量
        self.use_cache = self.config.use_cache #是否使用FAQ缓存
        # 缓存FAQ文件路径
        self.cache_file = self.config.cache_file
        # 初始化时加载FAQ(优先从缓存加载)
        self._load_faqs()
        #------------------------------------------------------------------------

    #
    def _create_step(
        self, step_type: WorkflowStepType, number: int, prompt: Any
    ) -> ChatStep:
        """ “创建工作流步骤”"""
        step_names = {
            WorkflowStepType.FOLLOW_UP:"问题追问",
            WorkflowStepType.MODIFY_QUERY:"问题改写",
            WorkflowStepType.ENTITY_RECOGNITION:"问题实体识别",
            #-------------------新增--------------------
            WorkflowStepType.SEMANTIC_SEARCH_FAQ:"语义搜索FAQ",
            WorkflowStepType.LOCATE_API:"API定位",
            #-------------------新增--------------------
            WorkflowStepType.LOCATE_TABLE:"表格定位",
            # WorkflowStepType.GENERATE_PROMPT:"上下文工程",
            WorkflowStepType.GENERATE_SQL:"SQL生成",
        }
        #---------------------------新增-------------------
        # 如果步骤类型不在预定义的名称中,则使用默认名称
        # 转换prompt为字符串（如果不是字符串的话）
        if not isinstance(prompt, str):
            if isinstance(prompt, list):
                # 对于列表，转换为JSON字符串
                prompt = json.dumps(prompt, ensure_ascii=False, indent=2)
            elif isinstance(prompt, dict):
                # 对于字典，转换为JSON字符串
                prompt = json.dumps(prompt, ensure_ascii=False, indent=2)
            else:
                # 其他类型，直接转换为字符串
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
        #
        """提取输入信息列表"""
        return request.messages[-self.config.history_round * 2:]
    

    def modify_query(
        self,
        input_messages: List[ChatMessage],
        enable_follow_up: bool,
    ) -> DataQAOptimizedQuery:
        """问题改写
        Args:
            input_messages (List[ChatMessage]):输入的消息列表
            enable_follow_up (bool):是否启用追问功能
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
                query = last_user_msg.content if last_user_msg else "",
                chat_history=input_messages_copy[:last_index + 1],
                optimization_type=optimization_type,
            )

            return optimized_query
        
        except Exception as e:
            logger.error(f"Modify query Error:{e}")
            traceback.print_exc()
            raise e
        
    
    #============================== 新增：短格式合约扩展 ==========================
    def _expand_short_contracts(self, query: str) -> str:
        """
        将短格式合约代码扩展为标准长格式。

        使用rolling decade rule将单位年份数字转换为完整年份：
        - 基于当前年份选择最接近的十年
        - 平局时选择较早的十年

        Args:
            query (str): 输入查询字符串

        Returns:
            str: 处理后的查询字符串，短格式合约代码已扩展

        Examples:
             # 2025年场景
            _expand_short_contracts("查询AP509的成交量")
            "查询AP2509的成交量"
            _expand_short_contracts("CU601期货价格") 
            "CU2601期货价格"

        """
        def expand_match(match):
            """处理单个短格式匹配的回调函数"""
            prefix = match.group(1)             # 品种代码，如 "AP"
            year_digit = int(match.group(3))    # 年份数字，如 "5"
            month = match.group(4)              # 月份，如 "09"

            # Rolling decade rule: 选择距离当前年份最近的选项
            current_year = datetime.now().year
            current_decade = (current_year // 10) * 10

            option1 = current_decade + year_digit           # 当前十年
            option2 = current_decade + 10 + year_digit      # 下个十年

            # 选择距离最近的，平局选较早的
            if abs(option1 - current_year) <= abs(option2 - current_year):
                target_year = option1
            else:
                target_year = option2

            # 构建完整合约代码
            result = f"{prefix}{str(target_year)[-2:]}{month}"
            logger.info(f"短格式扩展: {match.group(0)} -> {result}")
            return result

        # 正则模式匹配短格式合约代码：品种代码(1-3字母) + 年份(1位数字) + 月份(01-12)
        # 借鉴已验证的长格式正则，将年份从{2}修改为{1}
        short_pattern = r'(?<![A-Za-z])([A-Za-z]{1,3})(([0-9]{1})(0[1-9]|1[0-2]))(?![A-Za-z0-9])'
        result = re.sub(short_pattern, expand_match, query)
        return result
    #============================== 新增：短格式合约扩展 ==========================

    #------------------------------更新: entity_recognition----------------------------------------
    def entity_recognition(self, query: str):
        """增强版实体识别 ...
        增强:在原有实体类型识别基础上增强合约代码识别能力
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
                if entity['id'] != '' and entity['text'] in enhanced_query:
                    if entity['label'] == '合约':
                        #标准化合约代码- 将所有分隔符转换为标准模式
                        normalized_code = re.sub(r'[-_\.\s/]+', '', entity["text"].upper())
                        substring = f"{normalized_code} ({entity['label']})"
                    else:
                        #其他实体正确格式化
                        substring = f"{entity['text']} ({entity['label']})"
                    enhanced_query = enhanced_query.replace(entity['text'], substring, 1)
            logger.info(f"实体识别完成: {query} -> {enhanced_query}")
            return enhanced_query
        except Exception as e:
            logger.error(f"实体识别失败:{e}")
            #实体识别失败返回原查询,不中断流程
            return query
    #----------------------------------entity_recognition----------------------------------------

    #-------------------新增FAQ相关方法(4个)-------------------
    def _load_faqs(self):
        """    加载FAQ知识库文件- 优先从缓存加载,否则重新计算"""
        #尝试从缓存加载
        if self.use_cache and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.faq_data = cache_data['faq_data']
                    cache_time = cache_data.get('created_at', 'unknown')
                    logger.info(f"成功从缓存加载了{len(self.faq_data)}个FAQ(创建时间:{cache_time})")
                    return # 成功加载缓存,直接返回
            except Exception as e:
                logger.warning(f"加载缓存失败:{e},将重新计算")
        #缓存不存在或加载失败,重新计算
        logger.info("开始计算FAQ嵌入向量...")
        start_time = datetime.now()

        tables_dir = Path.cwd().parent / "test_data" / "tables"
        if not tables_dir.exists():
            logger.warning(f'Tables directory not found: {tables_dir}')
        #
        for table_dir in tables_dir.iterdir():
            if not table_dir.is_dir():
                continue
            # 查询SQL知识库文件
            for file_path in table_dir.glob("*sql*知识库.txt"):
                table_name = table_dir.name # 获取表中文名
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # 解析QA对
                    pattern = r'问题[:：](.*?)\n(?:--.*?\n)*((?:WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*?)(?=\n\n问题[:：]|\n\n$|$)'
                    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                    #处理每一对问题和SQL
                    for question, sql in matches:
                        question = question.strip() #移除文本问题首尾空白
                        sql = sql.strip() # 移除SQL语句首尾空白
                        #对FAQ/问题进行实体识别增强并计算嵌入向量
                        enhanced_question = self.entity_recognition(question)
                        embedding = embedder.get_embedding(enhanced_question) #计算嵌入向量
                        # 存储到faq_data数据库
                        self.faq_data.append({
                            'question': enhanced_question,#知识库问题
                            'sql': sql, # SQL语句
                            'table': table_name, #表中文名
                            'embedding': np.array(embedding) # 向量化表示
                        })
        #计算完成,保存到缓存
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"成功加载了{len(self.faq_data)}个FAQs,耗时{elapsed_time:.2f}秒")
        #保存缓存
        if self.use_cache:
            self._save_cache()

    def _save_cache(self):
        """保存FAQ缓存"""
        try:
            cache_data = {
                'faq_data': self.faq_data,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"FAQ缓存已成功保存到: {self.cache_file}")
        except Exception as e:
            logger.error(f"保存缓存失败:{e}")

    def refresh_faq_cache(self):
        """手动刷新缓存"""
        logger.info("开始刷新FAQ缓存...")
        #删除旧的缓存文件
        if self.cache_file.exists():
            os.remove(self.cache_file)
            logger.info("已成功删除旧缓存")
        #清空当前数据
        self.faq_data = []
        #重新加载(会自动创建新缓存)
        self._load_faqs()
        logger.info("FAQ缓存刷新完成")

    def semantic_search_faq(self, query: str, top_k: int = 3) -> List[dict]:
        """语义搜索FAQ - 在FAQ知识库中找到与用户查询最相似的问题和对应的SQL语句
        Args:
            query (str):用户输入的查询
            top_k (int, optional):最相似的前K个结果,默认为前3个
        Returns:
            List[dict]:返回前K个相似FAQ的列表,每个字典包含FAQ知识库问题、SQL语句、表中文名和相似度
        """
        #检查FAQ数据是否存在
        if not self.faq_data:
            return [] # 如果没有加载任何FAQ数据,直接返回空列表
        # 将用户查询(注:此处的查询应是实体识别增强后的查询)转换为嵌入向量
        query_embedding = np.array(embedder.get_embedding(query))
        #计算查询与所有FAQ的相似度
        similarities = [] #存储相似度分数
        for faq in self.faq_data:
            # 计算余弦相似度: 范围-1到1之间,越接近1越相似
            similarities.append(cosine_similarity(query_embedding, faq['embedding']))
        # 获取top k相似度最高的top-K个索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        # 构建返回结果列表
        results = []
        for idx in top_indices:
            if similarities[idx] >= 0.5: #只返回相似度大于0.5的结果(基本阈值过滤)
                results.append({
                    'question': self.faq_data[idx]['question'],#FAQ知识库问题
                    'sql': self.faq_data[idx]['sql'], #对应SQL语句
                    'table': self.faq_data[idx]['table'],# 中文名
                    'similarity': float(similarities[idx]) #相似度分数
                })
        return results
    #-------------------新增FAQ相关方法(4个)-------------------
    #定位表
    def locate_table(
        self, 
        query: str, 
        request: DataQACompletionRequest,
        knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """  定位表格
        Args:
            query:查询内容
            knowledge_base_ids:查询的知识库ID
        Returns:
            表格信息列表
        """
        input_messages = self._extract_input_messages(request)
        search_content = '\n'.join([f"{msg.role}: {msg.content}" for msg in input_messages]) + '\nUser: ' + query
        logger.info(f"Locate Table - Search Content: {search_content}")

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


    def generate_sql(
        self,
        table_schema: str,
        input_messages: List[ChatMessage],
        faq_results: Optional[Dict[str, Any]] = None,
        faq_score: Optional[float] = 0.0,
        thinking: Optional[bool] = False,
    ):
        """   
        生成SQL

        Args:
            table_schema (str): 表格模式
            input_messages (List[ChatMessage]): 输入消息
            faq_results (Optional[Dict[str, Any]], optional): FAQ结果. Defaults to None.
            faq_score (Optional[float], optional): FAQ得分. Defaults to 0.0.
            thinking (Optional[bool], optional): 是否思考. Defaults to False.

        Returns:
            response: 返回生成的SQL响应
        """
        query = input_messages[-1].content
        content = dataqa_prompt.format(table_schema=table_schema, 
                                       question=query, 
                                       faq_results=faq_results,
                                       faq_score=faq_score)
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
        """”处理追问逻辑
        Args:
            optimized_query (DataQAOptimizedQuery):优化后的查询对象
            request (DataQACompletionRequest):数据问答请求对象
        Returns:
            Tuple[追问响应(如果需要),更新后的查询对象]
        """
        #计算追问轮数
        follow_up_num = (
            request.follow_up_num + 1 if not optimized_query.is_sufficient else 0
        )
        #需要追问且未超过最大追问轮数
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
                    id=f"follow-up-{int(datetime.now().timestamp())}",  # 添加id字段
                    model=request.model or "follow-up",  # 添加model字段
                    created=int(datetime.now().timestamp()),  # 添加created字段
                    choices=choices,
                    usage=ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),  # 添加usage字段
                    steps=[step],
                    follow_up_num=follow_up_num,
                ),
                optimized_query,
            )
        elif follow_up_num > self.config.follow_up_round:
            #超过追问轮数,使用本轮数据问答的会话内容进行检索
            input_messages = self._extract_input_messages(request)
            recent_messages = input_messages[-self.config.follow_up_round * 2:]
            manual_query = " ".join(
                [msg.content for msg in recent_messages if msg.role == "user"]
            )
            return None, DataQAOptimizedQuery(
                original_query=optimized_query.original_query,
                rewritten_query=manual_query or optimized_query.rewritten_query,
                is_sufficient=True, #超过轮数后认为查询已足够
            )
        #不需要追问
        return None, optimized_query
    
    # 定位api
    def locate_api(
        self, 
        query: str, 
        knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """  定位API
        Args:
            query:查询内容
            knowledge_base_ids:查询的知识库ID
        Returns:
            表格信息列表
        """
        try:
            ranked_api = document_kb.search(
                self.config.api_collection,
                query,
                knowledge_ids=knowledge_base_ids,
                top_k=self.config.max_table_results,
                use_reranker=self.config.enable_reranker,
            )

            apis = [
                {
                    "chunk_uuid": api.chunk_id,
                    "api_name": api.data.table_name,
                    "score": api.reranking_score,
                }
                for api in ranked_api
            ]
            return api_list
        
        except Exception as e:
            logger.error(f"Locate API Error: {e}")
            return []

    def do_generate(
        self,
        request: DataQACompletionRequest,
        enable_follow_up: bool = True,
        knowledge_base_ids: Optional[List[str]] = None,
        thinking: Optional[bool] = False,
    ) -> DataQAChatCompletionResponse:
        """  生成回答
        Args:
            input_messages (List[ChatMessage]):输入信息列表
            enable_follow_up(bool):是否启用追问功能
            knowledge_base_ids (Optional[List[str]]):知识库ID
            thinking(bool):是否开启thinking模式, 可选, default: False
        """
        #提取输入信息
        input_messages = self._extract_input_messages(request)

        # Step1: modify_query
        optimized_query = self.modify_query(
            input_messages=input_messages,
            enable_follow_up=enable_follow_up,
        )
        #处理追问逻辑
        if enable_follow_up:
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

        #Step3: semantic search FAQ
        faq_results = self.semantic_search_faq(entity_enriched_query, top_k=3)
        #---------------------新增:如果找到高相似度的FAQ(即0.9 =< 相似度),直接使用----------------------------
        if faq_results and len(faq_results) > 0 and faq_results[0]['similarity'] >= 0.9:
            step3 = self._create_step(
                WorkflowStepType.SEMANTIC_SEARCH_FAQ,3,"已找到高相似度(>=0.9)FAQ"
            )
            # FAQ直接命中,快速构造响应
            best_faq = faq_results[0]
            response_content = f"""以下是问题分析和sql参考示例:
                基于知识库最佳匹配(相似度: {best_faq['similarity']:.3f})
            ```sql:
            {best_faq['sql']}
            ```
            来源:{best_faq['table']}知识库
            匹配问题:{best_faq['question']}
            ---分析完成
            """
            
            choices = [
                ChatCompletionChoice(
                    finish_reason="stop",
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_content,
                        reasoning_content=None,
                        is_follow_up=False,
                    )
                )
            ]

            return DataQAChatCompletionResponse(
                id=f"faq-{int(datetime.now().timestamp())}",
                model="faq",
                created=int(datetime.now().timestamp()),
                choices=choices,
                usage=ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                steps=[step1, step2, step3],
            )
        #---------------------新增:如果找到高相似度的FAQ(即0.9 =< 相似度),直接使用----------------------------

        else:
            # 如果没找到高相似度FAQ,继续原流程
            #------------------------------新增:更新step3-------------------------------------------
            if faq_results:
                step3 = self._create_step(WorkflowStepType.SEMANTIC_SEARCH_FAQ,3,"找到相关FAQ, 但是不满足0.9阈值")
            else:
                step3 = self._create_step(WorkflowStepType.SEMANTIC_SEARCH_FAQ,3,"没有找到相关FAQ")
                faq_results = None
            #----------------------------------------------------------------------------------------

            # Step4: locate table
            located_table = self.locate_table(entity_enriched_query, request, knowledge_base_ids)
            step4 = self._create_step(WorkflowStepType.LOCATE_TABLE, 4, located_table)

            #-----------------------新增:如果有相关的FAQ(即0.7 =< 相似度 < 0.9),可添加参考---------------------------------
            # 创建input_messages副本并更新最后一条消息为实体识别增强后的查询
            enhanced_input_messages = copy.deepcopy(input_messages)
            enhanced_input_messages[-1].content = entity_enriched_query

            if faq_results and len(faq_results) > 0 and faq_results[0]['similarity'] >= 0.7:
                # step5: generate_sql (有FAQ参考)
                table_schema = located_table[0]['table_info'] if located_table else ""
                response = self.generate_sql(
                    table_schema=table_schema,
                    input_messages=enhanced_input_messages,  # 使用增强后的消息
                    faq_results=faq_results,  # 添加FAQ结果
                    faq_score=faq_results[0]['similarity'],  # FAQ最高相似度分数
                    thinking=thinking,
                )
                step5 = self._create_step(WorkflowStepType.GENERATE_SQL, 5, response)
            else:
                # step5: generate_sql (相似度 < 0.7或者没有FAQ参考，继续走原流程)
                table_schema = located_table[0]['table_info'] if located_table else ""
                response = self.generate_sql(
                    table_schema=table_schema,
                    input_messages=enhanced_input_messages,  # 使用增强后的消息
                    faq_results=None,  # 无FAQ参考
                    faq_score=0.0,
                    thinking=thinking,
                )
                logger.info(f"Print table schema: {table_schema}")
                step5 = self._create_step(WorkflowStepType.GENERATE_SQL, 5, response)
                
            #---------------------------------------------------------------------------------------------------------
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
        """异步问题改写
        Args:
            input_messages (List[ChatMessage]):输入的消息列表
            enable_follow_up (bool):是否启用追问功能
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
            
            optimized_query = self.query_optimizer.agenerate_optimized_query(
                query=input_messages_copy[-1].content if input_messages_copy else "",
                chat_history=input_messages_copy[:-1],
                optimization_type=optimization_type,
            )
            return optimized_query

        except Exception as e:
            logger.error(f"Modify query Error:{e}")
            traceback.print_exc()
            raise e
