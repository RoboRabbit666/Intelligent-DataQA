# coding: utf-8
"""
工作流增强版
"""
import copy
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from typing import List, Optional, Dict, Tuple
import re as regex_module
import copy
import traceback
import numpy as np
from datetime import datetime

# ========== 新增导入 ==========
import pickle
import os
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

from czce_ai.knowledge import SearchType, SQLSchemaKnowledge
from app.core.components import (
    mxbai_reranker,
    embedder,
    tokenizer,
    qwen3_llm,
    qwen3_thinking_llm,
)
from app.core.components.query_optimizer import (
    OptimizedQuery,
    QueryOptimizationType,
    QueryOptimizer,
    DataQAOptimizedQuery,
)

from resources import (
    USER_DICT_PATH,
    SYNONYM_DICT_PATH,
    STOP_WORDS_PATH,
    NER_PATTERNs_PATH,
)
from app.models import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatReference,
    ChatStep,
    ChatUsage,
)
from data_qa.prompt import dataqa_prompt
# ==============================

from app.core.components import mxbai_reranker, sql_kb, tokenizer, minio, embedder, document_kb
from app.config.config import settings
from czce_ai.embedder.bgem3 import BgeM3Embedder

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

# ========== 新增导入NLP工具 ==========
from czce_ai.nlp import NLPToolkit
from resources import (
    USER_DICT_PATH,
    SYNONYM_DICT_PATH,
    STOP_WORDS_PATH,
    NER_PATTERNs_PATH,
)
# =====================================

from .entities import WorkflowStepType
from .prompt import dataqa_prompt

def cosine_similarity(a, b):
    # 计算余弦相似度公式：cos(θ) = (A·B) / (|A|×|B|)
    # A·B：两个向量的点积
    # |A|、|B|：两个向量的模长（范数）
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
        history_round: int = 1,
        reranking_threshold: float = 0.2,
        config: Optional[WorkflowConfig] = None,
        knowledge_id: Optional[str] = "3cc33ed2-21fb-4452-9e10-528867bd5f99",
        bucket_name: Optional[str] = "czce-ai-dev",
        collection: Optional[str] = "hybrid_sql",
        use_cache: bool = True
    ):
        self.knowledge_id = knowledge_id
        self.bucket_name = bucket_name
        self.url = 'http://10.251.146.131:19530'
        self.reranking_threshold = reranking_threshold
        self.history_round = history_round
        self.ans_client = ans_llm
        self.ans_thinking_client = ans_thinking_llm
        self.collection = collection
        self.query_optimizer = QueryOptimizer(query_llm)
        self.config = config or WorkflowConfig()
        
        # ========== 新增：初始化FAQ数据 ==========
        self.faq_data = []
        self.use_cache = use_cache
        self.cache_file = Path(__file__).parent.parent / "test_data" / "tables" / "faq_cache.pkl"
        self._load_faqs()
        # ==========================================

    def _create_step(
        self, step_type: WorkflowStepType, number: int, prompt: Any
    ) -> ChatStep:
        """创建工作流步骤"""
        step_names = {
            WorkflowStepType.FOLLOW_UP: "问题追问",
            WorkflowStepType.MODIFY_QUERY: "问题改写",
            WorkflowStepType.ENTITY_RECOGNITION: "问题实体识别",
            # ========== 新增：FAQ语义搜索步骤 ==========
            WorkflowStepType.SEMANTIC_SEARCH_FAQ: "语义搜索FAQ",
            # =========================================
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

    # ========== 增强版实体识别（替换原函数） ==========
    def entity_recognition(self, query: str):
        """
        ========== 增强版实体识别（替换原函数） ==========
        修复：合约代码识别问题
        """
        if not self.config.enable_entity_recognition:
            return query
        try:
            enhanced_query = query
            
            # 使用增强的NLPToolkit
            nlp_tokenizer = NLPToolkit(
                user_dict_path=USER_DICT_PATH, 
                syn_dict_path=SYNONYM_DICT_PATH,
                stop_words_path=STOP_WORDS_PATH,
                patterns_path=NER_PATTERNs_PATH
            )
            entity_list = nlp_tokenizer.recognize(query)
            
            for entity in entity_list:
                if entity['id'] != '' and entity['text'] in enhanced_query:
                    # 特殊处理合约代码
                    if entity['label'] == '合约':
                        normalized_code = regex_module.sub(r'[-_\.\s/]+', '', entity['text'].upper())
                        substring = f"{normalized_code}(合约)"
                    else:
                        substring = f"{entity['id']}({entity['label']})"
                    
                    enhanced_query = enhanced_query.replace(entity['text'], substring, 1)
            
            return enhanced_query
        except Exception as e:
            logger.error(f"Entity recognition Error: {e}")
            return query
    # ==================================================

    # ========== 新增：FAQ相关方法（4个） ==========
    def _load_faqs(self):
        """
        加载所有SQL知识库文件 - 优先从缓存加载，如果缓存不存在则直接重新计算
        功能：从指定目录加载SQL知识库文件，解析问题和SQL语句，并计算嵌入向量. 通过embedder计算每个FAQ问题的嵌入向量，并存储在faq_data中

        Args:
            None
        Returns:
            None
        """
        # 尝试从缓存加载
        if self.use_cache and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.faq_data = cache_data['faq_data']
                    cache_time = cache_data.get('created_at', 'unknown')
                    logger.info(f"从缓存加载了{len(self.faq_data)}个FAQ (创建时间: {cache_time})")
                    return  # 成功加载缓存，直接返回
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}，将重新计算")
        
        # 缓存不存在或加载失败，重新计算
        logger.info("开始计算FAQ嵌入向量...")
        start_time = datetime.now()

        # Get path relative to the module file, not the current working directory
        current_file_dir = Path(__file__).parent
        tables_dir = current_file_dir.parent / "test_data" / "tables"
        
        if not tables_dir.exists():
            logger.warning(f"Tables directory not found: {tables_dir}")
            return
        
        for table_dir in tables_dir.iterdir():
            if not table_dir.is_dir():
                continue
                
            # 查找SQL知识库文件
            for file_path in table_dir.glob("*sql*知识库.txt"):
                table_name = table_dir.name
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析Q&A对
                pattern = r'问题[:：](.*?)\n(?:--.*?\n)*((?:WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*?);'
                matches = regex_module.findall(pattern, content, regex_module.DOTALL | regex_module.IGNORECASE)

                for question, sql in matches:
                    question = question.strip()
                    sql = sql.strip()
                    
                    # 对FAQ问题进行实体识别增强并计算嵌入向量
                    enhanced_question = self.entity_recognition(question)
                    embedding = embedder.get_embedding(enhanced_question)
                    
                    self.faq_data.append({
                        'question': enhanced_question,
                        'sql': sql,
                        'table': table_name,
                        'embedding': np.array(embedding)
                    })
        
        # === 计算完成，保存到缓存 ===
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"加载了{len(self.faq_data)}个FAQ，耗时{elapsed_time:.2f}秒")
        
        # 保存缓存
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
            logger.info(f"FAQ缓存已保存到: {self.cache_file}")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def refresh_faq_cache(self):
        """手动刷新缓存"""
        logger.info("开始刷新FAQ缓存...")
        
        # 删除旧缓存文件
        if self.cache_file.exists():
            os.remove(self.cache_file)
            logger.info("已删除旧缓存")
        
        # 清空当前数据
        self.faq_data = []
        
        # 重新加载（会自动创建新缓存）
        self._load_faqs()
        logger.info("FAQ缓存刷新完成")


    def semantic_search_faq(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        语义搜索FAQ函数
        
        功能：在FAQ知识库中找到与用户查询最相似的问题和对应的SQL语句
        
        Args:
            query: 用户输入的查询问题
            top_k: 返回最相似的前K个结果，默认3个
            
        Returns:
            List[Dict]: 包含相似FAQ的列表，每个字典包含问题、SQL、表名和相似度
        """
        
        # 1. 检查FAQ数据是否存在
        if not self.faq_data:
            return []  # 如果没有加载任何FAQ数据，直接返回空列表
        
        # # 2. 查询预处理 - 实体识别增强
        # enhanced_query = self.entity_recognition(query)
        # # 例如："查询FG2509成交量" -> "查询FG2509(合约)成交量"
        
        # 3. 将用户查询（注：此处的查询应该是实体识别增强后的查询）转换为嵌入向量
        query_embedding = np.array(embedder.get_embedding(query))
        # 得到一个高维向量表示，例如：[0.1, 0.2, 0.3, ..., 0.9]
        
        # 4. 计算查询与所有FAQ的相似度
        similarities = []  # 存储相似度分数
        
        for faq in self.faq_data:
            similarities.append(cosine_similarity(query_embedding, faq['embedding']))
            # 相似度范围：-1到1，越接近1越相似
        
        # 5. 获取相似度最高的Top-K个索引
        # np.argsort()：返回排序后的索引数组（从小到大）
        # [-top_k:]：取最后K个（即最大的K个）
        # [::-1]：反转数组（从大到小排列）
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 举例：如果similarities = [0.2, 0.8, 0.1, 0.9, 0.3]，top_k=3
        # argsort() -> [2, 0, 4, 1, 3]  (索引按相似度从小到大)
        # [-3:] -> [4, 1, 3]            (取最后3个)
        # [::-1] -> [3, 1, 4]           (反转，变成从大到小)
        
        # 6. 构建结果列表
        results = []
        for idx in top_indices:
            # 只返回相似度大于0.5的结果（基本阈值过滤）
            if similarities[idx] > 0.5:
                results.append({
                    'question': self.faq_data[idx]['question'],    # 原始问题
                    'sql': self.faq_data[idx]['sql'],              # 对应SQL语句
                    'table': self.faq_data[idx]['table'],          # 来源表名
                    'similarity': float(similarities[idx])         # 相似度分数
                })
        
        # 7. 返回结果
        return results
        # 返回格式示例：
        # [
        #     {
        #         'question': '查询FG2509的成交额',
        #         'sql': 'SELECT trd_amt FROM ... WHERE comd_code = "FG2509"',
        #         'table': '郑商所合约信息统计表',
        #         'similarity': 0.92
        #     },
        #     {
        #         'question': '统计期货成交量',
        #         'sql': 'SELECT SUM(trd_qty) FROM ...',
        #         'table': '期货交易统计表',
        #         'similarity': 0.76
        #     }
        # ]

    # ================================================

    def modify_query(
        self,
        input_messages: List[ChatMessage],
        enable_follow_up: bool,
    ) -> DataQAOptimizedQuery:
        """问题改写（保持不变）"""
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

    def domain_knowledge_search(
        self, query: str, document_id: Optional[str] = None
    ) -> str:
        """搜索领域知识（保持不变）"""
        try:
            embedder = BgeM3Embedder(base_url=settings.embedder.base_url, api_key=settings.embedder.api_key)
            query_embedding = embedder.get_embedding(query)
            domain_doc = document_kb.get_by_ids(collection=self.config.domain_collection, doc_id=document_id)
            domains = domain_doc[0].data.content
            sentences = [s.strip() for s in domains.split("\n") if s.strip()]
            sentence_embeddings = [embedder.get_embedding(s) for s in sentences]
            similarities = [cosine_similarity(query_embedding, sent_emb) for sent_emb in sentence_embeddings]
            most_similar_idx = np.argmax(similarities)
            most_similar_sentence = sentences[most_similar_idx]
            similarity_score = similarities[most_similar_idx]
            return most_similar_sentence, similarity_score
        except Exception as e:
            logger.error(f"Domain knowledge search Error:{e}")
            traceback.print_exc()
            raise e

    def locate_table(
        self, query: str, knowledge_base_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """定位表格（保持不变）"""
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

    def generate_single_table_prompt(self, tables: List[Dict[str, Any]]) -> str:
        """生成单表查询的prompt（保持不变）"""
        table_info = tables[0].get("table_info", "")
        table_prompt = f"已知如下数据表信息:\n{table_info}\n"
        return table_prompt

    def generate_sql(
        self,
        table_schema: str,
        input_messages: List[ChatMessage],
        thinking: Optional[bool] = False,
    ):
        """生成SQL（保持不变）"""
        query = input_messages[-1].content
        content = dataqa_prompt.format(table_schema=table_schema, question=query)
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
        """处理追问逻辑（保持不变）"""
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
                DataQAChatCompletionResponse(
                    choices=choices,
                    steps=[step],
                    follow_up_num=follow_up_num,
                ),
                optimized_query,
            )
        elif follow_up_num > self.config.follow_up_round:
            input_messages = self._extract_input_messages(request)
            recent_messages = input_messages[-self.config.follow_up_round * 2:]
            manual_query = " ".join(
                [msg.content for msg in recent_messages if msg.role == "user"]
            )
            return None, DataQAOptimizedQuery(
                original_query=optimized_query.original_query,
                rewritten_query=manual_query or optimized_query.rewritten_query,
                is_sufficient=True,
            )
        return None, optimized_query

    def do_generate(
        self,
        request: DataQACompletionRequest,
        enable_follow_up: bool = True,
        knowledge_base_ids: Optional[List[str]] = None,
        thinking: Optional[bool] = False,
    ) -> DataQAChatCompletionResponse:
        """
        生成回答
        ========== 最小化改动：仅在实体识别后插入FAQ搜索 ==========
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
        
        # Step3: semantic_search_faq（新增步骤）
        # 使用增强的实体识别查询进行FAQ语义搜索
        # 注意：这里的entity_enriched_query是经过实体识别增强的查询
        #       这一步骤在原有流程中是没有的
        #       目的是在查询表格之前，先尝试匹配FAQ
        #       如果FAQ匹配成功，则直接返回结果，避免不必要的表格查询
        #       如果没有匹配成功，则继续原有流程
        #       这样可以提高查询效率，减少不必要的计算
        # ========== 新增：FAQ快速路径 ==========
        faq_results = self.semantic_search_faq(entity_enriched_query, top_k=1)
        if faq_results and faq_results[0]['similarity'] >= 0.85:
            # Step3: semantic_search_faq（FAQ语义搜索）
            step3 = self._create_step(
                WorkflowStepType.SEMANTIC_SEARCH_FAQ, 3, f"找到高相似度FAQ：{faq_results[0]['question']}，相似度：{faq_results[0]['similarity']:.2f}"
            )
            # 直接返回FAQ结果，不再进行后续步骤
            # FAQ直接命中，快速返回
            best_faq = faq_results[0]
            response_content = f"""基于知识库匹配（相似度：{best_faq['similarity']:.2f}）：

```sql
{best_faq['sql']}
```

来源表：{best_faq['table']}"""
            
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
                id=f"faq-{int(datetime.now().timestamp())}",
                model="faq",
                created=int(datetime.now().timestamp()),
                choices=choices,
                usage=ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                steps=[step1, step2, step3],
            )

        # ========================================
        else:
            # 如果没有FAQ匹配成功，则继续原有流程
            if faq_results:
                step3 = self._create_step(
                    WorkflowStepType.SEMANTIC_SEARCH_FAQ,
                    3,
                    f"找到FAQ：{faq_results[0]['question']}，相似度：{faq_results[0]['similarity']:.2f}，但不满足阈值"
                )
            else:
                step3 = self._create_step(
                    WorkflowStepType.SEMANTIC_SEARCH_FAQ,
                    3,
                    "没有找到相关FAQ"
                )

            # Step4: locate table（继续原流程）
            located_table = self.locate_table(entity_enriched_query, knowledge_base_ids)
            step4 = self._create_step(WorkflowStepType.LOCATE_TABLE, 4, located_table)

            # Step5: generate single table prompt
            single_table_prompt = self.generate_single_table_prompt(located_table)
            
            # ========== 新增：如果有FAQ且大于等于0.7，添加到prompt参考 ==========
            if faq_results and faq_results[0]['similarity'] >= 0.7:
                single_table_prompt += f"\n参考示例:\n问题：{faq_results[0]['question']}\nSQL：{faq_results[0]['sql']}\n"
            # ========================================================

            step5 = self._create_step(
                WorkflowStepType.GENERATE_PROMPT, 5, single_table_prompt
            )
            
            # Step6: generate_sql
            response = self.generate_sql(
                table_schema=single_table_prompt,
                input_messages=input_messages,
                thinking=thinking,
            )
            step6 = self._create_step(WorkflowStepType.GENERATE_SQL, 6, response)

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
            steps=[step1, step2, step3, step4, step5, step6],
        )