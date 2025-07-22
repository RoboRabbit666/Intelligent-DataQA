from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

from typing import List, Optional, Tuple
import re
import copy
import traceback

from czce_ai.knowledge import SearchType, SQLSchemaKnowledge
from czce_ai.nlp import NLPToolkit
from czce_ai.llm.message import Message as ChatMessage
from czce_ai.llm.chat import LLMChat as LLMModel
from app.core.components import (
    mxbai_reranker,
    embedder,
    tokenizer,
)
from app.core.components.query_optimizer import (
    OptimizedQuery,
    QueryOptimizationType,
    QueryOptimizer,
)
from app.core.components import qwen3_llm, qwen3_thinking_llm
from resources import (
    USER_DICT_PATH,
    SYNONYM_DICT_PATH,
    STOP_WORDS_PATH,
    NER_PATTERNS_PATH,
)
from app.models import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatReference,
    ChatStep,
    ChatUsage,
)
from data_qa.prompt import dataqa_prompt
from czce_ai.utils.log import logger

# <<< ADDED: NER结果精炼器 >>>
# 这个类封装了您验证过的核心冲突解决算法，作为独立的工具使用。
class NERRefiner:
    """NER结果精炼器，用于解决实体冲突，融合多源结果。"""
    
    def resolve_conflicts(self, entities: List[Tuple]) -> List[Tuple]:
        if not entities:
            return []

        entities.sort(key=lambda x: (x[2], -len(x[0])))
        
        resolved = []
        for current_entity in entities:
            is_subsumed_or_inferior = False
            for i, existing_entity in enumerate(list(resolved)):
                if current_entity[2] < existing_entity[3] and current_entity[3] > existing_entity[2]:
                    if current_entity[2] >= existing_entity[2] and current_entity[3] <= existing_entity[3]:
                        is_subsumed_or_inferior = True; break
                    if current_entity[2] <= existing_entity[2] and current_entity[3] >= existing_entity[3]:
                        resolved.pop(i); continue
                    if current_entity[1] == 'CONTRACT_CODE' and existing_entity[1] != 'CONTRACT_CODE':
                         resolved.pop(i); continue
                    if existing_entity[1] == 'CONTRACT_CODE' and current_entity[1] != 'CONTRACT_CODE':
                        is_subsumed_or_inferior = True; break
            if not is_subsumed_or_inferior:
                resolved.append(current_entity)
        
        return sorted(resolved, key=lambda x: x[2])


class DataQaWorkflow:
    def __init__(
        self,
        ans_llm: LLMModel,
        ans_thinking_llm: LLMModel,
        query_llm: LLMModel,
        history_round: int = 1,
        reranking_threshold: float = 0.2,
        knowledge_id: Optional[str] = "3cc33ed2-21fb-4452-9e10-528867bd5f99",
        bucket_name: Optional[str] = "czce-ai-dev",
        collection: Optional[str] = "hybrid_sql"
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
        
        # <<< CHANGED: 初始化精炼器 >>>
        self.ner_refiner = NERRefiner()

    # <<< REPLACED: 融合了您改进思想的最终实体识别方法 >>>
    def entity_recognition(self, query: str):
        """实体识别 (混合策略：融合 NLPToolkit 和 您的改进算法)"""
        try:
            candidate_entities = []

            # --- 步骤1: 使用 NLPToolkit 进行基础识别 ---
            tokenizer_toolkit = NLPToolkit(user_dict_path=USER_DICT_PATH, syn_dict_path=SYNONYM_DICT_PATH, stop_words_path=STOP_WORDS_PATH)
            toolkit_results = tokenizer_toolkit.recognize(query)
            for entity_dict in toolkit_results:
                if entity_dict.get('id'):
                    entity_text = entity_dict['text']
                    entity_type = entity_dict['label']
                    for match in re.finditer(re.escape(entity_text), query):
                        candidate_entities.append(
                            (entity_text, entity_type, match.start(), match.end())
                        )

            # --- 步骤2: 使用增强的正则，捕获复杂合约代码 ---
            variant_patterns = [
                r'\b([A-Z]{1,3})\s*[-\s./_]\s*(\d{4})\b',
                r'\b([A-Z]{1,3})\((\d{4})\)\b',
                # <<< CHANGED: 增强的合约代码识别规则 >>>
                # 这条新规则可以识别 "ruAP2502" 这种由多个字母开头的代码
                # [a-zA-Z]{2,6} 匹配2到6个字母，如 ruAP, p, c, jm 等
                # (?:25|26)\d{2} 匹配年份（例如25, 26）和月份
                r'\b[a-zA-Z]{2,6}(?:25|26)\d{2}\b'
            ]
            for pattern in variant_patterns:
                for match in re.finditer(pattern, query, re.IGNORECASE):
                    candidate_entities.append(
                        (match.group(), 'CONTRACT_CODE', match.start(), match.end())
                    )

            # --- 步骤3: 使用您的核心算法精炼结果 ---
            final_entities = self.ner_refiner.resolve_conflicts(candidate_entities)
            
            # --- 步骤4: 安全地重建查询字符串 ---
            new_query = list(query)
            for entity_text, entity_type, start, end in reversed(final_entities):
                replacement = f"{entity_text}({entity_type})"
                new_query[start:end] = [replacement]
            
            return "".join(new_query)

        except Exception as e:
            logger.error(f"Hybrid Entity recognition Error:{e}")
            traceback.print_exc()
            return query

    # --- 后续所有方法保持完整，不做省略 ---

    def sql_knowledge(self):
        sql_kl = SQLSchemaKnowledge(tokenizer, embedder, self.url, mxbai_reranker)
        return sql_kl

    def modify_query(
        self,
        input_messages: List[ChatMessage],
    ) -> OptimizedQuery:
        try:
            input_messages_mq = copy.deepcopy(input_messages)
            optimized_query = self.query_optimizer.generate_optimized_query(
                query=input_messages[-1].content,
                chat_history=input_messages_mq[:-1],
                optimization_type=QueryOptimizationType.DATAQA,
            )
            return optimized_query
        except Exception as e:
            logger.error(f"Modify query Error:{e}")
            traceback.print_exc()
            raise e

    def locate_table(
        self,
        query: str,
    ) -> List[dict]:
        sql_kl = self.sql_knowledge()
        ranked_tables = sql_kl.search(
            self.collection,
            query,
            search_type=SearchType.hybrid,
            knowledge_ids=self.knowledge_id,
            top_k=3,
            use_reranker=True
        )
        tables = list(
            map(
                lambda x: {'chunk_uuid': x.chunk_id, 'table_name': x.data.table_name, 'score': x.reranking_score},
                ranked_tables,
            )
        )
        return tables
    
    def generate_single_table_prompt(
        self,
        chunk_id:str
    ):
        sql_kl = self.sql_knowledge()
        table_content = sql_kl.get_by_ids(self.collection, chunk_id)
        table_info = table_content[0].data.table_info
        table_prompt = f"已知如下数据表信息: \n{table_info}\n"
        return table_prompt
    
    def extract_info(
        self,
        text:str,
        pattern:str
    ):
        extract_pattern = re.compile(pattern, re.DOTALL)
        match = extract_pattern.search(text)
        if match:
            return match.group(1)
        else:
            return None

    def generate_sql_code(
        self,
        table_schema:str,
        input_messages: List[ChatMessage],
        thinking: Optional[bool] = False,
    ):
        query = input_messages[-1].content
        content = dataqa_prompt.format(table_schema=table_schema, question=query)
        system_msg = ChatMessage(
            role="system",
            content=content,
        )
        if thinking is True:
            response = self.ans_thinking_client.invoke(
                messages=[system_msg] + input_messages[:]
            )
        else:
            response = self.ans_client.invoke(messages=[system_msg] + input_messages[:])
        return response

    def do_generate(
        self,
        input_messages: List[ChatMessage],
        knowledge_base_ids: Optional[List[str]] = None,
        thinking: Optional[bool] = False,
    ):
        if len(input_messages[1:-1]) > self.history_round * 2:
            del input_messages[1 : -1 - self.history_round * 2]

        optimized_input = self.modify_query(input_messages)
        step1 = ChatStep(
            key="modify_query", name="改写问题", number=1,
            prompt=optimized_input.rewritten_query, finished=True,
        )

        entitled_query = self.entity_recognition(optimized_input.rewritten_query)
        step2 = ChatStep(
            key="query_entity_recognition", name="问题实体识别", number=2,
            prompt=entitled_query, finished=True,
        )
        
        # 使用带有实体标注的查询进行后续步骤
        modified_messages = copy.deepcopy(input_messages)
        modified_messages[-1].content = entitled_query

        located_table = self.locate_table(optimized_input.rewritten_query)
        step3 = ChatStep(
            key="locate_table", name="定位表格", number=3,
            prompt=str(located_table), finished=True,
        )

        single_table_prompt = self.generate_single_table_prompt(located_table[0]['chunk_uuid'])
        step4 = ChatStep(
            key="generate_single_table_prompt", name="生成单表提示词", number=4,
            prompt=single_table_prompt, finished=True,
        )

        response = self.generate_sql_code(table_schema=single_table_prompt, input_messages=modified_messages, thinking=thinking)
        step5 = ChatStep(
            key="generate_sql", name="生成SQL", number=5,
            prompt=str(response), finished=True,
        )

        usage = ChatUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        choices = list(
            map(
                lambda x: ChatCompletionChoice(
                    finish_reason=x.finish_reason, index=x.index,
                    message=ChatMessage(
                        role=x.message.role, content=x.message.content,
                        reasoning_content=x.message.reasoning_content,
                    ),
                ),
                response.choices,
            )
        )

        return ChatCompletionResponse(
            id=response.id, model=response.model, created=response.created,
            choices=choices, usage=usage,
            steps=[step1, step2, step3, step4, step5],
        )
    
    '''
    #llm:app/core/components.py
    query="苹果是在哪个交易所上市的"
    dataqa = DataQaWorkflow(llm=qwen3_llm,query=query)
    '''

    '''
    #测试locate_table函数
    chunk_id,table_name,tabel_score = dataqa.locate_table()
    print(chunk_id)
    print(table_name)
    print(tabel_score)
    '''


    '''
    #测试generate_single_table_prompt函数

    chunk_id = "1e7fcf17-cac8-5322-912e-b96900beae78"
    prompt = dataqa.generate_single_table_prompt(chunk_id)
    analysis_info,sql_code = dataqa.generate_sql_code(prompt,query)
    print(analysis_info)
    print('---------')
    print(sql_code)
    '''


    '''
    #测试entity_recognition函数
    query = "华泰期货在郑商所的持仓是多少?"
    query = dataqa.entity_recognition(query)
    print(query)
    '''