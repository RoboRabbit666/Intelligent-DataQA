from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

from typing import List, Optional
import re
import copy
import traceback
import spacy
from spacy.pipeline import EntityRuler

from czce_ai.knowledge import SearchType, SQLSchemaKnowledge
from czce_ai.nlp.nlp.nlp import NLPToolkit
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


    def sql_knowledge(self):
        sql_kl = SQLSchemaKnowledge(tokenizer, embedder, self.url, mxbai_reranker)
        return sql_kl

    def entity_recognition(self, query: str):
        """Enhanced entity recognition using REGEX patterns
        
        Args:
            query: 本轮问题
        Returns:
            query: 增加实体识别后的query
        """

        try:
            enhanced_query = query
            
            tokenizer = NLPToolkit(
                user_dict_path=USER_DICT_PATH, 
                syn_dict_path=SYNONYM_DICT_PATH,
                stop_words_path=STOP_WORDS_PATH,
                patterns_path=NER_PATTERNS_PATH
            )
            entity_list = tokenizer.recognize(query)
            
            for entity in entity_list:
                if entity['id'] != '' and entity['text'] in enhanced_query:
                    if entity['label'] == '合约':
                        # 合约代码标准化
                        normalized_code = re.sub(r'[-_\.\s/]+', '', entity['text'].upper())
                        substring = f"{normalized_code}(合约)"
                    else:
                        # 其他实体正确格式化
                        substring = f"{entity['id']}({entity['label']})"
                    
                    enhanced_query = enhanced_query.replace(entity['text'], substring, 1)
            
            logger.info(f"实体识别完成: {query} -> {enhanced_query}")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"实体识别失败: {e}")
            return query
        

    def modify_query(
        self,
        input_messages: List[ChatMessage],
    ) -> OptimizedQuery:
        """问题改写
        
        Args:
            input_messages: 输入的消息列表
        Returns:
            out_messages: 最后一个用户query被修改的消息列表
        """
        
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

    # 定位表
    def locate_table(
        self,
        query: str,
    ) -> List[ChatReference]:
        """
        根据查询内容定位到相关的表格
        :return: 表格标题及分数
        """
        
        sql_kl = self.sql_knowledge()  # 获取SQLSchemaKnowledge实例
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
                lambda x: {'chunk_uuid':x.chunk_id,'table_name':x.data.table_name, 'score':x.reranking_score},
                ranked_tables,
            )
        )
        return tables
    

    #生成单表查询的prompt
    def generate_single_table_prompt(
        self,
        chunk_id:str
    ):
        # 从数据库中获取该表的字段信息，该字段信息需要包含完整的字段信息（包括英文名称，中文名称，解释等）
        # 用fsql生成的prompt的一部分
        sql_kl = self.sql_knowledge()
        table_content = sql_kl.get_by_ids(self.collection,chunk_id)
        table_info = table_content[0].data.table_info
        # 生成prompt
        table_prompt = f"已知如下数据表信息: \n{table_info}\n"
        return table_prompt
    

    def extract_info(
        self,
        text:str,
        pattern:str
    ):
        # 使用正则表达式提取信息
        extract_pattern = re.compile(pattern,re.DOTALL)
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
        content=dataqa_prompt.format(table_schema=table_schema,question=query)
        #print(content)
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
        """生成回答"""
        # 保留最后中间的对话，中间的对话最多保留 self.history_round * 2 轮
        if len(input_messages[1:-1]) > self.history_round * 2:
            del input_messages[1 : -1 - self.history_round * 2]

        # step1 modify_query
        optimized_input_messages = self.modify_query(input_messages)
        step1 = ChatStep(
            key="modify_query",
            name="改写问题",
            number=1,
            prompt=optimized_input_messages.rewritten_query,
            finished=True,
        )

        # step2 query entity recognition
        query = optimized_input_messages[-1].content
        entitled_query = self.entity_recognition(query)
        optimized_input_messages[-1].content = entitled_query
        step2 = ChatStep(
            key="query_entity_recognition",
            name="问题实体识别",
            number=2,
            prompt=entitled_query,
            finished=True,
        )

        # step3 locate table
        located_table = self.locate_table(optimized_input_messages.rewritten_query)
        step3 = ChatStep(
            key="locate_table",
            name="定位表格",
            number=3,
            prompt=located_table,
            finished=True,
        )

        # step4 generate single table prompt
        single_table_prompt = self.generate_single_table_prompt(located_table[0]['chunk_uuid'])
        step4 = ChatStep(
            key="generate_single_table_prompt",
            name="生成单表提示词",
            number=4,
            prompt=single_table_prompt,
            finished=True,
        )

        # step5 generate_sql
        response = self.generate_sql_code(single_table_prompt, optimized_input_messages, thinking)
        step5 = ChatStep(
            key="generate_sql",
            name="生成SQL",
            number=5,
            prompt=response,
            finished=True,
        )

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
                    ),
                ),
                response.choices,
            )
        )

        return ChatCompletionResponse(
            id=response.id,
            model=response.model,
            created=response.created,
            choices=choices,
            usage=usage,
            steps=[step1, step2, step3, step4, step5],
        )
