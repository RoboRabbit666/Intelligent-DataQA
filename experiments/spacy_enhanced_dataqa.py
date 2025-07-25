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
        
        # Initialize spaCy pipeline - SIMPLE SETUP
        self._setup_spacy_pipeline()

    def _setup_spacy_pipeline(self):
        """Setup spaCy pipeline with zh_core_web_md and contract code patterns"""
        try:
            # Load Chinese model
            self.nlp = spacy.load("zh_core_web_md")
            logger.info("✅ Loaded zh_core_web_md model")
        except OSError:
            logger.warning("⚠️ zh_core_web_md not found, trying zh_core_web_sm")
            try:
                self.nlp = spacy.load("zh_core_web_sm")
                logger.info("✅ Loaded zh_core_web_sm model")
            except OSError:
                logger.error("❌ No Chinese spaCy model found")
                raise

        # Add EntityRuler for contract codes - SIMPLE ADDITION
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        
        # Define contract code patterns - CONCISE BUT COMPLETE
        patterns = self._create_contract_patterns()
        ruler.add_patterns(patterns)
        
        logger.info(f"✅ Added {len(patterns)} contract code patterns to spaCy pipeline")

    def _create_contract_patterns(self):
        """Create contract code patterns for EntityRuler - SIMPLE AND EFFECTIVE"""
        
        # All major futures product codes
        products = [
            # 上期所
            'CU', 'AL', 'ZN', 'PB', 'SN', 'NI', 'AU', 'AG', 'RB', 'HC', 'SS', 'FU', 'BU', 'RU', 'NR', 'SP', 'AO', 'SC', 'LU', 'BC',
            # 大商所
            'A', 'B', 'M', 'Y', 'P', 'C', 'CS', 'JD', 'LH', 'FB', 'BB', 'L', 'V', 'PP', 'EG', 'EB', 'PG', 'BE', 'I', 'J', 'JM', 'LM',
            # 郑商所  
            'WH', 'PM', 'RI', 'LR', 'JR', 'CF', 'CY', 'SR', 'RS', 'OI', 'RM', 'AP', 'CJ', 'PK', 'ZC', 'FG', 'SA', 'MA', 'TA', 'UR', 'SM', 'SF', 'PF', 'SH',
            # 中金所
            'IF', 'IH', 'IC', 'IM', 'TS', 'TF', 'T', 'TL',
            # 上期能源 & 广期所
            'SC', 'NR', 'SI', 'LC', 'XS'
        ]
        
        patterns = []
        
        for code in products:
            # Standard format: AP2405, CU2312 (both upper and lower case)
            patterns.extend([
                {"label": "合约代码", "pattern": [{"TEXT": {"REGEX": f"^{code.upper()}\\d{{4}}$"}}]},
                {"label": "合约代码", "pattern": [{"TEXT": {"REGEX": f"^{code.lower()}\\d{{4}}$"}}]}
            ])
            
            # With separators: AP-2405, AP_2405, AP.2405, AP/2405
            for sep in ["-", "_", "\\.", "/"]:
                patterns.extend([
                    {"label": "合约代码", "pattern": [{"TEXT": {"REGEX": f"^{code.upper()}{sep}\\d{{4}}$"}}]},
                    {"label": "合约代码", "pattern": [{"TEXT": {"REGEX": f"^{code.lower()}{sep}\\d{{4}}$"}}]}
                ])
            
            # Space separated: AP 2405 (two tokens)
            patterns.extend([
                {"label": "合约代码", "pattern": [
                    {"TEXT": {"REGEX": f"^{code.upper()}$"}}, 
                    {"TEXT": {"REGEX": "^\\d{4}$"}}
                ]},
                {"label": "合约代码", "pattern": [
                    {"TEXT": {"REGEX": f"^{code.lower()}$"}}, 
                    {"TEXT": {"REGEX": "^\\d{4}$"}}
                ]}
            ])
        
        return patterns

    def sql_knowledge(self):
        sql_kl = SQLSchemaKnowledge(tokenizer, embedder, self.url, mxbai_reranker)
        return sql_kl

    def entity_recognition(self, query: str):
        """Enhanced entity recognition using spaCy + original NLP toolkit
        
        Args:
            query: 本轮问题
        Returns:
            query: 增加实体识别后的query
        """
        
        try:
            # Step 1: Use spaCy for enhanced entity recognition (NEW)
            doc = self.nlp(query)
            spacy_enhanced_query = query
            
            # Apply spaCy entities (including contract codes)
            entities_found = []
            for ent in doc.ents:
                entities_found.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            # Annotate query with spaCy entities (process from right to left)
            entities_found.sort(key=lambda x: x['start'], reverse=True)
            for entity in entities_found:
                start, end = entity['start'], entity['end']
                entity_text = entity['text']
                label = entity['label']
                
                # Normalize contract codes (remove separators, uppercase)
                if label == "合约代码":
                    normalized = re.sub(r'[-_\.\s/]+', '', entity_text.upper())
                    replacement = f"{normalized}({label})"
                else:
                    replacement = f"{entity_text}({label})"
                
                spacy_enhanced_query = spacy_enhanced_query[:start] + replacement + spacy_enhanced_query[end:]
            
            # Step 2: Apply original entity recognition (for compatibility)
            try:
                tokenizer_nlp = NLPToolkit(
                    user_dict_path=USER_DICT_PATH, 
                    syn_dict_path=SYNONYM_DICT_PATH, 
                    stop_words_path=STOP_WORDS_PATH
                )
                entity_list = tokenizer_nlp.recognize(spacy_enhanced_query)
                
                # Apply original entities that weren't caught by spaCy
                final_query = spacy_enhanced_query
                for entity in entity_list:
                    if entity['id'] != '' and entity['text'] in final_query:
                        # Only add if not already annotated
                        if f"({entity['label']})" not in final_query:
                            substring = entity['id'] + '(' + entity['label'] + ')'
                            final_query = final_query.replace(entity['text'], substring, 1)
            
            except Exception as original_error:
                logger.warning(f"Original entity recognition failed: {original_error}")
                final_query = spacy_enhanced_query
            
            logger.info(f"Entity recognition: {query} -> {final_query}")
            return final_query
        
        except Exception as e:
            logger.error(f"Enhanced entity recognition Error: {e}")
            traceback.print_exc()
            
            # Fallback to original method
            try:
                tokenizer_nlp = NLPToolkit(
                    user_dict_path=USER_DICT_PATH, 
                    syn_dict_path=SYNONYM_DICT_PATH, 
                    stop_words_path=STOP_WORDS_PATH
                )
                entity_list = tokenizer_nlp.recognize(query)
                for entity in entity_list:
                    if entity['id'] != '':
                        substring = entity['id'] + '(' + entity['label'] + ')'
                        query = query.replace(entity['text'], substring)
                return query
            except Exception as fallback_error:
                logger.error(f"Fallback entity recognition failed: {fallback_error}")
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

        # step2 query entity recognition (Enhanced with spaCy zh_core_web_md)
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
        single_table_prompt = self.generate_single_table_prompt(located_table)
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


# Test function for spaCy-enhanced entity recognition
def test_spacy_entity_recognition():
    """Test the spaCy-enhanced entity recognition"""
    
    print("🧪 Testing spaCy-Enhanced Entity Recognition")
    print("=" * 60)
    
    # Mock LLM for testing
    class MockLLM:
        pass
    
    try:
        # Initialize workflow with spaCy
        dataqa = DataQaWorkflow(MockLLM(), MockLLM(), MockLLM())
        
        test_cases = [
            "苹果期货AP2405的价格走势如何？",
            "请查询郑商所AP-2405合约的持仓数据",
            "华泰期货对CU_2312铜期货的分析报告",
            "大商所M.2405豆粕期货今日收盘价格",
            "上期所RB/2405螺纹钢期货走势分析",
            "中金所IF 2406沪深300期货基差变化",
            "比较AP2405和CU2312两个合约的表现",
            "分析ap2405苹果期货的多空格局",
            "华泰期货研究所认为，郑商所AP-2405和上期所CU_2312值得关注"
        ]
        
        for i, query in enumerate(test_cases, 1):
            print(f"\n{i}. 原始查询:")
            print(f"   {query}")
            
            try:
                enhanced = dataqa.entity_recognition(query)
                print(f"   增强查询:")
                print(f"   {enhanced}")
                
                # Check what was recognized
                contracts = enhanced.count("(合约代码)")
                entities = enhanced.count("(") - contracts
                print(f"   📊 识别结果: {contracts}个合约代码, {entities}个其他实体")
                
            except Exception as e:
                print(f"   ❌ 处理失败: {e}")
        
        print(f"\n✅ spaCy Enhanced Entity Recognition Test Completed!")
        print(f"💡 Key Features:")
        print(f"   - Uses zh_core_web_md model for better Chinese understanding")
        print(f"   - Recognizes contract codes in multiple formats")
        print(f"   - Integrates with existing entity recognition")
        print(f"   - Maintains backward compatibility")
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        print(f"💡 Make sure to install: python -m spacy download zh_core_web_md")


if __name__ == "__main__":
    test_spacy_entity_recognition()
