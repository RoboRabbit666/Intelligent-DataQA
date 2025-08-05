from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

from typing import List, Optional, Dict, Tuple
import re as regex_module
import copy
import traceback
import numpy as np
import pickle
import os
from datetime import datetime

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
        collection: Optional[str] = "hybrid_sql",
        use_cache: bool = True  # 新增参数：是否使用缓存
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
        
        # FAQ相关属性
        self.faq_data = []  # 存储FAQ的问题、SQL和嵌入向量
        self.use_cache = use_cache # 是否使用缓存

        # 缓存文件路径
        self.cache_file = Path(__file__).parent.parent / "test_data" / "tables" / "faq_cache.pkl"

        # 初始化时加载FAQ (优先从缓存加载，如果缓存不存在则从文件加载)
        self._load_faqs()

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
            # 计算余弦相似度公式：cos(θ) = (A·B) / (|A|×|B|)
            # A·B：两个向量的点积
            # |A|、|B|：两个向量的模长（范数）
            similarity = np.dot(query_embedding, faq['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(faq['embedding'])
            )
            similarities.append(similarity)
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

    def sql_knowledge(self):
        sql_kl = SQLSchemaKnowledge(tokenizer, embedder, self.url, mxbai_reranker)
        return sql_kl

    def entity_recognition(self, query: str):
        """Enhanced entity recognition using REGEX patterns"""
        try:
            enhanced_query = query
            
            tokenizer = NLPToolkit(
                user_dict_path=USER_DICT_PATH, 
                syn_dict_path=SYNONYM_DICT_PATH,
                stop_words_path=STOP_WORDS_PATH,
                patterns_path=NER_PATTERNs_PATH
            )
            entity_list = tokenizer.recognize(query)
            
            for entity in entity_list:
                if entity['id'] != '' and entity['text'] in enhanced_query:
                    if entity['label'] == '合约':
                        normalized_code = regex_module.sub(r'[-_\.\s/]+', '', entity['text'].upper())
                        substring = f"{normalized_code}(合约)"
                    else:
                        substring = f"{entity['id']}({entity['label']})"
                    
                    enhanced_query = enhanced_query.replace(entity['text'], substring, 1)
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"实体识别失败: {e}")
            return query

    def modify_query(self, input_messages: List[ChatMessage]) -> OptimizedQuery:
        """问题改写"""
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

    def locate_table(self, query: str) -> List[ChatReference]:
        """根据查询内容定位到相关的表格"""
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
                lambda x: {'chunk_uuid':x.chunk_id,'table_name':x.data.table_name, 'score':x.reranking_score},
                ranked_tables,
            )
        )
        return tables

    def generate_single_table_prompt(self, chunk_id:str):
        """生成单表查询的prompt"""
        sql_kl = self.sql_knowledge()
        table_content = sql_kl.get_by_ids(self.collection,chunk_id)
        table_info = table_content[0].data.table_info
        table_prompt = f"已知如下数据表信息: \n{table_info}\n"
        return table_prompt

    def extract_info(self, text:str, pattern:str):
        """使用正则表达式提取信息"""
        extract_pattern = regex_module.compile(pattern,regex_module.DOTALL)
        match = extract_pattern.search(text)
        if match:
            return match.group(1)
        else:
            return None

    def generate_sql_code(self, table_schema:str, input_messages: List[ChatMessage], thinking: Optional[bool] = False):
        """生成SQL代码"""
        query = input_messages[-1].content
        content=dataqa_prompt.format(table_schema=table_schema,question=query)
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
        # 保留最后中间的对话
        if len(input_messages[1:-1]) > self.history_round * 2:
            del input_messages[1 : -1 - self.history_round * 2]

        # Step 1: 改写问题
        optimized_input_messages = self.modify_query(input_messages)
        step1 = ChatStep(
            key="modify_query",
            name="改写问题",
            number=1,
            prompt=optimized_input_messages.rewritten_query,
            finished=True,
        )

        # Step 2: 实体识别
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

        # Step 3: 语义搜索FAQ
        faq_results = self.semantic_search_faq(entitled_query, top_k=3)
        # 调用语义搜索函数，在FAQ知识库中寻找与用户查询最相似的3个问题
        # entitled_query: 经过实体识别增强的查询，如"查询FG2509(合约)成交量"
        # 返回格式：[{'question': '...', 'sql': '...', 'table': '...', 'similarity': 0.92}, ...]

                
        # 如果找到高相似度的FAQ，直接使用（快速响应路径）
        if faq_results and faq_results[0]['similarity'] >= 0.85:
            # 条件检查：
            # 1. faq_results不为空（找到了FAQ）
            # 2. 第一个结果的相似度 >= 0.85（非常高的相似度，可以直接使用）
            step3 = ChatStep(
                key="semantic_search_faq",
                name="语义搜索(直接命中)", # 标记为直接命中
                number=3,
                prompt=f"找到高相似度FAQ，相似度：{faq_results[0]['similarity']:.3f}",
                finished=True,
            )
            
            # 构造FAQ响应
            best_faq = faq_results[0] # 取相似度最高的FAQ
            response_content = f"""基于知识库最佳匹配（相似度：{best_faq['similarity']:.3f}）：

```sql
{best_faq['sql']}
```

来源：{best_faq['table']}知识库
匹配问题：{best_faq['question']}"""
            
            # 响应格式：展示相似度、SQL代码块、来源表、匹配的原问题
            # 创建模拟的LLM响应对象（因为没有调用真正的LLM）
            # 使用Python的type()动态创建类来模拟响应结构
            # 这里模拟了一个响应对象，实际使用中应替换为真实的LLM响应
            # 注意：这里的response_content是模拟的内容，实际使用中应从LLM响应中获取
            # 例如：response = self.ans_client.invoke(messages=[system_msg] + input_messages[:])
            # 这里的response_content是直接构造的字符串，实际使用中应从LLM响应中获取内容
            # 创建简单的响应对象
            response = type('Response', (), {  # 创建Response类
                'id': 'faq_response',
                'model': 'semantic_search',    # 标记为语义搜索模型
                'created': 0,
                'usage': type('Usage', (), {    # 创建Usage类
                    # 模拟使用情况，实际使用中应从LLM响应中获取
                    # 这里的值都是0，因为这是模拟的响应，没有实际调用LLM
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                })(),
                'choices': [type('Choice', (), {
                    'finish_reason': 'stop',
                    'index': 0,
                    'message': type('Message', (), {
                        'role': 'assistant',
                        'content': response_content,
                        'reasoning_content': None
                    })()
                })()]
            })()
            # 模拟的响应对象，实际使用中应替换为真实的LLM响应，这个复杂的对象结构是为了模拟真实的LLM响应格式
            
        else:
            # FAQ相似度不够高（< 0.85），继续原流程（表格定位+LLM生成SQL）
            if faq_results:
                # 找到了FAQ但相似度不够高（< 0.85）
                step3 = ChatStep(
                    key="semantic_search_faq",
                    name="语义搜索",
                    number=3,
                    prompt=f"找到{len(faq_results)}个FAQ，最高相似度：{faq_results[0]['similarity']:.3f}",
                    finished=True,
                )
            else:
                # 没有找到任何FAQ
                step3 = ChatStep(
                    key="semantic_search_faq",
                    name="语义搜索",
                    number=3,
                    prompt="未找到相关FAQ",
                    finished=True,
                )
            
            # Step 4: 定位表格（继续原有的向量搜索流程）
            located_table = self.locate_table(optimized_input_messages.rewritten_query)
            step4 = ChatStep(
                key="locate_table",
                name="定位表格",
                number=4,
                prompt=located_table,
                finished=True,
            )

            # Step 5: 生成单表提示词
            single_table_prompt = self.generate_single_table_prompt(located_table[0]['chunk_uuid'])
            
            # 如果有中等相似度的FAQ，作为参考添加到prompt中
            if faq_results and faq_results[0]['similarity'] >= 0.7:
                # 相似度在0.7-0.85之间，作为参考而不是直接答案
                single_table_prompt += f"\n\n参考FAQ：\n问题：{faq_results[0]['question']}\nSQL：{faq_results[0]['sql']}\n"
            
            step5 = ChatStep(
                key="generate_single_table_prompt",
                name="生成单表提示词",
                number=5,
                prompt=single_table_prompt,
                finished=True,
            )

            # Step 6: 生成SQL（调用LLM）
            response = self.generate_sql_code(single_table_prompt, optimized_input_messages, thinking)
            step6 = ChatStep(
                key="generate_sql",
                name="生成SQL",
                number=6,
                prompt=response,
                finished=True,
            )

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
            steps=[step1, step2, step3, step4, step5, step6],
        )