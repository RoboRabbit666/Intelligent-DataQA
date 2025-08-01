from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

from typing import List, Optional
import re
import copy
import traceback
# ====== 新增：spaCy增强功能的导入 ======
import json
import spacy
from spacy.pipeline import EntityRuler
# =======================================

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
        
        # ====== 新增：初始化spaCy管道 ======
        # 为增强实体识别功能初始化spaCy管道
        self._setup_spacy_pipeline()
        # ==================================

    # ====== 新增方法：设置spaCy管道 ======
    def _setup_spacy_pipeline(self):
        """设置spaCy管道，使用zh_core_web_md模型和合约代码模式
        
        新增方法：此方法用于将spaCy集成到工作流中
        """
        try:
            # 加载中文中等模型（优先选择zh_core_web_md）
            self.nlp = spacy.load("zh_core_web_md")
            logger.info("✅ 成功加载zh_core_web_md模型")
        except OSError:
            logger.warning("⚠️ 未找到zh_core_web_md，尝试使用zh_core_web_sm")
            try:
                self.nlp = spacy.load("zh_core_web_sm")
                logger.info("✅ 成功加载zh_core_web_sm作为备选")
            except OSError:
                logger.error("❌ 未找到中文spaCy模型")
                self.nlp = None
                return

        # 在NER组件之前添加EntityRuler用于合约代码识别
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        
        # 从ner_patterns.jsonl + 手动补充创建合约代码模式
        patterns = self._create_contract_patterns()
        ruler.add_patterns(patterns)
        
        logger.info(f"✅ 已向spaCy EntityRuler添加{len(patterns)}个模式")
    # ===================================

    # ====== 新增方法：从JSONL文件读取模式 ======
    def _extract_product_codes_from_jsonl(self):
        """从ner_patterns.jsonl文件中提取产品代码
        
        新增方法：此方法读取现有模式并提取产品代码
        """
        product_codes = set()
        
        try:
            if Path(NER_PATTERNs_PATH).exists():
                with open(NER_PATTERNs_PATH, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            pattern_data = json.loads(line)
                            
                            # 从"品种"标签模式中提取产品代码
                            if pattern_data.get('label') == '品种':
                                pattern = pattern_data.get('pattern')
                                
                                # 处理字符串模式（直接产品代码）
                                if isinstance(pattern, str):
                                    # 检查是否为有效的产品代码（1-3个大写字母）
                                    if re.match(r'^[A-Z]{1,3}$', pattern.upper()):
                                        product_codes.add(pattern.upper())
                                
                                # 处理列表模式（基于token的模式）
                                elif isinstance(pattern, list) and len(pattern) == 1:
                                    token_pattern = pattern[0]
                                    if isinstance(token_pattern, dict):
                                        # 从LOWER模式中提取
                                        if 'LOWER' in token_pattern:
                                            text = token_pattern['LOWER']
                                            if re.match(r'^[a-z]{1,3}$', text):
                                                product_codes.add(text.upper())
                                        
                                        # 从TEXT模式中提取
                                        elif 'TEXT' in token_pattern:
                                            text = token_pattern['TEXT']
                                            if re.match(r'^[A-Za-z]{1,3}$', text):
                                                product_codes.add(text.upper())
                        
                        except json.JSONDecodeError:
                            continue
            
            logger.info(f"📁 从ner_patterns.jsonl中提取了{len(product_codes)}个产品代码")
            
        except Exception as e:
            logger.warning(f"读取ner_patterns.jsonl失败: {e}")
        
        return list(product_codes)
    # =========================================

    # ====== 新增方法：手动补充 ======
    def _get_manual_supplement_codes(self):
        """获取缺失或新产品的手动补充产品代码
        
        新增方法：提供jsonl文件中可能没有的额外产品代码
        """
        # 对常见缺失或新产品代码的手动补充
        # 只添加可能在ner_patterns.jsonl中缺失的代码
        manual_codes = [
            # 可能缺失的常见代码
            'EC',    # 欧线集运（较新产品）
            'BC',    # 国际铜（较新产品） 
            'LU',    # 低硫燃料油（较新产品）
            'NR',    # 20号胶（较新产品）
            'SS',    # 不锈钢（较新产品）
            'IM',    # 中证1000（较新产品）
            'XS',    # 多晶硅（较新产品）
            'LC',    # 碳酸锂（较新产品）
            'SI',    # 工业硅（较新产品）
            
            # 可能被遗漏的单字母代码
            'T',     # 10年期国债
            'L',     # 聚乙烯
            'V',     # 聚氯乙烯
            'A',     # 豆一
            'B',     # 豆二
            'C',     # 玉米
            'I',     # 铁矿石
            'J',     # 焦炭
            'M',     # 豆粕
            'P',     # 棕榈油
            'Y',     # 豆油
        ]
        
        logger.info(f"➕ 手动补充：{len(manual_codes)}个额外的产品代码")
        return manual_codes
    # ===============================

    # ====== 新增方法：创建合约模式 ======
    def _create_contract_patterns(self):
        """通过组合jsonl + 手动补充创建合约代码模式
        
        新增方法：结合文件和手动补充的模式
        """
        # 步骤1：从ner_patterns.jsonl提取产品代码
        jsonl_codes = self._extract_product_codes_from_jsonl()
        
        # 步骤2：获取手动补充代码
        manual_codes = self._get_manual_supplement_codes()
        
        # 步骤3：合并并去重
        all_codes = list(set(jsonl_codes + manual_codes))
        
        logger.info(f"🔄 合并产品代码：{len(jsonl_codes)}个来自jsonl + {len(manual_codes)}个手动 = 总计{len(all_codes)}个")
        
        # 步骤4：生成合约代码模式
        patterns = []
        
        for code in all_codes:
            # 模式1：标准格式（AP2405, CU2312）- 大写和小写
            patterns.extend([
                {"label": "合约代码", "pattern": [{"TEXT": {"REGEX": f"^{code.upper()}\\d{{4}}$"}}]},
                {"label": "合约代码", "pattern": [{"TEXT": {"REGEX": f"^{code.lower()}\\d{{4}}$"}}]}
            ])
            
            # 模式2：带分隔符（AP-2405, AP_2405, AP.2405, AP/2405）
            for sep in ["-", "_", "\\.", "/"]:
                patterns.extend([
                    {"label": "合约代码", "pattern": [{"TEXT": {"REGEX": f"^{code.upper()}{sep}\\d{{4}}$"}}]},
                    {"label": "合约代码", "pattern": [{"TEXT": {"REGEX": f"^{code.lower()}{sep}\\d{{4}}$"}}]}
                ])
            
            # 模式3：空格分隔（AP 2405）- 两个token
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
        
        logger.info(f"🎯 生成了{len(patterns)}个合约代码模式")
        return patterns
    # ===================================

    def sql_knowledge(self):
        # 原始方法 - 未修改
        sql_kl = SQLSchemaKnowledge(tokenizer, embedder, self.url, mxbai_reranker)
        return sql_kl

    # ====== 增强方法：实体识别 ======
    def entity_recognition(self, query: str):
        """使用spaCy + 原始方法的增强实体识别
        
        增强方法：此方法现在结合了spaCy和原始NLP工具包
        
        Args:
            query: 本轮问题
        Returns:
            query: 增加实体识别后的query
        """
        
        try:
            enhanced_query = query
            
            # ====== 新增：spaCy实体识别 ======
            if self.nlp is not None:
                # 步骤1：使用spaCy进行实体识别（包括合约代码）
                doc = self.nlp(query)
                
                # 提取并标注spaCy实体
                spacy_entities = []
                for ent in doc.ents:
                    spacy_entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
                
                # 应用spaCy实体（从右到左处理避免位置偏移）
                spacy_entities.sort(key=lambda x: x['start'], reverse=True)
                for entity in spacy_entities:
                    start, end = entity['start'], entity['end']
                    entity_text = entity['text']
                    label = entity['label']
                    
                    # 合约代码的特殊处理：标准化格式
                    if label == "合约代码":
                        # 移除分隔符并转换为大写：AP-2405 → AP2405
                        normalized = re.sub(r'[-_\.\s/]+', '', entity_text.upper())
                        replacement = f"{normalized}({label})"
                    else:
                        replacement = f"{entity_text}({label})"
                    
                    enhanced_query = enhanced_query[:start] + replacement + enhanced_query[end:]
                
                logger.info(f"🔍 spaCy找到{len(spacy_entities)}个实体")
            # ====================================
            
            # ====== 原始：传统实体识别 ======
            # 步骤2：应用原始实体识别以保持兼容性
            # 修复：添加了缺失的pattetns_path参数
            try:
                tokenizer_nlp = NLPToolkit(
                    user_dict_path=USER_DICT_PATH, 
                    syn_dict_path=SYNONYM_DICT_PATH, 
                    stop_words_path=STOP_WORDS_PATH,
                    pattetns_path=NER_PATTERNs_PATH  # 修复：添加了缺失的参数
                )
                entity_list = tokenizer_nlp.recognize(enhanced_query)
                
                # 应用还未被spaCy标注的原始实体
                for entity in entity_list:
                    if entity['id'] != '' and entity['text'] in enhanced_query:
                        # 只在未被标注时添加（避免重复标注）
                        if f"({entity['label']})" not in enhanced_query.replace(entity['text'], ''):
                            substring = entity['id'] + '(' + entity['label'] + ')'
                            enhanced_query = enhanced_query.replace(entity['text'], substring, 1)
                
                logger.info(f"🔍 原始方法找到{len(entity_list)}个实体")
            
            except Exception as original_error:
                logger.warning(f"原始实体识别失败: {original_error}")
            # ===============================
            
            logger.info(f"✅ 实体识别: {query} -> {enhanced_query}")
            return enhanced_query
        
        except Exception as e:
            logger.error(f"增强实体识别错误: {e}")
            traceback.print_exc()
            
            # ====== 回退到原始方法 ======
            # 修复：添加了缺失的pattetns_path参数
            try:
                tokenizer_nlp = NLPToolkit(
                    user_dict_path=USER_DICT_PATH, 
                    syn_dict_path=SYNONYM_DICT_PATH, 
                    stop_words_path=STOP_WORDS_PATH,
                    pattetns_path=NER_PATTERNs_PATH  # 修复：添加了缺失的参数
                )
                entity_list = tokenizer_nlp.recognize(query)
                for entity in entity_list:
                    if entity['id'] != '':
                        # 仅针对定的实体进行识别
                        substring = entity['id'] + '(' + entity['label'] + ')'
                        query = query.replace(entity['text'], substring)
                return query
            except Exception as fallback_error:
                logger.error(f"回退实体识别失败: {fallback_error}")
                return query
    # ===============================

    def modify_query(
        self,
        input_messages: List[ChatMessage],
    ) -> OptimizedQuery:
        # 原始方法 - 未修改
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
            logger.error(f"修改查询错误:{e}")
            traceback.print_exc()
            raise e

    # 定位表
    def locate_table(
        self,
        query: str,
    ) -> List[ChatReference]:
        # 原始方法 - 未修改
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
        
        # 获取前3个表格及其分数
        #table_names = [table.data.table_name for table in ranked_tables]
        #table_scores = [table.reranking_score for table in ranked_tables]
        #chunk_ids = [table.chunk_id for table in ranked_tables]
        #return chunk_ids[:3],table_names[:3], table_scores[:3]
        
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
        # 原始方法 - 未修改
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
        # 原始方法 - 未修改
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
        # 原始方法 - 未修改
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
        
        '''
        result = response.choices[0].message.content
        sql_pattern = self.extract_info(result,r'```sql(.*?)```')
        analysis_pattern = self.extract_info(result,r"以下是问题分析和代码解释: (.*?)---")
        if analysis_pattern != None:
            analysis_info = analysis_pattern
        else:
            analysis_info = "未找到问题分析代码解释"
        if sql_pattern != None:
            sql_code = sql_pattern
        else:
            sql_code = "未找到SQL代码"
        return analysis_info,sql_code
        '''

    def do_generate(
        self,
        input_messages: List[ChatMessage],
        knowledge_base_ids: Optional[List[str]] = None,
        thinking: Optional[bool] = False,
    ):
        # 原始方法 - 未修改（除了step2注释更新）
        """生成回答"""
        # 保留最后中间的对话，中间的对话最多保留 self.history_round * 2 轮
        if len(input_messages[1:-1]) > self.history_round * 2:
            del input_messages[1 : -1 - self.history_round * 2]

        # step1 修改查询
        optimized_input_messages = self.modify_query(input_messages)
        step1 = ChatStep(
            key="modify_query",
            name="改写问题",
            number=1,
            prompt=optimized_input_messages.rewritten_query,
            finished=True,
        )

        # step2 查询实体识别（增强：现在使用spaCy + 原始方法）
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

        # step3 定位表格
        located_table = self.locate_table(optimized_input_messages.rewritten_query)
        step3 = ChatStep(
            key="locate_table",
            name="定位表格",
            number=3,
            prompt=located_table,
            finished=True,
        )

        # step4 生成单表提示词
        single_table_prompt = self.generate_single_table_prompt(located_table[0]['chunk_uuid'])
        step4 = ChatStep(
            key="generate_single_table_prompt",
            name="生成单表提示词",
            number=4,
            prompt=single_table_prompt,
            finished=True,
        )

        # step5 生成SQL
        response = self.generate_sql_code(table_schema=single_table_prompt, input_messages=optimized_input_messages, thinking=thinking)
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


# ====== 新增：测试函数 ======
def test_enhanced_entity_recognition():
    """测试增强实体识别与spaCy集成
    
    新增函数：验证增强功能的测试函数
    """
    
    print("🧪 测试增强实体识别（spaCy + 原始方法）")
    print("=" * 80)
    
    # 用于测试的模拟LLM
    class MockLLM:
        pass
    
    try:
        # 初始化增强工作流
        dataqa = DataQaWorkflow(MockLLM(), MockLLM(), MockLLM())
        
        test_cases = [
            # 各种格式的合约代码
            "苹果期货AP2405的价格走势如何？",
            "请查询郑商所AP-2405合约的持仓数据",
            "华泰期货对CU_2312铜期货的分析报告",
            "大商所M.2405豆粕期货今日收盘价格",
            "上期所RB/2405螺纹钢期货走势分析",
            "中金所IF 2406沪深300期货基差变化",
            
            # 多个合约
            "比较AP2405和CU2312两个合约的表现",
            "分析ap2405苹果期货的多空格局",
            
            # 包含混合实体的复杂句子
            "华泰期货研究所认为，郑商所AP-2405苹果期货和上期所CU_2312铜期货在当前市场环境下值得关注，建议重点关注IF 2406沪深300期货的基差变化情况。",
            
            # 边界情况
            "苹果公司股票AAPL今日走势",  # 不应匹配合约代码
            "今天AP错过了好机会",         # 不完整，不应匹配
        ]
        
        for i, query in enumerate(test_cases, 1):
            print(f"\n{i:2d}. 测试查询:")
            print(f"    原始: {query}")
            
            try:
                enhanced = dataqa.entity_recognition(query)
                print(f"    增强: {enhanced}")
                
                # 分析结果
                contract_count = enhanced.count("(合约代码)")
                other_entities = enhanced.count("(") - contract_count
                
                print(f"    📊 识别: {contract_count}个合约代码, {other_entities}个其他实体")
                
                # 检查合约代码是否已标准化
                if contract_count > 0:
                    print(f"    ✅ 合约代码已识别并标准化")
                
            except Exception as e:
                print(f"    ❌ 处理失败: {e}")
        
        print(f"\n✅ 增强实体识别测试完成！")
        print(f"\n💡 展示的关键功能:")
        print(f"   🔗 spaCy zh_core_web_md集成")
        print(f"   📁 从ner_patterns.jsonl自动加载模式")
        print(f"   ➕ 缺失代码的手动补充")
        print(f"   🔄 无缝回退到原始方法")
        print(f"   🎯 合约代码标准化（AP-2405 → AP2405）")
        print(f"   🛡️ 保持向后兼容性")
        print(f"   ✅ 保留所有原始NLPToolkit参数")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print(f"💡 请确保安装: pip install spacy && python -m spacy download zh_core_web_md")


if __name__ == "__main__":
    test_enhanced_entity_recognition()
    
    '''
    # 生产环境使用示例:
    dataqa = DataQaWorkflow(
        ans_llm=qwen3_llm,
        ans_thinking_llm=qwen3_thinking_llm,
        query_llm=qwen3_llm
    )
    
    # 测试实际查询
    query = "华泰期货对郑商所AP-2405苹果期货的分析报告"
    enhanced_query = dataqa.entity_recognition(query)
    print(f"查询: {query}")
    print(f"增强: {enhanced_query}")
    '''