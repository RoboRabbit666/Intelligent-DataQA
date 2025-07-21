import re
import time
import json
import copy
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("🔧 实用微改进方案 - 真实性能指标测试")
print("=" * 80)

# %% [markdown]
# ## 1. 测试语料库

def create_focused_test_corpus():
    """创建专注于合约代码识别的测试语料库"""
    
    # 1. 核心合约代码查询
    contract_code_queries = [
        "郑商所AP2405期货怎么样了",
        "上期所CU2405铜期货价格如何",
        "大商所M2405豆粕期货走势分析", 
        "中金所IF2405期货今日表现",
        "郑商所SR2405白糖期货行情",
        "上期所RB2405螺纹钢期货数据",
        "大商所I2405铁矿石期货报价",
        "郑商所TA2405PTA期货分析",
        "上期所AL2405铝期货走势",
        "中金所IC2405中证500期货",
        
        # 变体表达
        "请问郑商所AP2405苹果期货怎么样",
        "今天上期所CU2405铜期货价格",
        "大商所M2405豆粕合约如何",
        "中金所IF2405沪深300期货行情",
        "郑商所SR2405白糖合约分析",
        
        # 更复杂的表达
        "郑商所AP2405苹果期货主力合约价格走势",
        "上期所CU2405铜期货今日收盘价格",
        "大商所M2405豆粕期货夜盘交易情况",
        "中金所IF2405期货基差变化分析",
        "郑商所SR2405白糖期货持仓量数据",
    ]
    
    # 2. 带公司信息的复杂查询
    company_contract_queries = [
        "华泰期货对郑商所AP2405期货的分析",
        "中信期货发布上期所CU2405铜期货报告", 
        "永安期货关于大商所M2405豆粕期货建议",
        "国泰君安期货研究中金所IF2405期货",
        "海通期货解读郑商所SR2405白糖期货",
        "方正中期期货看好上期所RB2405螺纹钢",
        "光大期货分析大商所I2405铁矿石期货",
        "银河期货策略郑商所TA2405PTA期货",
        "招商期货建议上期所AL2405铝期货",
        "广发期货解析中金所IC2405期货",
    ]
    
    # 3. 重复实体问题测试案例
    duplicate_issue_queries = [
        "苹果期货AP2405苹果期货价格",  # 重复
        "郑商所交易所AP2405期货",  # 冗余
        "AP2405期货AP苹果分析",  # 拆分
        "CU铜期货2405合约价格",  # 部分匹配
        "M豆粕2405期货走势",  # 简化
        "华泰期货华泰期货公司研究所分析AP2405",  # 嵌套实体
        "螺纹钢钢期货RB2405价格波动",  # 品种名称错误
        "原油油期货交割细则",  # 品种名称简化
        "股指期货IF2405指数基差分析",  # 术语混合
        "期货期货公司AP2405风险管理",  # 术语重复
    ]
    
    # 4. 误识别问题测试案例
    false_positive_queries = [
        "苹果公司股价分析报告",
        "华泰证券研究团队发布报告", 
        "铜价格走势分析图表",
        "白糖现货市场行情研究",
        "期货市场整体走势分析",
        "交易所监管政策解读",
        "大宗商品价格指数研究",
        "金融衍生品市场分析",
        "投资策略建议报告",
        "市场风险评估报告",
    ]
    
    # 5. 格式变体测试案例
    format_variant_queries = [
        "郑商所AP-2405期货合约分析",  # 连字符
        "上期所CU 2405期货交易",     # 空格
        "大商所M.2405豆粕期货价格",   # 点号
        "中金所IF/2405期货走势",     # 斜杠
        "郑商所SR_2405白糖期货",     # 下划线
        "上期所RB(2405)螺纹钢期货",  # 括号
        "大商所I期货2405铁矿石",     # 顺序变化
        "中金所期货IC2405走势",      # 位置变化
        "AP 24 05期货价格分析",      # 多空格
        "CU-24-05铜期货走势",       # 多连字符
    ]
    
    # 合并所有测试类型
    all_test_categories = {
        "核心合约代码查询": contract_code_queries,
        "公司与合约组合": company_contract_queries,
        "重复实体问题": duplicate_issue_queries,
        "误识别问题": false_positive_queries,
        "格式变体测试": format_variant_queries,
    }
    
    # 展平所有测试文本
    flat_tests = []
    for category, texts in all_test_categories.items():
        flat_tests.extend(texts)
    
    return flat_tests, all_test_categories

# 创建测试语料
test_texts, categorized_tests = create_focused_test_corpus()

print(f"📊 微改进专用测试语料库统计")
print("=" * 50)
print(f"总文本数量: {len(test_texts)}")
print(f"测试类别: {len(categorized_tests)}")
print("\n各类别文本数量:")
for category, texts in categorized_tests.items():
    print(f"  {category}: {len(texts)} 条")

# %% [markdown]
# ## 2. 原始RegEx方法实现

class OriginalRegexNERMethod:
    """原始RegEx实体识别方法"""
    
    def __init__(self):
        self.entity_patterns = {
            "EXCHANGE": [
                r"郑商所|郑州商品交易所|CZCE|czce",
                r"上期所|上海期货交易所|SHFE|shfe", 
                r"大商所|大连商品交易所|DCE|dce",
                r"中金所|中国金融期货交易所|CFFEX|cffex|CFE",
                r"上期能源|上海国际能源交易中心|INE|ine",
                r"广期所|广州期货交易所|GFEX|gfex"
            ],
            "FUTURES_COMPANY": [
                r"华泰期货(?:有限公司)?",
                r"中信期货(?:有限公司)?",
                r"永安期货(?:股份有限公司)?",
                r"国泰君安期货(?:有限公司)?",
                r"海通期货(?:有限公司)?",
                r"方正中期期货(?:有限公司)?",
                r"光大期货(?:有限公司)?",
                r"银河期货(?:有限公司)?",
                r"招商期货(?:有限公司)?",
                r"广发期货(?:有限公司)?"
            ],
            "PRODUCT": [
                r"苹果|AP|ap",
                r"棉花|CF|cf", 
                r"白糖|SR|sr",
                r"PTA|TA|ta",
                r"甲醇|MA|ma",
                r"玻璃|FG|fg",
                r"铜|CU|cu",
                r"铝|AL|al", 
                r"螺纹钢|RB|rb",
                r"热轧卷板|HC|hc",
                r"豆粕|M|(?<!A)m(?!a)",
                r"豆油|Y|(?<!C)y",
                r"玉米|C|(?<!D)c(?!u)",
                r"豆一|A|(?<!T)a(?!l)",
                r"铁矿石|I|(?<!F)i(?!f)",
                r"焦炭|J|(?<!D)j",
                r"焦煤|JM|jm",
                r"沪深300|IF|if",
                r"上证50|IH|ih", 
                r"中证500|IC|ic"
            ],
            "CONTRACT_CODE": [
                r"(?:AP|ap)(?:24|25)\d{2}",
                r"(?:CU|cu)(?:24|25)\d{2}",
                r"(?:M|(?<!A)m(?!a))(?:24|25)\d{2}",
                r"(?:SR|sr)(?:24|25)\d{2}",
                r"(?:TA|ta)(?:24|25)\d{2}",
                r"(?:RB|rb)(?:24|25)\d{2}",
                r"(?:I|(?<!F)i(?!f))(?:24|25)\d{2}",
                r"(?:IF|if)(?:24|25)\d{2}",
                r"(?:AL|al)(?:24|25)\d{2}",
                r"(?:IC|ic)(?:24|25)\d{2}",
            ],
            "PRICE_VALUE": [
                r"\d+(?:,\d{3})*\.?\d*元/吨",
                r"\d+(?:\.\d+)?%",
                r"\d+(?:,\d{3})*手",
                r"\d+(?:\.\d+)?万吨",
                r"\d+(?:,\d{3})*(?:\.\d+)?亿元"
            ],
            "TIME": [
                r"\d{4}年\d{1,2}月(?:\d{1,2}日)?",
                r"\d{4}年Q[1-4]",
                r"\d{1,2}:\d{2}-\d{1,2}:\d{2}",
                r"夜盘|收盘|开盘"
            ]
        }
        self.compiled_patterns = self._compile_patterns()
        
    def _compile_patterns(self):
        compiled = {}
        for entity_type, patterns in self.entity_patterns.items():
            compiled[entity_type] = []
            for pattern in patterns:
                try:
                    compiled[entity_type].append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    print(f"警告: 编译模式失败 {pattern}: {e}")
        return compiled
        
    def recognize_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        entities = []
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity_text = match.group()
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    is_duplicate = any(
                        start_pos == existing[2] and end_pos == existing[3]
                        for existing in entities
                    )
                    
                    if not is_duplicate:
                        entities.append((entity_text, entity_type, start_pos, end_pos))
        
        entities.sort(key=lambda x: x[2])
        return entities

# %% [markdown]
# ## 3. 微小改进RegEx方法实现

class MicroImprovedRegexNER:
    """微小改进RegEx方法 - 基于后处理的保守改进"""
    
    def __init__(self):
        # 使用与原始方法完全相同的基础模式
        self.original_method = OriginalRegexNERMethod()
        
        # 微小改进配置
        self.improvement_config = {
            'deduplicate_overlapping': True,
            'filter_false_positives': True,
            'enhance_contract_formats': True,
        }
    
    def _deduplicate_overlapping_entities(self, entities: List[Tuple]) -> List[Tuple]:
        """去除重叠实体"""
        if not entities:
            return entities
        
        # 按位置排序
        entities.sort(key=lambda x: x[2])
        
        deduplicated = []
        
        for current in entities:
            current_text, current_type, current_start, current_end = current
            
            should_add = True
            for i, existing in enumerate(deduplicated):
                existing_text, existing_type, existing_start, existing_end = existing
                
                # 检查重叠
                if not (current_end <= existing_start or current_start >= existing_end):
                    # 有重叠，保留更完整的实体
                    
                    # 规则1：如果一个完全包含另一个，保留更长的
                    if (current_start >= existing_start and current_end <= existing_end):
                        should_add = False
                        break
                    elif (existing_start >= current_start and existing_end <= current_end):
                        deduplicated.pop(i)
                        break
                    
                    # 规则2：优先保留合约代码
                    elif current_type == 'CONTRACT_CODE' and existing_type != 'CONTRACT_CODE':
                        deduplicated.pop(i)
                        break
                    elif existing_type == 'CONTRACT_CODE' and current_type != 'CONTRACT_CODE':
                        should_add = False
                        break
                    
                    # 规则3：保留更长的文本
                    elif len(current_text) > len(existing_text):
                        deduplicated.pop(i)
                        break
                    else:
                        should_add = False
                        break
            
            if should_add:
                deduplicated.append(current)
        
        return deduplicated
    
    def _filter_false_positives(self, entities: List[Tuple], text: str) -> List[Tuple]:
        """过滤误识别"""
        if not entities:
            return entities
        
        filtered = []
        
        # 定义明显不是期货相关的上下文
        non_futures_patterns = [
            r'苹果公司',
            r'苹果股价',
            r'苹果手机',
            r'华泰证券',
            r'铜价格(?!.*期货)',
            r'白糖价格(?!.*期货)',
        ]
        
        for entity in entities:
            entity_text, entity_type, start_pos, end_pos = entity
            
            should_keep = True
            
            # 主要对PRODUCT类型进行过滤
            if entity_type == 'PRODUCT':
                # 获取上下文
                context_window = 15
                context_start = max(0, start_pos - context_window)
                context_end = min(len(text), end_pos + context_window)
                context = text[context_start:context_end]
                
                # 检查是否匹配非期货模式
                for pattern in non_futures_patterns:
                    if re.search(pattern, context, re.IGNORECASE):
                        should_keep = False
                        break
            
            if should_keep:
                filtered.append(entity)
        
        return filtered
    
    def _enhance_contract_formats(self, entities: List[Tuple], text: str) -> List[Tuple]:
        """增强合约代码格式支持"""
        enhanced = copy.deepcopy(entities)
        
        # 定义合约代码变体模式
        variant_patterns = [
            r'\b([A-Z]{1,3})\s*-\s*(\d{4})\b',      # AP-2405
            r'\b([A-Z]{1,3})\s+(\d{4})\b',          # AP 2405
            r'\b([A-Z]{1,3})\.(\d{4})\b',           # AP.2405
            r'\b([A-Z]{1,3})\s*/\s*(\d{4})\b',      # AP/2405
            r'\b([A-Z]{1,3})\s*_\s*(\d{4})\b',      # AP_2405
        ]
        
        valid_codes = ['AP', 'CU', 'SR', 'TA', 'MA', 'RB', 'IF', 'AL', 'IC', 
                      'M', 'Y', 'C', 'A', 'I', 'J', 'JM', 'IH']
        
        for pattern in variant_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                code_part = match.group(1).upper()
                number_part = match.group(2)
                
                if code_part in valid_codes:
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # 检查是否已被覆盖
                    is_covered = any(
                        not (end_pos <= existing[2] or start_pos >= existing[3])
                        for existing in enhanced
                    )
                    
                    if not is_covered:
                        enhanced.append((
                            match.group(),
                            'CONTRACT_CODE',
                            start_pos,
                            end_pos
                        ))
        
        return enhanced
    
    def recognize_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """微小改进的实体识别"""

        # 第一步：使用原始方法识别
        entities = self.original_method.recognize_entities(text)

        # 第二步：应用微小改进
        if self.improvement_config.get('deduplicate_overlapping', False):
            entities = self._deduplicate_overlapping_entities(entities)
        
        if self.improvement_config.get('filter_false_positives', False):
            entities = self._filter_false_positives(entities, text)
        
        if self.improvement_config.get('enhance_contract_formats', False):
            entities = self._enhance_contract_formats(entities, text)
        
        # 最终排序
        entities.sort(key=lambda x: x[2])
        return entities

# %% [markdown]
# ## 4. 标准答案和性能计算

def create_ground_truth_mapping():
    """创建标准答案映射"""
    return {
        # 交易所标准答案
        "郑商所": "EXCHANGE", "上期所": "EXCHANGE", "大商所": "EXCHANGE", 
        "中金所": "EXCHANGE", "CZCE": "EXCHANGE", "SHFE": "EXCHANGE",
        "DCE": "EXCHANGE", "CFFEX": "EXCHANGE",
        
        # 期货公司标准答案
        "华泰期货": "FUTURES_COMPANY", "中信期货": "FUTURES_COMPANY",
        "永安期货": "FUTURES_COMPANY", "国泰君安期货": "FUTURES_COMPANY",
        "海通期货": "FUTURES_COMPANY", "方正中期期货": "FUTURES_COMPANY",
        "光大期货": "FUTURES_COMPANY", "银河期货": "FUTURES_COMPANY",
        "招商期货": "FUTURES_COMPANY", "广发期货": "FUTURES_COMPANY",
        
        # 合约代码标准答案
        "AP2405": "CONTRACT_CODE", "CU2405": "CONTRACT_CODE", 
        "M2405": "CONTRACT_CODE", "SR2405": "CONTRACT_CODE",
        "TA2405": "CONTRACT_CODE", "RB2405": "CONTRACT_CODE",
        "I2405": "CONTRACT_CODE", "IF2405": "CONTRACT_CODE",
        "AL2405": "CONTRACT_CODE", "IC2405": "CONTRACT_CODE",
        
        # 格式变体也算作合约代码
        "AP-2405": "CONTRACT_CODE", "CU 2405": "CONTRACT_CODE",
        "M.2405": "CONTRACT_CODE", "SR/2405": "CONTRACT_CODE",
        "TA_2405": "CONTRACT_CODE",
        
        # 品种名称
        "苹果": "PRODUCT", "铜": "PRODUCT", "豆粕": "PRODUCT",
        "白糖": "PRODUCT", "PTA": "PRODUCT", "螺纹钢": "PRODUCT",
        "铁矿石": "PRODUCT", "沪深300": "PRODUCT", "铝": "PRODUCT",
        "中证500": "PRODUCT",
        
        # 价格数值
        "8760元/吨": "PRICE_VALUE", "2.3%": "PRICE_VALUE", 
        "15847手": "PRICE_VALUE", "15%": "PRICE_VALUE",
        
        # 时间
        "2024年3月": "TIME", "21:00-23:00": "TIME", "夜盘": "TIME"
    }

def extract_ground_truth_entities(text: str, ground_truth_mapping: Dict) -> List[Tuple]:
    """从文本中提取标准答案实体"""
    entities = []
    for entity_text, entity_type in ground_truth_mapping.items():
        start_pos = 0
        while True:
            pos = text.find(entity_text, start_pos)
            if pos == -1:
                break
            entities.append((entity_text, entity_type, pos, pos + len(entity_text)))
            start_pos = pos + 1
    
    entities = list(set(entities))
    entities.sort(key=lambda x: x[2])
    return entities

def calculate_performance_metrics(predicted_entities: List[Tuple], 
                                true_entities: List[Tuple]) -> Dict:
    """计算性能指标"""
    pred_set = set([(ent[0], ent[1]) for ent in predicted_entities])
    true_set = set([(ent[0], ent[1]) for ent in true_entities])
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 按实体类型分析
    entity_type_metrics = {}
    all_types = set([ent[1] for ent in true_entities] + [ent[1] for ent in predicted_entities])
    
    for entity_type in all_types:
        true_type = set([(ent[0], ent[1]) for ent in true_entities if ent[1] == entity_type])
        pred_type = set([(ent[0], ent[1]) for ent in predicted_entities if ent[1] == entity_type])
        
        tp_type = len(pred_type & true_type)
        fp_type = len(pred_type - true_type)
        fn_type = len(true_type - pred_type)
        
        precision_type = tp_type / (tp_type + fp_type) if (tp_type + fp_type) > 0 else 0.0
        recall_type = tp_type / (tp_type + fn_type) if (tp_type + fn_type) > 0 else 0.0
        f1_type = 2 * precision_type * recall_type / (precision_type + recall_type) if (precision_type + recall_type) > 0 else 0.0
        
        entity_type_metrics[entity_type] = {
            'precision': precision_type,
            'recall': recall_type, 
            'f1': f1_type,
            'tp': tp_type,
            'fp': fp_type,
            'fn': fn_type,
            'support': len(true_type)
        }
    
    return {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_true': len(true_entities),
            'total_pred': len(predicted_entities)
        },
        'by_entity_type': entity_type_metrics
    }

# %% [markdown]
# ## 5. 执行真实性能测试

def run_micro_improvement_benchmark():
    """运行微改进基准测试"""
    
    methods = {
        "original_method": {
            "instance": OriginalRegexNERMethod(),
            "description": "原始RegEx方法"
        },
        "micro_improved": {
            "instance": MicroImprovedRegexNER(), 
            "description": "微改进RegEx方法"
        }
    }
    
    results = {}
    ground_truth_mapping = create_ground_truth_mapping()
    
    print(f"\n🚀 开始微改进基准测试")
    print("=" * 80)
    print(f"📊 测试文本数量: {len(test_texts)}")
    print(f"🎯 测试方法数量: {len(methods)}")
    print("=" * 80)
    
    for method_name, method_config in methods.items():
        print(f"\n📊 测试方法: {method_config['description']}")
        print("-" * 60)
        
        method_instance = method_config["instance"]
        
        print("⚡ 开始性能测试...")
        start_time = time.time()
        detailed_results = []
        processing_times = []
        
        for i, text in enumerate(test_texts):
            text_start = time.time()
            
            predicted_entities = method_instance.recognize_entities(text)
            
            text_time = time.time() - text_start
            processing_times.append(text_time)
            
            true_entities = extract_ground_truth_entities(text, ground_truth_mapping)
            
            metrics = calculate_performance_metrics(predicted_entities, true_entities)
            
            detailed_results.append({
                "text_id": i,
                "text": text,
                "entities": predicted_entities,
                "true_entities": true_entities,
                "processing_time": text_time,
                "metrics": metrics
            })
            
            if (i + 1) % 20 == 0:
                print(f"  已处理: {i + 1}/{len(test_texts)} 个文本")
        
        total_time = time.time() - start_time
        
        # 汇总指标
        all_predicted = []
        all_true = []
        for result in detailed_results:
            all_predicted.extend(result['entities'])
            all_true.extend(result['true_entities'])
        
        overall_metrics = calculate_performance_metrics(all_predicted, all_true)
        
        results[method_name] = {
            "description": method_config["description"],
            "total_time": total_time,
            "avg_time_per_text": total_time / len(test_texts),
            "texts_per_second": len(test_texts) / total_time,
            "overall_metrics": overall_metrics,
            "detailed_results": detailed_results
        }
        
        overall = results[method_name]['overall_metrics']['overall']
        
        print(f"✅ 测试完成!")
        print(f"⚡ 平均延迟: {results[method_name]['avg_time_per_text']*1000:.1f}毫秒")
        print(f"🎯 整体F1分数: {overall['f1']:.3f}")
        print(f"📊 精确率: {overall['precision']:.3f}")
        print(f"📋 召回率: {overall['recall']:.3f}")
    
    return results

# 执行测试
print("🔄 开始执行微改进性能测试...")
micro_improvement_results = run_micro_improvement_benchmark()

# %% [markdown]
# ## 6. 结果分析

def analyze_micro_improvement_results(results):
    """分析微改进结果"""
    
    print(f"\n📈 微改进性能分析")
    print("=" * 80)
    
    # 创建对比表
    comparison_data = []
    
    for method_name, result in results.items():
        overall_metrics = result['overall_metrics']['overall']
        
        comparison_data.append({
            "方法": method_name,
            "描述": result["description"], 
            "平均延迟(毫秒)": result['avg_time_per_text'] * 1000,
            "处理速度(文本/秒)": result['texts_per_second'],
            "精确率": overall_metrics['precision'],
            "召回率": overall_metrics['recall'],
            "F1分数": overall_metrics['f1'],
            "真正例": overall_metrics['tp'],
            "假正例": overall_metrics['fp'],
            "假负例": overall_metrics['fn'],
            "预测实体总数": overall_metrics['total_pred'],
            "真实实体总数": overall_metrics['total_true']
        })
    
    df = pd.DataFrame(comparison_data)
    
    print(f"📊 微改进方法性能对比表")
    print("-" * 80)
    print(df.round(3).to_string(index=False)) 
    
    # 计算改进效果
    original_metrics = results['original_method']['overall_metrics']['overall']
    improved_metrics = results['micro_improved']['overall_metrics']['overall']
    
    print(f"\n📈 微改进效果分析")
    print("-" * 50)
    
    f1_change = improved_metrics['f1'] - original_metrics['f1']
    precision_change = improved_metrics['precision'] - original_metrics['precision']
    recall_change = improved_metrics['recall'] - original_metrics['recall']
    
    print(f"F1分数变化: {original_metrics['f1']:.3f} → {improved_metrics['f1']:.3f} ({f1_change:+.3f})")
    print(f"精确率变化: {original_metrics['precision']:.3f} → {improved_metrics['precision']:.3f} ({precision_change:+.3f})")
    print(f"召回率变化: {original_metrics['recall']:.3f} → {improved_metrics['recall']:.3f} ({recall_change:+.3f})")
    
    # 分析具体改进类型
    print(f"\n🔍 具体改进分析:")
    
    # 按类别分析改进效果
    category_improvements = {}
    
    for category, texts in categorized_tests.items():
        original_scores = []
        improved_scores = []
        
        for text in texts:
            # 找到对应的结果
            original_result = None
            improved_result = None
            
            for result in results['original_method']['detailed_results']:
                if result['text'] == text:
                    original_result = result
                    break
            
            for result in results['micro_improved']['detailed_results']:
                if result['text'] == text:
                    improved_result = result
                    break
            
            if original_result and improved_result:
                original_scores.append(original_result['metrics']['overall']['f1'])
                improved_scores.append(improved_result['metrics']['overall']['f1'])
        
        if original_scores and improved_scores:
            avg_original = np.mean(original_scores)
            avg_improved = np.mean(improved_scores)
            improvement = avg_improved - avg_original
            
            category_improvements[category] = {
                'original_avg_f1': avg_original,
                'improved_avg_f1': avg_improved,
                'improvement': improvement
            }
    
    # 显示各类别改进情况
    for category, stats in category_improvements.items():
        improvement = stats['improvement']
        status = "✅" if improvement > 0 else "⚠️" if improvement < -0.05 else "➖"
        print(f"{status} {category}: F1 {stats['original_avg_f1']:.3f} → {stats['improved_avg_f1']:.3f} ({improvement:+.3f})")
    
    # 显示最佳改进案例
    print(f"\n🎯 最显著改进案例:")
    
    improvement_cases = []
    for i, text in enumerate(test_texts):
        original_result = results['original_method']['detailed_results'][i]
        improved_result = results['micro_improved']['detailed_results'][i]
        
        original_f1 = original_result['metrics']['overall']['f1']
        improved_f1 = improved_result['metrics']['overall']['f1']
        
        if improved_f1 > original_f1 + 0.1:  # 显著改进
            improvement_cases.append({
                'text': text,
                'original_f1': original_f1,
                'improved_f1': improved_f1,
                'improvement': improved_f1 - original_f1,
                'original_entities': original_result['entities'],
                'improved_entities': improved_result['entities']
            })
    
    # 按改进幅度排序
    improvement_cases.sort(key=lambda x: x['improvement'], reverse=True)
    
    for i, case in enumerate(improvement_cases[:3]):  # 显示前3个
        print(f"\n{i+1}. 文本: {case['text']}")
        print(f"   F1改进: {case['original_f1']:.3f} → {case['improved_f1']:.3f} (+{case['improvement']:.3f})")
        print(f"   原始识别: {len(case['original_entities'])} 个实体")
        print(f"   改进识别: {len(case['improved_entities'])} 个实体")
    
    return df, results

# 分析结果
df_micro_results, micro_details = analyze_micro_improvement_results(micro_improvement_results)

# 保存结果
def save_micro_improvement_results(results, df):
    """保存微改进测试结果"""
    
    print(f"\n💾 保存微改进测试结果")
    print("=" * 50)
    
    clean_results = {}
    for method_name, result in results.items():
        clean_result = {
            "description": result["description"],
            "total_time": result["total_time"],
            "avg_time_per_text": result["avg_time_per_text"],
            "texts_per_second": result["texts_per_second"],
            "overall_metrics": result["overall_metrics"]
        }
        clean_results[method_name] = clean_result
    
    with open("micro_improvement_results.json", "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    if not df.empty:
        df.to_csv("micro_improvement_comparison.csv", index=False, encoding="utf-8")
    
    print(f"✅ 结果已保存到: micro_improvement_results.json")
    print(f"✅ 对比表已保存到: micro_improvement_comparison.csv")

# 保存结果
save_micro_improvement_results(micro_improvement_results, df_micro_results)

# 总结
print(f"\n🎉 **微改进测试完成!**")
print("=" * 80)

if micro_details:
    original_metrics = micro_details['original_method']['overall_metrics']['overall']
    improved_metrics = micro_details['micro_improved']['overall_metrics']['overall']
    
    print(f"🏆 **测试结论**:")
    print(f"📊 在{len(test_texts)}个测试文本上对比了原始方法和微改进方法")
    
    print(f"\n📈 **核心性能指标**:")
    print(f"🎯 F1分数: {original_metrics['f1']:.3f} → {improved_metrics['f1']:.3f}")
    print(f"📊 精确率: {original_metrics['precision']:.3f} → {improved_metrics['precision']:.3f}")  
    print(f"📋 召回率: {original_metrics['recall']:.3f} → {improved_metrics['recall']:.3f}")
    
    f1_change = improved_metrics['f1'] - original_metrics['f1']
    precision_change = improved_metrics['precision'] - original_metrics['precision']
    recall_change = improved_metrics['recall'] - original_metrics['recall']
    
    print(f"\n💡 **改进评估**:")
    if f1_change > 0:
        print(f"✅ 整体性能提升: F1分数提升 {f1_change:.3f}")
    else:
        print(f"⚠️ 整体性能变化: F1分数变化 {f1_change:.3f}")
        
    if precision_change > 0:
        print(f"✅ 精确率提升: +{precision_change:.3f} (减少误识别)")
    else:
        print(f"⚠️ 精确率变化: {precision_change:.3f}")
        
    if recall_change >= -0.05:  # 小幅下降可接受
        print(f"✅ 召回率保持稳定: {recall_change:.3f}")
    else:
        print(f"⚠️ 召回率下降: {recall_change:.3f}")