# %% [markdown]
# ## NER模型性能基准测试 - 增强版：专业语料和高级指标分析
# 
# ## 📋 新增功能
# # 1. 专业挑战性测试语料（期货合约代码等）
# # 2. 精确率、召回率、F1分数等详细指标
# # 3. 失败模式量化分析
# # 4. 实体级别和类型级别的详细性能分析

# %%
import spacy
import time
import json
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, Markdown
import warnings
import re
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support, classification_report
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

print("🚀 增强版 NER模型性能基准测试系统")
print("=" * 80)
print("✅ 环境设置完成")
print(f"📦 spaCy版本: {spacy.__version__}")
print(f"📝 Pandas版本: {pd.__version__}")
print(f"📊 NumPy版本: {np.__version__}")

# %% [markdown]
# ## 2. 增强测试语料库创建 - 添加专业挑战性测试

# %%
def create_enhanced_professional_test_corpus():
    """创建增强的专业期货交易领域测试语料库，包含更具挑战性的案例"""
    
    # 原有基础测试保持不变
    basic_entity_tests = [
        "苹果期货在郑州商品交易所交易，华泰期货公司参与其中",
        "上海期货交易所的铜期货价格上涨，中信期货发布研究报告", 
        "大连商品交易所推出新的豆粕期货合约，永安期货积极参与交易",
        "郑商所白糖期货主力合约收盘价格为5200元/吨",
        "中国金融期货交易所股指期货IF2024合约波动加剧",
        "永安期货、海通期货、申银万国期货三家公司持仓排名前三",
        "国泰君安期货研究所发布螺纹钢期货投资策略报告",
        "方正中期期货在棉花期货交易中表现活跃",
        "光大期货与银河期货在原油期货市场份额领先",
        "招商期货、广发期货、东证期货联合发布市场分析"
    ]
    
    # ⭐ 新增：专业挑战性测试语料（包含合约代码和复杂金融术语）
    professional_challenging_tests = [
        # 合约代码识别挑战
        "郑商所AP2502期货怎么样了",
        "上期所CU2503合约今日涨停，华泰期货建议关注",
        "大商所M2505豆粕期货主力合约成交活跃",
        "中金所IF2503沪深300股指期货基差收窄",
        "INE原油SC2504合约与布伦特原油价差扩大",
        
        # 复杂机构名称和产品组合
        "方正中期期货风险管理子公司在PTA2504合约上建立空头套保头寸",
        "国投安信期货资产管理部门看好RB2503螺纹钢期货后市表现",
        "华泰期货研究所分析师认为ZC2505动力煤期货价格将震荡上行",
        "永安资本管理有限公司在CF2504棉花期货上增持多头头寸",
        "中信期货衍生品事业部推出基于SR2505白糖期货的结构化产品",
        
        # 技术分析术语 + 合约代码
        "TA2504PTA期货突破前期高点，MACD指标显示多头信号",
        "AG2503白银期货形成head and shoulders顶部形态，建议减仓",
        "NI2504镍期货price action呈现三角形整理，等待方向选择",
        "FG2505玻璃期货implied volatility飙升，期权skew加剧",
        "I2504铁矿石期货basis point value计算中需考虑duration risk",
        
        # 时间敏感和数值精确性测试
        "2024年12月17日，RB2505螺纹钢期货收盘价4120元/吨，较前日上涨2.3%",
        "截至2024年12月收盘，华泰期货在AP2502苹果期货上持仓量达到15000手",
        "郑商所CF2505棉花期货最后交易日为2025年5月15日，交割月临近",
        "上期所AU2506黄金期货夜盘21:00-02:30成交量放大至8.7万手",
        "中金所IC2503中证500期货保证金比例调整为12%，涨跌停板幅度6%",
        
        # 混合语言和缩写挑战
        "shfe RB2503螺纹钢期货volume weighted average price突破关键阻力位",
        "czce TA2504 PTA期货open interest创历史新高，market depth显著改善",
        "dce M2505豆粕期货与CBOT大豆期货价差（basis）持续扩大",
        "CFFEX IF2503指数期货roll yield为负，contango结构明显",
        "INE SC2504原油期货与WTI crude oil联动性增强，correlation coefficient达0.85",
        
        # 复杂财务和风险管理术语
        "华泰期货风险子公司通过CF2505棉花期货进行basis trading，hedge ratio设定为0.8",
        "永安期货资管部门构建涵盖RB2503、HC2503、I2504的黑色系期货组合，VaR控制在2%以内",
        "中信期货量化团队基于CU2504铜期货的momentum策略，年化Sharpe ratio达1.8",
        "国泰君安期货结构化产品部门设计与AU2506黄金期货挂钩的capital protected note",
        "光大期货机构业务部为钢铁企业定制RB2504螺纹钢期货套期保值方案，hedge effectiveness达95%",
        
        # 监管和合规术语
        "证监会要求期货公司加强对客户AP2502苹果期货交易的适当性管理",
        "中期协发布关于TA2505 PTA期货风险控制的自律规则补充条款",
        "郑商所对CF2504棉花期货异常交易行为实施监管措施，涉及5家期货公司",
        "期货保证金监控中心加强对ZC2505动力煤期货大户持仓的实时监控",
        "上期所发布SC2504原油期货交割库存周报，现货库存环比下降3.2%",
        
        # 跨境和国际化术语
        "中国期货业协会与芝商所就IF2503股指期货跨境监管达成合作协议",
        "上海国际能源交易中心SC2504原油期货与迪拜商品交易所阿曼原油期货价差分析",
        "大商所铁矿石期货I2504国际化进程加速，境外参与者数量增长显著",
        "郑商所PTA期货TA2505引入境外投资者，外资持仓占比达8.3%",
        "中金所正在研究推出基于MSCI中国指数的期货合约，预计2025年上市",
        
        # 极具挑战性的复合测试
        "华泰期货研究所首席分析师王明在其最新发布的《2025年黑色系期货投资策略报告》中指出，基于DCF估值模型和蒙特卡洛模拟，RB2503螺纹钢期货fair value区间为4000-4200元/吨，当前价格4120元/吨处于合理估值区间上沿，建议investor采用covered call策略获取alpha收益，同时通过dynamic hedging管理portfolio的duration risk和convexity risk。",
        
        "永安资本管理有限公司量化投资部门基于machine learning算法构建的multi-factor model显示，CF2505棉花期货价格与美棉期货、人民币汇率、原油价格的beta coefficients分别为0.75、-0.32、0.28，模型的adjusted R-squared达到0.82，out-of-sample testing的information ratio为1.65，建议通过pair trading策略对冲systematic risk。"
    ]
    
    # 复杂句法结构测试（保持原有并增强）
    enhanced_complex_syntax_tests = [
        "据华泰期货研究所最新发布的报告显示，受国际原油价格波动影响，shfe原油期货主力合约sc2024在昨日收盘时上涨3.2%",
        "郑州商品交易所白糖期货SR2024合约在经历了连续三个交易日的下跌后，今日开盘价格为5180元/吨，较前一交易日收盘价上涨0.8%",
        "中信期货分析师认为，在当前宏观经济环境下，大连商品交易所豆粕期货m2024合约价格将在3200-3400元/吨区间震荡运行",
        # ⭐ 新增复杂语法测试
        "永安期货公司旗下永安资本管理有限公司风险管理子公司在为某大型钢铁企业提供基于RB2505螺纹钢期货的套期保值服务时，采用了动态对冲策略，hedge ratio根据realized volatility和implied volatility的差异进行实时调整，有效control了企业raw material价格风险exposure。",
        "国泰君安期货衍生品研究团队通过对CF2504棉花期货历史价格进行econometric analysis发现，该合约与ICE棉花期货的长期cointegration relationship显著，error correction model的adjustment coefficient为-0.15，意味着price deviation会在6.7个交易日内correction 50%。"
    ]
    
    # 边界歧义测试（增强版）
    enhanced_boundary_ambiguity_tests = [
        "苹果公司股价上涨，但苹果期货价格下跌",
        "中国银行发布报告，中国银行期货子公司业务增长",
        # ⭐ 新增边界歧义测试
        "华泰证券华泰期货母子公司在AP2502苹果期货业务合作中发挥协同效应",
        "永安期货永安资本永安资管三个主体在RB2503螺纹钢期货套保业务中分工明确",
        "中信期货中信证券中信建投三家机构联合研究CF2505棉花期货投资价值",
        "大商所大连商品交易所大连期货三个不同概念需要准确区分",
        "郑商所AP2502苹果期货苹果现货苹果公司三个不同实体的market correlation analysis"
    ]
    
    # ⭐ 新增：RegEx失败案例专项测试
    regex_failure_cases = [
        # 合约代码与文字混合
        "郑商所AP2502期货怎么样了",
        "RB2503螺纹钢期货今天表现如何",
        "CF2505棉花期货价格走势分析",
        "TA2504PTA期货基本面研究",
        "SC2504原油期货技术分析报告",
        
        # 复杂嵌套结构
        "华泰期货在郑商所AP2502期货合约上的持仓情况",
        "永安期货对上期所CU2503铜期货的最新观点",
        "中信期货关于大商所M2505豆粕期货的投资建议",
        "国泰君安期货针对中金所IF2503股指期货的策略调整",
        "光大期货在INE SC2504原油期货上的风险管理措施",
        
        # 多实体混合
        "华泰期货、中信期货、永安期货三家公司对RB2503螺纹钢期货的观点分歧",
        "郑商所、大商所、上期所三大交易所的AP2502、M2505、CU2503主力合约分析",
        "期货公司、交易所、监管机构对TA2504PTA期货市场的不同态度"
    ]
    
    # 整合所有测试类型
    all_enhanced_tests = {
        "基础实体识别": basic_entity_tests,
        "专业挑战性测试": professional_challenging_tests,
        "增强复杂句法": enhanced_complex_syntax_tests,
        "增强边界歧义": enhanced_boundary_ambiguity_tests,
        "RegEx失败案例": regex_failure_cases
    }
    
    # 展平所有测试文本
    flat_tests = []
    for category, texts in all_enhanced_tests.items():
        flat_tests.extend(texts)
    
    return flat_tests, all_enhanced_tests

# ⭐ 新增：标签映射函数
def create_label_mapping():
    """创建spaCy标签到期货领域标签的映射"""
    return {
        # 组织机构映射
        "ORG": "FUTURES_COMPANY",  # 默认组织映射为期货公司
        "ORGANIZATION": "FUTURES_COMPANY",
        
        # 地理政治实体可能是交易所
        "GPE": "EXCHANGE",
        
        # 设施也可能是交易所
        "FAC": "EXCHANGE",
        
        # 保持不变的映射
        "EXCHANGE": "EXCHANGE",
        "FUTURES_COMPANY": "FUTURES_COMPANY", 
        "PRODUCT": "PRODUCT",
        "CONTRACT_CODE": "CONTRACT_CODE"
    }

# ⭐ 新增：智能实体后处理函数
def postprocess_entities_with_domain_knowledge(entities, text):
    """使用领域知识对实体进行后处理和重新标记"""
    
    processed_entities = []
    label_mapping = create_label_mapping()
    
    for start, end, label in entities:
        entity_text = text[start:end]
        new_label = label
        
        # ⭐ 基于文本内容的智能重新标记
        if label in ["ORG", "ORGANIZATION"]:
            # 期货公司识别
            if any(keyword in entity_text for keyword in ["期货", "资本", "资管"]):
                new_label = "FUTURES_COMPANY"
            # 交易所识别  
            elif any(keyword in entity_text for keyword in ["交易所", "商所", "所"]) and "期货" not in entity_text:
                new_label = "EXCHANGE"
            else:
                new_label = "FUTURES_COMPANY"  # 默认为期货公司
                
        elif label in ["GPE", "FAC"]:
            # 交易所识别
            if any(keyword in entity_text for keyword in ["交易所", "商所", "所", "郑商所", "大商所", "上期所", "中金所"]):
                new_label = "EXCHANGE"
            else:
                new_label = label_mapping.get(label, label)
                
        # ⭐ 合约代码识别 (正则表达式)
        import re
        if re.match(r'^[A-Z]{1,3}\d{4}$', entity_text):
            new_label = "CONTRACT_CODE"

# ⭐ 新增：性能指标计算类
class DetailedPerformanceMetrics:
    """详细性能指标计算类"""
    
    def __init__(self):
        self.true_entities = []
        self.pred_entities = []
        self.entity_type_performance = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        self.failure_cases = []
        
    def add_example(self, text, true_entities, pred_entities):
        """添加一个测试样例的结果"""
        self.true_entities.extend([(text, ent) for ent in true_entities])
        self.pred_entities.extend([(text, ent) for ent in pred_entities])
        
        # 计算实体级别的TP, FP, FN
        true_set = set(true_entities)
        pred_set = set(pred_entities)
        
        # 按类型统计
        for ent in true_set:
            if len(ent) >= 3:
                entity_type = ent[2]
                if ent in pred_set:
                    self.entity_type_performance[entity_type]["tp"] += 1
                else:
                    self.entity_type_performance[entity_type]["fn"] += 1
                    self.failure_cases.append({
                        "text": text,
                        "type": "FN",
                        "entity": ent,
                        "entity_type": entity_type
                    })
        
        for ent in pred_set:
            if len(ent) >= 3:
                entity_type = ent[2]
                if ent not in true_set:
                    self.entity_type_performance[entity_type]["fp"] += 1
                    self.failure_cases.append({
                        "text": text,
                        "type": "FP", 
                        "entity": ent,
                        "entity_type": entity_type
                    })
    
    def calculate_metrics(self):
        """计算详细的性能指标"""
        results = {}
        
        for entity_type, counts in self.entity_type_performance.items():
            tp = counts["tp"]
            fp = counts["fp"] 
            fn = counts["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn
            }
        
        # 计算宏平均和微平均
        total_tp = sum(counts["tp"] for counts in self.entity_type_performance.values())
        total_fp = sum(counts["fp"] for counts in self.entity_type_performance.values())
        total_fn = sum(counts["fn"] for counts in self.entity_type_performance.values())
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        precisions = [r["precision"] for r in results.values() if r["support"] > 0]
        recalls = [r["recall"] for r in results.values() if r["support"] > 0]
        f1s = [r["f1"] for r in results.values() if r["support"] > 0]
        
        macro_precision = np.mean(precisions) if precisions else 0
        macro_recall = np.mean(recalls) if recalls else 0
        macro_f1 = np.mean(f1s) if f1s else 0
        
        results["micro_avg"] = {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "support": total_tp + total_fn
        }
        
        results["macro_avg"] = {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "support": total_tp + total_fn
        }
        
        return results
    
    def analyze_failure_modes(self):
        """分析失败模式"""
        failure_analysis = {
            "by_type": defaultdict(lambda: {"FP": 0, "FN": 0}),
            "by_text_length": defaultdict(lambda: {"FP": 0, "FN": 0}),
            "common_errors": [],
            "detailed_cases": self.failure_cases
        }
        
        for case in self.failure_cases:
            entity_type = case["entity_type"]
            error_type = case["type"]
            text_length = len(case["text"])
            
            failure_analysis["by_type"][entity_type][error_type] += 1
            
            if text_length < 20:
                length_category = "short"
            elif text_length < 50:
                length_category = "medium"
            else:
                length_category = "long"
            
            failure_analysis["by_text_length"][length_category][error_type] += 1
        
        return failure_analysis

# 创建增强测试语料
print("🔄 创建增强测试语料库...")
enhanced_test_texts, enhanced_categorized_tests = create_enhanced_professional_test_corpus()
gold_standard = create_gold_standard_annotations()

print(f"📊 增强测试语料库统计")
print("=" * 60)
print(f"总文本数量: {len(enhanced_test_texts)}")
print(f"测试类别: {len(enhanced_categorized_tests)}")
print(f"黄金标准样本: {len(gold_standard)}")
print("\n各类别文本数量:")
for category, texts in enhanced_categorized_tests.items():
    print(f"  {category}: {len(texts)} 条")

print(f"\n📝 新增专业挑战性测试示例:")
print("-" * 40)
for i, text in enumerate(enhanced_categorized_tests["专业挑战性测试"][:3]):
    print(f"{i+1}. {text}")

# %% [markdown]
# ## 3. 增强基准测试执行

# %%
def enhanced_benchmark_with_detailed_metrics():
    """增强的基准测试，包含详细性能指标"""
    
    configurations = {
        "sm_full": {
            "model": "zh_core_web_sm",
            "exclude": [],
            "description": "小型模型完整配置 (46MB)"
        },
        "sm_ner_only": {
            "model": "zh_core_web_sm", 
            "exclude": ["parser", "tagger", "lemmatizer", "attribute_ruler"],
            "description": "小型模型仅NER (46MB)"
        },
        "md_full": {
            "model": "zh_core_web_md",
            "exclude": [],
            "description": "中型模型完整配置 (74MB)"
        },
        "md_ner_only": {
            "model": "zh_core_web_md", 
            "exclude": ["parser", "tagger", "lemmatizer", "attribute_ruler"],
            "description": "中型模型仅NER (74MB)"
        },
        "trf_full": {
            "model": "zh_core_web_trf",
            "exclude": [],
            "description": "Transformer模型完整配置 (396MB)"
        },
        "trf_ner_only": {
            "model": "zh_core_web_trf",
            "exclude": ["parser", "tagger", "lemmatizer", "attribute_ruler"],
            "description": "Transformer模型仅NER (396MB)"
        }
    }
    
    results = {}
    
    print("🚀 开始增强版NER模型质量基准测试")
    print("=" * 80)
    print(f"📊 测试语料: {len(enhanced_test_texts)} 个文本样本")
    print(f"🎯 测试配置: {len(configurations)} 个")
    print(f"🏆 关注指标: 精确率、召回率、F1分数、失败模式分析")
    print("=" * 80)
    
    for config_name, config in configurations.items():
        print(f"\n📊 测试配置: {config['description']}")
        print("-" * 60)
        
        try:
            # 加载模型
            print(f"⏳ 正在加载模型...")
            start_load = time.time()
            nlp = spacy.load(config["model"], exclude=config["exclude"])
            load_time = time.time() - start_load
            
            print(f"✅ 模型加载成功: {load_time:.2f}秒")
            
            # 预热模型
            print(f"🔥 预热模型...")
            for sample in enhanced_test_texts[:3]:
                _ = nlp(sample)
            
            # ⭐ 质量分析测试 - 专注于识别准确性
            print(f"⚡ 开始质量分析测试...")
            
            # 初始化性能指标计算器
            performance_metrics = DetailedPerformanceMetrics()
            
            all_entities = []
            entity_type_counts = {}
            category_performance = {}
            
            # 按类别测试
            for category, texts in enhanced_categorized_tests.items():
                category_metrics = DetailedPerformanceMetrics()
                
                for text in texts:
                    doc = nlp(text)
                    
                    # 提取预测实体
                    pred_entities = [(ent.start_char, ent.end_char, ent.label_) 
                                   for ent in doc.ents]
                    
                    # 获取黄金标准（如果存在）
                    true_entities = gold_standard.get(text, [])
                    
                    # 统计实体类型
                    for _, _, label in pred_entities:
                        entity_type_counts[label] = entity_type_counts.get(label, 0) + 1
                    
                    # 添加到性能指标计算
                    if true_entities:  # 只对有标注的样本计算详细指标
                        performance_metrics.add_example(text, true_entities, pred_entities)
                        category_metrics.add_example(text, true_entities, pred_entities)
                    
                    all_entities.append({
                        "text_id": len(all_entities),
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "category": category,
                        "true_entities": true_entities,
                        "pred_entities": pred_entities,
                        "entity_count": len(pred_entities),
                        "text_length": len(text)
                    })
                
                # 计算分类别性能
                if category in ["专业挑战性测试", "RegEx失败案例"]:  # 重点关注的类别
                    category_performance[category] = category_metrics.calculate_metrics()
            
            # ⭐ 计算详细性能指标
            detailed_metrics = performance_metrics.calculate_metrics()
            failure_analysis = performance_metrics.analyze_failure_modes()
            
            # 计算基本统计指标
            total_entities = sum(item["entity_count"] for item in all_entities)
            
            results[config_name] = {
                "model": config["model"],
                "description": config["description"],
                "excluded_components": config["exclude"],
                "active_pipes": nlp.pipe_names,
                
                # 基本识别统计
                "load_time": load_time,
                "total_entities_found": total_entities,
                "avg_entities_per_text": total_entities / len(enhanced_test_texts),
                "entity_type_counts": entity_type_counts,
                
                # ⭐ 核心质量指标
                "detailed_metrics": detailed_metrics,
                "failure_analysis": failure_analysis,
                "category_performance": category_performance,
                
                # 样本结果（限制数量）
                "detailed_results": all_entities[:5]
            }
            
            print(f"✅ 测试完成!")
            print(f"📊 处理文本总数: {len(enhanced_test_texts)} 个")
            print(f"🎯 发现实体总数: {total_entities}")
            
            # ⭐ 显示详细质量指标
            if detailed_metrics:
                micro_f1 = detailed_metrics.get("micro_avg", {}).get("f1", 0)
                macro_f1 = detailed_metrics.get("macro_avg", {}).get("f1", 0)
                micro_precision = detailed_metrics.get("micro_avg", {}).get("precision", 0)
                micro_recall = detailed_metrics.get("micro_avg", {}).get("recall", 0)
                print(f"📊 Micro Precision: {micro_precision:.3f}")
                print(f"📊 Micro Recall: {micro_recall:.3f}")
                print(f"📊 Micro F1-Score: {micro_f1:.3f}")
                print(f"📊 Macro F1-Score: {macro_f1:.3f}")
                
                # 显示主要实体类型的性能
                main_types = ["EXCHANGE", "FUTURES_COMPANY", "PRODUCT", "CONTRACT_CODE"]
                for entity_type in main_types:
                    if entity_type in detailed_metrics:
                        metrics = detailed_metrics[entity_type]
                        print(f"  {entity_type}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
            
        except OSError as e:
            print(f"❌ 无法加载模型 {config['model']}: {e}")
            results[config_name] = None
        except Exception as e:
            print(f"❌ 测试过程出错: {e}")
            results[config_name] = None
    
    return results

# 执行增强基准测试
print("🔄 开始执行增强基准测试...")
print("⚠️  注意: 测试可能需要几分钟时间，请耐心等待...")
enhanced_benchmark_results = enhanced_benchmark_with_detailed_metrics()

# %% [markdown]
# ## 4. 增强结果分析和可视化

# %%
def analyze_enhanced_results(results):
    """分析增强的测试结果"""
    
    print(f"\n📈 增强测试结果分析")
    print("=" * 80)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ 没有有效的测试结果")
        return None, None
    
    print(f"✅ 成功测试配置: {len(valid_results)}/{len(results)}")
    
    # ⭐ 创建质量导向的性能对比DataFrame
    comparison_data = []
    detailed_metrics_data = []
    
    for config_name, result in valid_results.items():
        # 模型基本信息
        size_map = {"sm": "46MB", "md": "74MB", "lg": "575MB", "trf": "396MB"}
        model_size = next((size for key, size in size_map.items() if key in config_name), "未知")
        
        # ⭐ 获取详细性能指标
        detailed_metrics = result.get("detailed_metrics", {})
        micro_avg = detailed_metrics.get("micro_avg", {})
        macro_avg = detailed_metrics.get("macro_avg", {})
        
        basic_data = {
            "配置名称": config_name,
            "模型描述": result["description"],
            "模型大小": model_size,
            "加载时间(秒)": result['load_time'],
            "发现实体数": result['total_entities_found'],
            "平均实体数": result['avg_entities_per_text'],
            # ⭐ 核心质量指标
            "Micro_Precision": micro_avg.get("precision", 0),
            "Micro_Recall": micro_avg.get("recall", 0),
            "Micro_F1": micro_avg.get("f1", 0),
            "Macro_Precision": macro_avg.get("precision", 0),
            "Macro_Recall": macro_avg.get("recall", 0),
            "Macro_F1": macro_avg.get("f1", 0),
            "质量评分": (micro_avg.get("precision", 0) + micro_avg.get("recall", 0) + micro_avg.get("f1", 0)) * 100 / 3
        }
        comparison_data.append(basic_data)
        
        # ⭐ 实体类型详细指标
        for entity_type, metrics in detailed_metrics.items():
            if entity_type not in ["micro_avg", "macro_avg"] and isinstance(metrics, dict):
                detailed_metrics_data.append({
                    "配置名称": config_name,
                    "实体类型": entity_type,
                    "精确率": metrics.get("precision", 0),
                    "召回率": metrics.get("recall", 0),
                    "F1分数": metrics.get("f1", 0),
                    "支持度": metrics.get("support", 0),
                    "True_Positive": metrics.get("tp", 0),
                    "False_Positive": metrics.get("fp", 0),
                    "False_Negative": metrics.get("fn", 0)
                })
    
    df_basic = pd.DataFrame(comparison_data)
    df_detailed = pd.DataFrame(detailed_metrics_data)
    
    # 显示基本性能对比
    print(f"\n📊 基本性能对比表")
    print("-" * 80)
    display(df_basic.round(3))
    
    # 显示详细性能指标
    if not df_detailed.empty:
        print(f"\n📊 实体类型详细性能指标")
        print("-" * 80)
        # 只显示主要实体类型
        main_types = ["EXCHANGE", "FUTURES_COMPANY", "PRODUCT", "CONTRACT_CODE", "ORG"]
        df_main = df_detailed[df_detailed["实体类型"].isin(main_types)]
        if not df_main.empty:
            display(df_main.round(3))
    
    return df_basic, df_detailed

# ⭐ 新增：失败模式详细分析函数
def analyze_failure_modes_detailed(results):
    """详细分析失败模式"""
    
    print(f"\n🔍 失败模式详细分析")
    print("=" * 80)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    for config_name, result in valid_results.items():
        failure_analysis = result.get("failure_analysis", {})
        
        if not failure_analysis:
            continue
            
        print(f"\n📋 模型: {result['description']}")
        print("-" * 50)
        
        # 按实体类型的失败分析
        by_type = failure_analysis.get("by_type", {})
        if by_type:
            print("🎯 按实体类型的错误分布:")
            for entity_type, errors in by_type.items():
                total_errors = errors["FP"] + errors["FN"]
                if total_errors > 0:
                    print(f"  {entity_type}: FP={errors['FP']}, FN={errors['FN']}, 总错误={total_errors}")
        
        # 按文本长度的失败分析
        by_length = failure_analysis.get("by_text_length", {})
        if by_length:
            print("\n📏 按文本长度的错误分布:")
            for length_cat, errors in by_length.items():
                total_errors = errors["FP"] + errors["FN"]
                if total_errors > 0:
                    print(f"  {length_cat}: FP={errors['FP']}, FN={errors['FN']}, 总错误={total_errors}")
        
        # 显示一些具体失败案例
        detailed_cases = failure_analysis.get("detailed_cases", [])
        if detailed_cases:
            print(f"\n❌ 典型失败案例 (显示前3个):")
            for i, case in enumerate(detailed_cases[:3]):
                print(f"  {i+1}. 类型: {case['type']}, 实体类型: {case['entity_type']}")
                print(f"     文本: {case['text'][:80]}...")
                print(f"     实体: {case['entity']}")

# ⭐ 新增：专业测试语料性能分析
def analyze_professional_test_performance(results):
    """分析专业测试语料的性能"""
    
    print(f"\n🎓 专业测试语料性能分析")
    print("=" * 80)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    # 分析各模型在专业测试上的表现
    professional_performance = {}
    
    for config_name, result in valid_results.items():
        category_performance = result.get("category_performance", {})
        
        # 重点关注专业挑战性测试和RegEx失败案例
        professional_cats = ["专业挑战性测试", "RegEx失败案例"]
        
        for cat in professional_cats:
            if cat in category_performance:
                metrics = category_performance[cat]
                micro_avg = metrics.get("micro_avg", {})
                
                if config_name not in professional_performance:
                    professional_performance[config_name] = {}
                
                professional_performance[config_name][cat] = {
                    "precision": micro_avg.get("precision", 0),
                    "recall": micro_avg.get("recall", 0),
                    "f1": micro_avg.get("f1", 0)
                }
    
    # 创建专业测试性能对比表
    if professional_performance:
        prof_data = []
        for config_name, categories in professional_performance.items():
            for category, metrics in categories.items():
                prof_data.append({
                    "模型配置": config_name,
                    "测试类别": category,
                    "精确率": metrics["precision"],
                    "召回率": metrics["recall"],
                    "F1分数": metrics["f1"]
                })
        
        if prof_data:
            df_prof = pd.DataFrame(prof_data)
            print("📊 专业测试语料性能对比:")
            display(df_prof.round(3))
            
            # 分析哪个模型在专业测试上表现最好
            pivot_f1 = df_prof.pivot(index="模型配置", columns="测试类别", values="F1分数")
            if not pivot_f1.empty:
                print(f"\n🏆 专业测试F1分数排名:")
                for category in pivot_f1.columns:
                    best_model = pivot_f1[category].idxmax()
                    best_score = pivot_f1[category].max()
                    print(f"  {category}: {best_model} (F1={best_score:.3f})")

# 执行增强分析
df_basic_enhanced, df_detailed_enhanced = analyze_enhanced_results(enhanced_benchmark_results)
analyze_failure_modes_detailed(enhanced_benchmark_results)
analyze_professional_test_performance(enhanced_benchmark_results)

# %% [markdown]
# ## 5. 增强可视化图表

# %%
def create_quality_focused_visualizations(df_basic, df_detailed, results):
    """创建专注于质量的可视化图表"""
    
    if df_basic.empty:
        print("❌ 没有数据可用于可视化")
        return
    
    # 设置图表布局 - 专注于质量指标
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NER Model Quality Analysis (Accuracy-Focused)', fontsize=18, fontweight='bold')
    
    # 1. ⭐ Micro F1-Score 对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(df_basic)), df_basic['Micro_F1'], color='lightgreen', alpha=0.7)
    ax1.set_title('Micro F1-Score Comparison', fontweight='bold')
    ax1.set_ylabel('F1-Score')
    ax1.set_xticks(range(len(df_basic)))
    ax1.set_xticklabels(df_basic['配置名称'], rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. ⭐ Macro F1-Score 对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(df_basic)), df_basic['Macro_F1'], color='orange', alpha=0.7)
    ax2.set_title('Macro F1-Score Comparison', fontweight='bold')
    ax2.set_ylabel('F1-Score')
    ax2.set_xticks(range(len(df_basic)))
    ax2.set_xticklabels(df_basic['配置名称'], rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. ⭐ 精确率 vs 召回率散点图
    ax3 = axes[0, 2]
    scatter = ax3.scatter(df_basic['Micro_Precision'], df_basic['Micro_Recall'], 
                         c=df_basic['Micro_F1'], cmap='viridis', 
                         s=150, alpha=0.7, edgecolors='black')
    ax3.set_xlabel('Micro Precision')
    ax3.set_ylabel('Micro Recall')
    ax3.set_title('Precision vs Recall (colored by F1-Score)', fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # 添加对角线
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    for i, txt in enumerate(df_basic['配置名称']):
        ax3.annotate(txt, (df_basic['Micro_Precision'].iloc[i], df_basic['Micro_Recall'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter, ax=ax3, label='F1-Score')
    
    # 4. ⭐ 实体类型性能热力图
    ax4 = axes[1, 0]
    if not df_detailed.empty:
        # 创建透视表
        heatmap_data = df_detailed.pivot_table(
            index='实体类型', 
            columns='配置名称', 
            values='F1分数', 
            aggfunc='mean'
        ).fillna(0)
        
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=ax4, cbar_kws={'label': 'F1-Score'})
            ax4.set_title('F1-Score by Entity Type', fontweight='bold')
            ax4.set_xlabel('Model Configuration')
            ax4.set_ylabel('Entity Type')
    
    # 5. ⭐ 质量指标雷达图
    ax5 = axes[1, 1]
    
    # 选择质量指标进行雷达图展示
    metrics = ['Micro_F1', 'Macro_F1', 'Micro_Precision', 'Micro_Recall']
    metric_labels = ['Micro F1', 'Macro F1', 'Precision', 'Recall']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 完成圆形
    
    ax5 = plt.subplot(2, 3, 5, projection='polar')
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_basic)))
    
    for i, (_, row) in enumerate(df_basic.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]
        
        ax5.plot(angles, values, 'o-', linewidth=2, 
                label=row['配置名称'], color=colors[i])
        ax5.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(metric_labels)
    ax5.set_ylim(0, 1)
    ax5.set_title('Quality Metrics Radar Chart', fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 6. ⭐ 失败模式分析图
    ax6 = axes[1, 2]
    
    # 统计各模型的失败案例数量
    failure_counts = {}
    for config_name, result in results.items():
        if result is not None:
            failure_analysis = result.get("failure_analysis", {})
            detailed_cases = failure_analysis.get("detailed_cases", [])
            failure_counts[config_name] = len(detailed_cases)
    
    if failure_counts:
        configs = list(failure_counts.keys())
        counts = list(failure_counts.values())
        
        bars6 = ax6.bar(configs, counts, color='lightcoral', alpha=0.7)
        ax6.set_title('Failure Cases Count', fontweight='bold')
        ax6.set_ylabel('Number of Failures')
        ax6.tick_params(axis='x', rotation=45)
        
        for i, bar in enumerate(bars6):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # ⭐ 显示质量导向的Top 3 配置排名
    print(f"\n🏆 Quality-Focused Performance Ranking")
    print("=" * 60)
    
    # 按质量指标排名
    rankings = {
        "Overall F1-Score": df_basic.nlargest(3, 'Micro_F1'),
        "Precision": df_basic.nlargest(3, 'Micro_Precision'),
        "Recall": df_basic.nlargest(3, 'Micro_Recall'),
        "Macro F1": df_basic.nlargest(3, 'Macro_F1')
    }
    
    medals = ["🥇", "🥈", "🥉"]
    
    for ranking_type, top_3 in rankings.items():
        print(f"\n🎯 {ranking_type} Top 3:")
        for i, (_, row) in enumerate(top_3.iterrows()):
            if ranking_type == "Overall F1-Score":
                score = f"F1={row['Micro_F1']:.3f}"
            elif ranking_type == "Precision":
                score = f"Precision={row['Micro_Precision']:.3f}"
            elif ranking_type == "Recall":
                score = f"Recall={row['Micro_Recall']:.3f}"
            else:  # Macro F1
                score = f"Macro F1={row['Macro_F1']:.3f}"
            
            print(f"  {medals[i]} {row['模型描述']}")
            print(f"     {score}")

# 创建质量导向的可视化
if df_basic_enhanced is not None:
    create_quality_focused_visualizations(df_basic_enhanced, df_detailed_enhanced, enhanced_benchmark_results)

# %% [markdown]
# ## 6. 增强优化建议

# %%
def generate_quality_focused_recommendations(df_basic, df_detailed, results):
    """生成专注于质量的优化建议"""
    
    if df_basic.empty:
        print("❌ 没有数据可用于生成建议")
        return
    
    print(f"\n💡 质量导向的模型优化建议")
    print("=" * 80)
    
    # 分析最优配置
    best_f1_config = df_basic.loc[df_basic['Micro_F1'].idxmax()]
    best_precision_config = df_basic.loc[df_basic['Micro_Precision'].idxmax()]
    best_recall_config = df_basic.loc[df_basic['Micro_Recall'].idxmax()]
    best_macro_f1_config = df_basic.loc[df_basic['Macro_F1'].idxmax()]
    best_overall_config = df_basic.loc[df_basic['质量评分'].idxmax()]
    
    print(f"🎯 **最佳F1分数配置**: {best_f1_config['模型描述']}")
    print(f"   📊 Micro F1: {best_f1_config['Micro_F1']:.3f}")
    print(f"   📈 Precision: {best_f1_config['Micro_Precision']:.3f}")
    print(f"   📉 Recall: {best_f1_config['Micro_Recall']:.3f}")
    print(f"   🎯 发现实体: {best_f1_config['发现实体数']} 个")
    
    print(f"\n🎖️ **最佳精确率配置**: {best_precision_config['模型描述']}")
    print(f"   📈 Precision: {best_precision_config['Micro_Precision']:.3f}")
    print(f"   📊 F1: {best_precision_config['Micro_F1']:.3f}")
    print(f"   📉 Recall: {best_precision_config['Micro_Recall']:.3f}")
    
    print(f"\n🔍 **最佳召回率配置**: {best_recall_config['模型描述']}")
    print(f"   📉 Recall: {best_recall_config['Micro_Recall']:.3f}")
    print(f"   📊 F1: {best_recall_config['Micro_F1']:.3f}")
    print(f"   📈 Precision: {best_recall_config['Micro_Precision']:.3f}")
    
    print(f"\n🌟 **最佳宏平均F1配置**: {best_macro_f1_config['模型描述']}")
    print(f"   📊 Macro F1: {best_macro_f1_config['Macro_F1']:.3f}")
    print(f"   📊 Micro F1: {best_macro_f1_config['Micro_F1']:.3f}")
    
    print(f"\n🏆 **综合质量最优配置**: {best_overall_config['模型描述']}")
    print(f"   🎖️ 质量评分: {best_overall_config['质量评分']:.1f}")
    print(f"   📊 Micro F1: {best_overall_config['Micro_F1']:.3f}")
    print(f"   📈 Precision: {best_overall_config['Micro_Precision']:.3f}")
    print(f"   📉 Recall: {best_overall_config['Micro_Recall']:.3f}")
    
    # ⭐ 专业场景质量优化建议
    print(f"\n📋 **专业期货交易场景质量优化建议**:")
    print(f"=" * 60)
    
    scenarios = [
        {
            "scenario": "🎯 精确合约识别 (准确率优先)",
            "recommendation": best_precision_config['配置名称'],
            "rationale": "合约代码识别需要极高的精确率，避免误识别",
            "config": f"spacy.load('{best_precision_config['配置名称'].split('_')[0]}_core_web_{best_precision_config['配置名称'].split('_')[1]}', exclude={['parser','tagger','lemmatizer','attribute_ruler'] if 'ner_only' in best_precision_config['配置名称'] else []})",
            "use_cases": ["合约代码自动识别", "交易指令解析", "风险敞口计算"],
            "metrics": f"Precision={best_precision_config['Micro_Precision']:.3f}, F1={best_precision_config['Micro_F1']:.3f}"
        },
        {
            "scenario": "🔍 全面信息提取 (召回率优先)",
            "recommendation": best_recall_config['配置名称'],
            "rationale": "需要尽可能多地识别出所有相关实体，避免遗漏",
            "config": f"spacy.load('{best_recall_config['配置名称'].split('_')[0]}_core_web_{best_recall_config['配置名称'].split('_')[1]}', exclude={['parser','tagger','lemmatizer','attribute_ruler'] if 'ner_only' in best_recall_config['配置名称'] else []})",
            "use_cases": ["监管合规检查", "全量数据挖掘", "历史文档分析"],
            "metrics": f"Recall={best_recall_config['Micro_Recall']:.3f}, F1={best_recall_config['Micro_F1']:.3f}"
        },
        {
            "scenario": "⚖️ 平衡性能应用 (F1分数最优)",
            "recommendation": best_f1_config['配置名称'],
            "rationale": "在精确率和召回率之间找到最佳平衡点",
            "config": f"spacy.load('{best_f1_config['配置名称'].split('_')[0]}_core_web_{best_f1_config['配置名称'].split('_')[1]}', exclude={['parser','tagger','lemmatizer','attribute_ruler'] if 'ner_only' in best_f1_config['配置名称'] else []})",
            "use_cases": ["研究报告解析", "客户查询响应", "智能问答系统"],
            "metrics": f"F1={best_f1_config['Micro_F1']:.3f}, P={best_f1_config['Micro_Precision']:.3f}, R={best_f1_config['Micro_Recall']:.3f}"
        },
        {
            "scenario": "🌈 多类别均衡 (宏平均F1优先)",
            "recommendation": best_macro_f1_config['配置名称'],
            "rationale": "确保各种实体类型都有较好的识别效果",
            "config": f"spacy.load('{best_macro_f1_config['配置名称'].split('_')[0]}_core_web_{best_macro_f1_config['配置名称'].split('_')[1]}', exclude={['parser','tagger','lemmatizer','attribute_ruler'] if 'ner_only' in best_macro_f1_config['配置名称'] else []})",
            "use_cases": ["多元化信息提取", "跨类别分析", "完整性检查"],
            "metrics": f"Macro F1={best_macro_f1_config['Macro_F1']:.3f}, Micro F1={best_macro_f1_config['Micro_F1']:.3f}"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['scenario']}")
        print(f"   🔧 推荐配置: {scenario['recommendation']}")
        print(f"   💡 选择理由: {scenario['rationale']}")
        print(f"   📊 性能指标: {scenario['metrics']}")
        print(f"   💻 代码示例: {scenario['config']}")
        print(f"   📝 适用场景: {', '.join(scenario['use_cases'])}")
    
    # ⭐ 失败模式特定优化建议
    print(f"\n🛠️ **针对失败模式的质量提升策略**:")
    print(f"=" * 60)
    
    # 分析主要失败模式
    main_failure_types = set()
    for config_name, result in results.items():
        if result is not None:
            failure_analysis = result.get("failure_analysis", {})
            by_type = failure_analysis.get("by_type", {})
            main_failure_types.update(by_type.keys())
    
    optimization_strategies = {
        "CONTRACT_CODE": [
            "✅ 使用自定义规则匹配增强合约代码识别准确性",
            "✅ 构建专门的合约代码词典进行后处理验证",
            "✅ 训练针对期货合约代码的特化模型",
            "✅ 使用正则表达式预筛选候选实体，提高精确率",
            "✅ 建立合约代码格式验证机制"
        ],
        "EXCHANGE": [
            "✅ 建立交易所别名映射表，统一不同表述",
            "✅ 增加中英文混合表达的训练样本", 
            "✅ 使用基于规则的后处理纠正识别错误",
            "✅ 考虑上下文信息提高歧义消解能力",
            "✅ 维护交易所官方名称与简称对照表"
        ],
        "FUTURES_COMPANY": [
            "✅ 维护期货公司全称与简称对照表",
            "✅ 处理复杂的企业组织架构关系",
            "✅ 增强对子公司、分支机构的识别",
            "✅ 使用实体链接技术统一不同表述",
            "✅ 建立期货公司业务范围识别规则"
        ],
        "PRODUCT": [
            "✅ 区分期货品种与其他同名实体",
            "✅ 建立品种代码与中文名称映射关系",
            "✅ 处理品种名称的多种变体表达",
            "✅ 结合上下文判断实体的真实含义",
            "✅ 使用品种分类规则提高识别准确性"
        ]
    }
    
    for entity_type, strategies in optimization_strategies.items():
        if entity_type in main_failure_types:
            print(f"\n📋 {entity_type} 实体质量提升策略:")
            for strategy in strategies:
                print(f"   {strategy}")
    
    # ⭐ 质量优化代码实现
    print(f"\n💻 **质量优化代码实现建议**:")
    print(f"=" * 60)
    
    code_examples = f"""
# 1. 高精确率NER配置 (适用于关键业务)
import spacy
from typing import List, Tuple, Dict

def setup_high_precision_ner():
    '''设置高精确率的NER管道'''
    # 使用精确率最高的配置
    nlp = spacy.load("{best_precision_config['配置名称'].split('_')[0]}_core_web_{best_precision_config['配置名称'].split('_')[1]}")
    
    # 精确率优先的后处理
    def high_precision_postprocess(entities: List[Tuple], text: str, confidence_threshold: float = 0.8) -> List[Tuple]:
        '''后处理：提高精确率，降低误报'''
        validated_entities = []
        
        for start, end, label in entities:
            entity_text = text[start:end]
            confidence = calculate_entity_confidence(entity_text, label)
            
            # 只保留高置信度的实体
            if confidence >= confidence_threshold:
                validated_entities.append((start, end, label))
        
        return validated_entities
    
    def calculate_entity_confidence(entity_text: str, label: str) -> float:
        '''计算实体置信度'''
        # 基于规则的置信度计算
        confidence = 0.5  # 基础置信度
        
        if label == "CONTRACT_CODE":
            # 合约代码格式检查
            import re
            if re.match(r'^[A-Z]{{1,3}}\\d{{4}}

# 生成质量导向的优化建议
if df_basic_enhanced is not None:
    generate_quality_focused_recommendations(df_basic_enhanced, df_detailed_enhanced, enhanced_benchmark_results)

# ⭐ 保存质量分析结果
def save_quality_focused_results(basic_results, detailed_results, benchmark_results):
    """保存质量导向的测试结果"""
    
    print(f"\n💾 保存质量分析结果")
    print("=" * 60)
    
    # 保存基本质量对比
    if basic_results is not None and not basic_results.empty:
        basic_results.to_csv("quality_focused_ner_comparison.csv", index=False, encoding="utf-8")
        print(f"✅ 基本质量对比已保存到: quality_focused_ner_comparison.csv")
    
    # 保存详细实体性能
    if detailed_results is not None and not detailed_results.empty:
        detailed_results.to_csv("detailed_entity_quality_metrics.csv", index=False, encoding="utf-8")
        print(f"✅ 详细实体性能已保存到: detailed_entity_quality_metrics.csv")
    
    # 保存原始基准测试结果
    clean_results = {}
    for config_name, result in benchmark_results.items():
        if result is not None:
            clean_result = result.copy()
            # 移除不可序列化的对象
            clean_result.pop("detailed_results", None)
            # 简化失败分析数据
            if "failure_analysis" in clean_result:
                failure_analysis = clean_result["failure_analysis"]
                if "detailed_cases" in failure_analysis:
                    # 只保留前10个失败案例
                    failure_analysis["detailed_cases"] = failure_analysis["detailed_cases"][:10]
            clean_results[config_name] = clean_result
    
    with open("quality_focused_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ 完整质量分析结果已保存到: quality_focused_benchmark_results.json")
    print(f"📊 结果文件专注于识别质量分析，可在Excel中进一步分析")

# 保存结果
save_quality_focused_results(df_basic_enhanced, df_detailed_enhanced, enhanced_benchmark_results)

# 总结
print(f"\n🎉 **质量导向NER模型基准测试完成!**")
print("=" * 80)
print(f"📊 本次测试评估了 {len([r for r in enhanced_benchmark_results.values() if r is not None])} 个有效配置")
print(f"📝 使用了 {len(enhanced_test_texts)} 个增强测试文本，包含专业期货合约代码测试")
print(f"🎯 重点分析了精确率、召回率、F1分数等质量指标")
print(f"🔍 提供了详细的失败模式分析和质量提升建议")
print(f"💡 针对期货交易场景给出了专业的质量优化策略")
print(f"\n🏆 主要发现:")
if df_basic_enhanced is not None and not df_basic_enhanced.empty:
    best_f1_model = df_basic_enhanced.loc[df_basic_enhanced['Micro_F1'].idxmax()]
    best_precision_model = df_basic_enhanced.loc[df_basic_enhanced['Micro_Precision'].idxmax()]
    best_recall_model = df_basic_enhanced.loc[df_basic_enhanced['Micro_Recall'].idxmax()]
    print(f"   📈 最佳F1分数: {best_f1_model['模型描述']} (F1={best_f1_model['Micro_F1']:.3f})")
    print(f"   🎯 最佳精确率: {best_precision_model['模型描述']} (Precision={best_precision_model['Micro_Precision']:.3f})")
    print(f"   🔍 最佳召回率: {best_recall_model['模型描述']} (Recall={best_recall_model['Micro_Recall']:.3f})")
    print(f"   🎓 专业合约代码识别能力得到重点测试和分析")
print(f"\n💻 建议根据具体质量要求选择合适的模型配置进行部署!")
print(f"🎯 重点关注precision/recall权衡，针对业务场景优化识别质量!")
, entity_text):
                confidence += 0.4
        elif label == "EXCHANGE":
            # 交易所名称验证
            exchange_keywords = ["交易所", "商所", "所"]
            if any(keyword in entity_text for keyword in exchange_keywords):
                confidence += 0.3
        elif label == "FUTURES_COMPANY":
            # 期货公司名称验证
            if "期货" in entity_text:
                confidence += 0.3
        
        return min(confidence, 1.0)
    
    return nlp, high_precision_postprocess

# 2. 高召回率NER配置 (适用于信息收集)
def setup_high_recall_ner():
    '''设置高召回率的NER管道'''
    # 使用召回率最高的配置
    nlp = spacy.load("{best_recall_config['配置名称'].split('_')[0]}_core_web_{best_recall_config['配置名称'].split('_')[1]}")
    
    # 多模式匹配增强召回率
    from spacy.matcher import Matcher
    matcher = Matcher(nlp.vocab)
    
    # 添加更多匹配模式
    patterns = {{
        "CONTRACT_CODE": [
            [{{"TEXT": {{"REGEX": r"[A-Z]{{1,3}}\\d{{4}}"}}}}],  # 标准合约代码
            [{{"TEXT": {{"REGEX": r"[A-Za-z]{{1,3}}\\d{{4}}"}}}}],  # 包含小写字母
        ],
        "EXCHANGE_ALIAS": [
            [{{"LOWER": "上期所"}}],
            [{{"LOWER": "大商所"}}],
            [{{"LOWER": "郑商所"}}],
            [{{"LOWER": "中金所"}}],
        ]
    }}
    
    for label, pattern_list in patterns.items():
        matcher.add(label, pattern_list)
    
    def high_recall_extract(text: str) -> List[Tuple]:
        '''高召回率实体提取'''
        doc = nlp(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        
        # 添加规则匹配的结果
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = nlp.vocab.strings[match_id]
            entities.append((span.start_char, span.end_char, label))
        
        # 去重
        entities = list(set(entities))
        return sorted(entities)
    
    return high_recall_extract

# 3. 平衡F1分数的NER配置 (推荐用于生产环境)
def setup_balanced_ner():
    '''设置平衡F1分数的NER管道'''
    # 使用F1分数最高的配置
    nlp = spacy.load("{best_f1_config['配置名称'].split('_')[0]}_core_web_{best_f1_config['配置名称'].split('_')[1]}")
    
    def balanced_ner_pipeline(text: str) -> Dict[str, any]:
        '''平衡的NER管道，返回详细结果'''
        doc = nlp(text)
        
        entities = []
        for ent in doc.ents:
            entity_info = {{
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": calculate_entity_confidence(ent.text, ent.label_)
            }}
            entities.append(entity_info)
        
        return {{
            "entities": entities,
            "entity_count": len(entities),
            "text_length": len(text),
            "model_info": "{{}}".format("{best_f1_config['模型描述']}")
        }}
    
    return balanced_ner_pipeline

# 4. 质量监控和评估函数
def monitor_ner_quality(predictions: List[Tuple], ground_truth: List[Tuple]) -> Dict[str, float]:
    '''监控NER质量'''
    from sklearn.metrics import precision_recall_fscore_support
    
    # 转换为标签序列进行评估
    pred_labels = [label for _, _, label in predictions]
    true_labels = [label for _, _, label in ground_truth]
    
    if len(pred_labels) == 0 and len(true_labels) == 0:
        return {{"precision": 1.0, "recall": 1.0, "f1": 1.0}}
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='micro', zero_division=0
    )
    
    return {{
        "precision": precision,
        "recall": recall, 
        "f1": f1,
        "total_predictions": len(predictions),
        "total_ground_truth": len(ground_truth)
    }}

# 5. 生产环境部署示例
class ProductionNERService:
    '''生产环境NER服务'''
    
    def __init__(self, model_type: str = "balanced"):
        self.model_type = model_type
        self.quality_stats = {{
            "total_processed": 0,
            "total_entities": 0,
            "avg_confidence": 0.0
        }}
        
        if model_type == "precision":
            self.nlp, self.postprocess = setup_high_precision_ner()
        elif model_type == "recall":
            self.extract_func = setup_high_recall_ner()
        else:  # balanced
            self.pipeline = setup_balanced_ner()
    
    def extract_entities(self, text: str) -> Dict[str, any]:
        '''提取实体'''
        if self.model_type == "balanced":
            result = self.pipeline(text)
        else:
            # 其他类型的处理逻辑
            entities = self.extract_func(text) if self.model_type == "recall" else []
            result = {{"entities": entities}}
        
        # 更新统计信息
        self.quality_stats["total_processed"] += 1
        self.quality_stats["total_entities"] += len(result["entities"])
        
        return result
    
    def get_quality_report(self) -> Dict[str, any]:
        '''获取质量报告'''
        avg_entities = (self.quality_stats["total_entities"] / 
                       max(self.quality_stats["total_processed"], 1))
        
        return {{
            "model_type": self.model_type,
            "total_processed": self.quality_stats["total_processed"],
            "avg_entities_per_text": avg_entities,
            "recommended_for": self._get_recommendation()
        }}
    
    def _get_recommendation(self) -> str:
        '''获取使用建议'''
        recommendations = {{
            "precision": "关键业务场景，需要高准确率",
            "recall": "信息收集场景，需要高覆盖率", 
            "balanced": "一般业务场景，平衡准确率和覆盖率"
        }}
        return recommendations.get(self.model_type, "通用场景")

# 使用示例
if __name__ == "__main__":
    # 测试文本
    test_text = "郑商所AP2502期货怎么样了"
    
    # 不同质量目标的处理
    print("高精确率处理:")
    nlp_precision, postprocess = setup_high_precision_ner()
    doc = nlp_precision(test_text)
    entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    validated_entities = postprocess(entities, test_text)
    print(f"结果: {{validated_entities}}")
    
    print("\\n高召回率处理:")
    extract_recall = setup_high_recall_ner()
    recall_entities = extract_recall(test_text)
    print(f"结果: {{recall_entities}}")
    
    print("\\n平衡处理:")
    balanced_pipeline = setup_balanced_ner()
    balanced_result = balanced_pipeline(test_text)
    print(f"结果: {{balanced_result}}")
    
    # 生产服务示例
    service = ProductionNERService("balanced")
    result = service.extract_entities(test_text)
    quality_report = service.get_quality_report()
    print(f"\\n生产服务结果: {{result}}")
    print(f"质量报告: {{quality_report}}")
"""
    
    print(code_examples)

# 生成增强优化建议
if df_basic_enhanced is not None:
    generate_enhanced_optimization_recommendations(df_basic_enhanced, df_detailed_enhanced, enhanced_benchmark_results)

# ⭐ 保存增强结果
def save_enhanced_results(basic_results, detailed_results, benchmark_results):
    """保存增强的测试结果"""
    
    print(f"\n💾 保存增强测试结果")
    print("=" * 60)
    
    # 保存基本性能对比
    if basic_results is not None and not basic_results.empty:
        basic_results.to_csv("enhanced_ner_performance_comparison.csv", index=False, encoding="utf-8")
        print(f"✅ 基本性能对比已保存到: enhanced_ner_performance_comparison.csv")
    
    # 保存详细实体性能
    if detailed_results is not None and not detailed_results.empty:
        detailed_results.to_csv("detailed_entity_performance.csv", index=False, encoding="utf-8")
        print(f"✅ 详细实体性能已保存到: detailed_entity_performance.csv")
    
    # 保存原始基准测试结果
    clean_results = {}
    for config_name, result in benchmark_results.items():
        if result is not None:
            clean_result = result.copy()
            # 移除不可序列化的对象
            clean_result.pop("detailed_results", None)
            # 简化失败分析数据
            if "failure_analysis" in clean_result:
                failure_analysis = clean_result["failure_analysis"]
                if "detailed_cases" in failure_analysis:
                    # 只保留前10个失败案例
                    failure_analysis["detailed_cases"] = failure_analysis["detailed_cases"][:10]
            clean_results[config_name] = clean_result
    
    with open("enhanced_ner_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ 完整基准测试结果已保存到: enhanced_ner_benchmark_results.json")
    print(f"📊 结果文件可在Excel或其他工具中进一步分析")

# 保存结果
save_enhanced_results(df_basic_enhanced, df_detailed_enhanced, enhanced_benchmark_results)

# 总结
print(f"\n🎉 **增强版NER模型基准测试完成!**")
print("=" * 80)
print(f"📊 本次测试评估了 {len([r for r in enhanced_benchmark_results.values() if r is not None])} 个有效配置")
print(f"📝 使用了 {len(enhanced_test_texts)} 个增强测试文本，包含专业期货合约代码测试")
print(f"🎯 新增了精确率、召回率、F1分数等详细性能指标")
print(f"🔍 提供了详细的失败模式分析和优化建议")
print(f"💡 针对期货交易场景给出了专业的模型选择和部署建议")
print(f"\n🏆 主要发现:")
if df_basic_enhanced is not None and not df_basic_enhanced.empty:
    best_f1_model = df_basic_enhanced.loc[df_basic_enhanced['Micro_F1'].idxmax()]
    fastest_model = df_basic_enhanced.loc[df_basic_enhanced['平均延迟(毫秒)'].idxmin()]
    print(f"   📈 最佳F1分数: {best_f1_model['模型描述']} (F1={best_f1_model['Micro_F1']:.3f})")
    print(f"   ⚡ 最快处理速度: {fastest_model['模型描述']} ({fastest_model['平均延迟(毫秒)']:.1f}ms)")
    print(f"   🎯 专业合约代码识别能力得到重点测试和分析")
print(f"\n💻 建议根据具体业务场景选择合适的模型配置进行部署!")
, entity_text):
            new_label = "CONTRACT_CODE"
            
        # ⭐ 产品/品种识别
        elif any(keyword in entity_text for keyword in ["期货", "合约"]) and label not in ["FUTURES_COMPANY", "EXCHANGE"]:
            new_label = "PRODUCT"
            
        # 使用映射表进行最终转换
        final_label = label_mapping.get(new_label, new_label)
        processed_entities.append((start, end, final_label))
    
    return processed_entities

# ⭐ 新增：黄金标准实体标注函数  
def create_gold_standard_annotations():
    """创建黄金标准实体标注，用于精确计算性能指标"""
    
    gold_annotations = {
        # 基础测试样本的黄金标准
        "苹果期货在郑州商品交易所交易，华泰期货公司参与其中": [
            (0, 2, "PRODUCT"),           # 苹果
            (5, 12, "EXCHANGE"),         # 郑州商品交易所
            (15, 22, "FUTURES_COMPANY")  # 华泰期货公司
        ],
        
        "上海期货交易所的铜期货价格上涨，中信期货发布研究报告": [
            (0, 7, "EXCHANGE"),          # 上海期货交易所
            (8, 10, "PRODUCT"),          # 铜期货
            (17, 21, "FUTURES_COMPANY")  # 中信期货
        ],
        
        # ⭐ 专业挑战性测试的黄金标准
        "郑商所AP2502期货怎么样了": [
            (0, 3, "EXCHANGE"),          # 郑商所
            (3, 9, "CONTRACT_CODE"),     # AP2502
            (9, 11, "PRODUCT")           # 期货
        ],
        
        "上期所CU2503合约今日涨停，华泰期货建议关注": [
            (0, 3, "EXCHANGE"),          # 上期所
            (3, 9, "CONTRACT_CODE"),     # CU2503
            (9, 11, "PRODUCT"),          # 合约
            (16, 20, "FUTURES_COMPANY")  # 华泰期货
        ],
        
        "大商所M2505豆粕期货主力合约成交活跃": [
            (0, 3, "EXCHANGE"),          # 大商所
            (3, 8, "CONTRACT_CODE"),     # M2505
            (8, 10, "PRODUCT"),          # 豆粕
            (10, 12, "PRODUCT")          # 期货
        ],
        
        "方正中期期货风险管理子公司在PTA2504合约上建立空头套保头寸": [
            (0, 6, "FUTURES_COMPANY"),   # 方正中期期货
            (6, 13, "ORGANIZATION"),     # 风险管理子公司
            (15, 22, "CONTRACT_CODE"),   # PTA2504
            (22, 24, "PRODUCT")          # 合约
        ],
        
        "华泰期货研究所分析师认为ZC2505动力煤期货价格将震荡上行": [
            (0, 7, "FUTURES_COMPANY"),   # 华泰期货研究所
            (13, 19, "CONTRACT_CODE"),   # ZC2505
            (19, 22, "PRODUCT"),         # 动力煤
            (22, 24, "PRODUCT")          # 期货
        ]
    }
    
    return gold_annotations

# ⭐ 新增：性能指标计算类
class DetailedPerformanceMetrics:
    """详细性能指标计算类"""
    
    def __init__(self):
        self.true_entities = []
        self.pred_entities = []
        self.entity_type_performance = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        self.failure_cases = []
        
    def add_example(self, text, true_entities, pred_entities):
        """添加一个测试样例的结果"""
        self.true_entities.extend([(text, ent) for ent in true_entities])
        self.pred_entities.extend([(text, ent) for ent in pred_entities])
        
        # 计算实体级别的TP, FP, FN
        true_set = set(true_entities)
        pred_set = set(pred_entities)
        
        # 按类型统计
        for ent in true_set:
            if len(ent) >= 3:
                entity_type = ent[2]
                if ent in pred_set:
                    self.entity_type_performance[entity_type]["tp"] += 1
                else:
                    self.entity_type_performance[entity_type]["fn"] += 1
                    self.failure_cases.append({
                        "text": text,
                        "type": "FN",
                        "entity": ent,
                        "entity_type": entity_type
                    })
        
        for ent in pred_set:
            if len(ent) >= 3:
                entity_type = ent[2]
                if ent not in true_set:
                    self.entity_type_performance[entity_type]["fp"] += 1
                    self.failure_cases.append({
                        "text": text,
                        "type": "FP", 
                        "entity": ent,
                        "entity_type": entity_type
                    })
    
    def calculate_metrics(self):
        """计算详细的性能指标"""
        results = {}
        
        for entity_type, counts in self.entity_type_performance.items():
            tp = counts["tp"]
            fp = counts["fp"] 
            fn = counts["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn
            }
        
        # 计算宏平均和微平均
        total_tp = sum(counts["tp"] for counts in self.entity_type_performance.values())
        total_fp = sum(counts["fp"] for counts in self.entity_type_performance.values())
        total_fn = sum(counts["fn"] for counts in self.entity_type_performance.values())
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        precisions = [r["precision"] for r in results.values() if r["support"] > 0]
        recalls = [r["recall"] for r in results.values() if r["support"] > 0]
        f1s = [r["f1"] for r in results.values() if r["support"] > 0]
        
        macro_precision = np.mean(precisions) if precisions else 0
        macro_recall = np.mean(recalls) if recalls else 0
        macro_f1 = np.mean(f1s) if f1s else 0
        
        results["micro_avg"] = {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "support": total_tp + total_fn
        }
        
        results["macro_avg"] = {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "support": total_tp + total_fn
        }
        
        return results
    
    def analyze_failure_modes(self):
        """分析失败模式"""
        failure_analysis = {
            "by_type": defaultdict(lambda: {"FP": 0, "FN": 0}),
            "by_text_length": defaultdict(lambda: {"FP": 0, "FN": 0}),
            "common_errors": [],
            "detailed_cases": self.failure_cases
        }
        
        for case in self.failure_cases:
            entity_type = case["entity_type"]
            error_type = case["type"]
            text_length = len(case["text"])
            
            failure_analysis["by_type"][entity_type][error_type] += 1
            
            if text_length < 20:
                length_category = "short"
            elif text_length < 50:
                length_category = "medium"
            else:
                length_category = "long"
            
            failure_analysis["by_text_length"][length_category][error_type] += 1
        
        return failure_analysis

# 创建增强测试语料
print("🔄 创建增强测试语料库...")
enhanced_test_texts, enhanced_categorized_tests = create_enhanced_professional_test_corpus()
gold_standard = create_gold_standard_annotations()

print(f"📊 增强测试语料库统计")
print("=" * 60)
print(f"总文本数量: {len(enhanced_test_texts)}")
print(f"测试类别: {len(enhanced_categorized_tests)}")
print(f"黄金标准样本: {len(gold_standard)}")
print("\n各类别文本数量:")
for category, texts in enhanced_categorized_tests.items():
    print(f"  {category}: {len(texts)} 条")

print(f"\n📝 新增专业挑战性测试示例:")
print("-" * 40)
for i, text in enumerate(enhanced_categorized_tests["专业挑战性测试"][:3]):
    print(f"{i+1}. {text}")

# %% [markdown]
# ## 3. 增强基准测试执行

# %%
def enhanced_benchmark_with_detailed_metrics():
    """增强的基准测试，包含详细性能指标"""
    
    configurations = {
        "sm_full": {
            "model": "zh_core_web_sm",
            "exclude": [],
            "description": "小型模型完整配置 (46MB)"
        },
        "sm_ner_only": {
            "model": "zh_core_web_sm", 
            "exclude": ["parser", "tagger", "lemmatizer", "attribute_ruler"],
            "description": "小型模型仅NER (46MB)"
        },
        "md_full": {
            "model": "zh_core_web_md",
            "exclude": [],
            "description": "中型模型完整配置 (74MB)"
        },
        "md_ner_only": {
            "model": "zh_core_web_md", 
            "exclude": ["parser", "tagger", "lemmatizer", "attribute_ruler"],
            "description": "中型模型仅NER (74MB)"
        },
        "trf_full": {
            "model": "zh_core_web_trf",
            "exclude": [],
            "description": "Transformer模型完整配置 (396MB)"
        },
        "trf_ner_only": {
            "model": "zh_core_web_trf",
            "exclude": ["parser", "tagger", "lemmatizer", "attribute_ruler"],
            "description": "Transformer模型仅NER (396MB)"
        }
    }
    
    results = {}
    
    print("🚀 开始增强版NER模型质量基准测试")
    print("=" * 80)
    print(f"📊 测试语料: {len(enhanced_test_texts)} 个文本样本")
    print(f"🎯 测试配置: {len(configurations)} 个")
    print(f"🏆 关注指标: 精确率、召回率、F1分数、失败模式分析")
    print("=" * 80)
    
    for config_name, config in configurations.items():
        print(f"\n📊 测试配置: {config['description']}")
        print("-" * 60)
        
        try:
            # 加载模型
            print(f"⏳ 正在加载模型...")
            start_load = time.time()
            nlp = spacy.load(config["model"], exclude=config["exclude"])
            load_time = time.time() - start_load
            
            print(f"✅ 模型加载成功: {load_time:.2f}秒")
            
            # 预热模型
            print(f"🔥 预热模型...")
            for sample in enhanced_test_texts[:3]:
                _ = nlp(sample)
            
            # ⭐ 质量分析测试 - 专注于识别准确性
            print(f"⚡ 开始质量分析测试...")
            
            # 初始化性能指标计算器
            performance_metrics = DetailedPerformanceMetrics()
            
            all_entities = []
            entity_type_counts = {}
            category_performance = {}
            
            # 按类别测试
            for category, texts in enhanced_categorized_tests.items():
                category_metrics = DetailedPerformanceMetrics()
                
                for text in texts:
                    doc = nlp(text)
                    
                    # 提取预测实体
                    pred_entities = [(ent.start_char, ent.end_char, ent.label_) 
                                   for ent in doc.ents]
                    
                    # 获取黄金标准（如果存在）
                    true_entities = gold_standard.get(text, [])
                    
                    # 统计实体类型
                    for _, _, label in pred_entities:
                        entity_type_counts[label] = entity_type_counts.get(label, 0) + 1
                    
                    # 添加到性能指标计算
                    if true_entities:  # 只对有标注的样本计算详细指标
                        performance_metrics.add_example(text, true_entities, pred_entities)
                        category_metrics.add_example(text, true_entities, pred_entities)
                    
                    all_entities.append({
                        "text_id": len(all_entities),
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "category": category,
                        "true_entities": true_entities,
                        "pred_entities": pred_entities,
                        "entity_count": len(pred_entities),
                        "text_length": len(text)
                    })
                
                # 计算分类别性能
                if category in ["专业挑战性测试", "RegEx失败案例"]:  # 重点关注的类别
                    category_performance[category] = category_metrics.calculate_metrics()
            
            # ⭐ 计算详细性能指标
            detailed_metrics = performance_metrics.calculate_metrics()
            failure_analysis = performance_metrics.analyze_failure_modes()
            
            # 计算基本统计指标
            total_entities = sum(item["entity_count"] for item in all_entities)
            
            results[config_name] = {
                "model": config["model"],
                "description": config["description"],
                "excluded_components": config["exclude"],
                "active_pipes": nlp.pipe_names,
                
                # 基本识别统计
                "load_time": load_time,
                "total_entities_found": total_entities,
                "avg_entities_per_text": total_entities / len(enhanced_test_texts),
                "entity_type_counts": entity_type_counts,
                
                # ⭐ 核心质量指标
                "detailed_metrics": detailed_metrics,
                "failure_analysis": failure_analysis,
                "category_performance": category_performance,
                
                # 样本结果（限制数量）
                "detailed_results": all_entities[:5]
            }
            
            print(f"✅ 测试完成!")
            print(f"📊 处理文本总数: {len(enhanced_test_texts)} 个")
            print(f"🎯 发现实体总数: {total_entities}")
            
            # ⭐ 显示详细质量指标
            if detailed_metrics:
                micro_f1 = detailed_metrics.get("micro_avg", {}).get("f1", 0)
                macro_f1 = detailed_metrics.get("macro_avg", {}).get("f1", 0)
                micro_precision = detailed_metrics.get("micro_avg", {}).get("precision", 0)
                micro_recall = detailed_metrics.get("micro_avg", {}).get("recall", 0)
                print(f"📊 Micro Precision: {micro_precision:.3f}")
                print(f"📊 Micro Recall: {micro_recall:.3f}")
                print(f"📊 Micro F1-Score: {micro_f1:.3f}")
                print(f"📊 Macro F1-Score: {macro_f1:.3f}")
                
                # 显示主要实体类型的性能
                main_types = ["EXCHANGE", "FUTURES_COMPANY", "PRODUCT", "CONTRACT_CODE"]
                for entity_type in main_types:
                    if entity_type in detailed_metrics:
                        metrics = detailed_metrics[entity_type]
                        print(f"  {entity_type}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
            
        except OSError as e:
            print(f"❌ 无法加载模型 {config['model']}: {e}")
            results[config_name] = None
        except Exception as e:
            print(f"❌ 测试过程出错: {e}")
            results[config_name] = None
    
    return results

# 执行增强基准测试
print("🔄 开始执行增强基准测试...")
print("⚠️  注意: 测试可能需要几分钟时间，请耐心等待...")
enhanced_benchmark_results = enhanced_benchmark_with_detailed_metrics()

# %% [markdown]
# ## 4. 增强结果分析和可视化

# %%
def analyze_enhanced_results(results):
    """分析增强的测试结果"""
    
    print(f"\n📈 增强测试结果分析")
    print("=" * 80)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ 没有有效的测试结果")
        return None, None
    
    print(f"✅ 成功测试配置: {len(valid_results)}/{len(results)}")
    
    # ⭐ 创建质量导向的性能对比DataFrame
    comparison_data = []
    detailed_metrics_data = []
    
    for config_name, result in valid_results.items():
        # 模型基本信息
        size_map = {"sm": "46MB", "md": "74MB", "lg": "575MB", "trf": "396MB"}
        model_size = next((size for key, size in size_map.items() if key in config_name), "未知")
        
        # ⭐ 获取详细性能指标
        detailed_metrics = result.get("detailed_metrics", {})
        micro_avg = detailed_metrics.get("micro_avg", {})
        macro_avg = detailed_metrics.get("macro_avg", {})
        
        basic_data = {
            "配置名称": config_name,
            "模型描述": result["description"],
            "模型大小": model_size,
            "加载时间(秒)": result['load_time'],
            "发现实体数": result['total_entities_found'],
            "平均实体数": result['avg_entities_per_text'],
            # ⭐ 核心质量指标
            "Micro_Precision": micro_avg.get("precision", 0),
            "Micro_Recall": micro_avg.get("recall", 0),
            "Micro_F1": micro_avg.get("f1", 0),
            "Macro_Precision": macro_avg.get("precision", 0),
            "Macro_Recall": macro_avg.get("recall", 0),
            "Macro_F1": macro_avg.get("f1", 0),
            "质量评分": (micro_avg.get("precision", 0) + micro_avg.get("recall", 0) + micro_avg.get("f1", 0)) * 100 / 3
        }
        comparison_data.append(basic_data)
        
        # ⭐ 实体类型详细指标
        for entity_type, metrics in detailed_metrics.items():
            if entity_type not in ["micro_avg", "macro_avg"] and isinstance(metrics, dict):
                detailed_metrics_data.append({
                    "配置名称": config_name,
                    "实体类型": entity_type,
                    "精确率": metrics.get("precision", 0),
                    "召回率": metrics.get("recall", 0),
                    "F1分数": metrics.get("f1", 0),
                    "支持度": metrics.get("support", 0),
                    "True_Positive": metrics.get("tp", 0),
                    "False_Positive": metrics.get("fp", 0),
                    "False_Negative": metrics.get("fn", 0)
                })
    
    df_basic = pd.DataFrame(comparison_data)
    df_detailed = pd.DataFrame(detailed_metrics_data)
    
    # 显示基本性能对比
    print(f"\n📊 基本性能对比表")
    print("-" * 80)
    display(df_basic.round(3))
    
    # 显示详细性能指标
    if not df_detailed.empty:
        print(f"\n📊 实体类型详细性能指标")
        print("-" * 80)
        # 只显示主要实体类型
        main_types = ["EXCHANGE", "FUTURES_COMPANY", "PRODUCT", "CONTRACT_CODE", "ORG"]
        df_main = df_detailed[df_detailed["实体类型"].isin(main_types)]
        if not df_main.empty:
            display(df_main.round(3))
    
    return df_basic, df_detailed

# ⭐ 新增：失败模式详细分析函数
def analyze_failure_modes_detailed(results):
    """详细分析失败模式"""
    
    print(f"\n🔍 失败模式详细分析")
    print("=" * 80)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    for config_name, result in valid_results.items():
        failure_analysis = result.get("failure_analysis", {})
        
        if not failure_analysis:
            continue
            
        print(f"\n📋 模型: {result['description']}")
        print("-" * 50)
        
        # 按实体类型的失败分析
        by_type = failure_analysis.get("by_type", {})
        if by_type:
            print("🎯 按实体类型的错误分布:")
            for entity_type, errors in by_type.items():
                total_errors = errors["FP"] + errors["FN"]
                if total_errors > 0:
                    print(f"  {entity_type}: FP={errors['FP']}, FN={errors['FN']}, 总错误={total_errors}")
        
        # 按文本长度的失败分析
        by_length = failure_analysis.get("by_text_length", {})
        if by_length:
            print("\n📏 按文本长度的错误分布:")
            for length_cat, errors in by_length.items():
                total_errors = errors["FP"] + errors["FN"]
                if total_errors > 0:
                    print(f"  {length_cat}: FP={errors['FP']}, FN={errors['FN']}, 总错误={total_errors}")
        
        # 显示一些具体失败案例
        detailed_cases = failure_analysis.get("detailed_cases", [])
        if detailed_cases:
            print(f"\n❌ 典型失败案例 (显示前3个):")
            for i, case in enumerate(detailed_cases[:3]):
                print(f"  {i+1}. 类型: {case['type']}, 实体类型: {case['entity_type']}")
                print(f"     文本: {case['text'][:80]}...")
                print(f"     实体: {case['entity']}")

# ⭐ 新增：专业测试语料性能分析
def analyze_professional_test_performance(results):
    """分析专业测试语料的性能"""
    
    print(f"\n🎓 专业测试语料性能分析")
    print("=" * 80)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    # 分析各模型在专业测试上的表现
    professional_performance = {}
    
    for config_name, result in valid_results.items():
        category_performance = result.get("category_performance", {})
        
        # 重点关注专业挑战性测试和RegEx失败案例
        professional_cats = ["专业挑战性测试", "RegEx失败案例"]
        
        for cat in professional_cats:
            if cat in category_performance:
                metrics = category_performance[cat]
                micro_avg = metrics.get("micro_avg", {})
                
                if config_name not in professional_performance:
                    professional_performance[config_name] = {}
                
                professional_performance[config_name][cat] = {
                    "precision": micro_avg.get("precision", 0),
                    "recall": micro_avg.get("recall", 0),
                    "f1": micro_avg.get("f1", 0)
                }
    
    # 创建专业测试性能对比表
    if professional_performance:
        prof_data = []
        for config_name, categories in professional_performance.items():
            for category, metrics in categories.items():
                prof_data.append({
                    "模型配置": config_name,
                    "测试类别": category,
                    "精确率": metrics["precision"],
                    "召回率": metrics["recall"],
                    "F1分数": metrics["f1"]
                })
        
        if prof_data:
            df_prof = pd.DataFrame(prof_data)
            print("📊 专业测试语料性能对比:")
            display(df_prof.round(3))
            
            # 分析哪个模型在专业测试上表现最好
            pivot_f1 = df_prof.pivot(index="模型配置", columns="测试类别", values="F1分数")
            if not pivot_f1.empty:
                print(f"\n🏆 专业测试F1分数排名:")
                for category in pivot_f1.columns:
                    best_model = pivot_f1[category].idxmax()
                    best_score = pivot_f1[category].max()
                    print(f"  {category}: {best_model} (F1={best_score:.3f})")

# 执行增强分析
df_basic_enhanced, df_detailed_enhanced = analyze_enhanced_results(enhanced_benchmark_results)
analyze_failure_modes_detailed(enhanced_benchmark_results)
analyze_professional_test_performance(enhanced_benchmark_results)

# %% [markdown]
# ## 5. 增强可视化图表

# %%
def create_quality_focused_visualizations(df_basic, df_detailed, results):
    """创建专注于质量的可视化图表"""
    
    if df_basic.empty:
        print("❌ 没有数据可用于可视化")
        return
    
    # 设置图表布局 - 专注于质量指标
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NER Model Quality Analysis (Accuracy-Focused)', fontsize=18, fontweight='bold')
    
    # 1. ⭐ Micro F1-Score 对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(df_basic)), df_basic['Micro_F1'], color='lightgreen', alpha=0.7)
    ax1.set_title('Micro F1-Score Comparison', fontweight='bold')
    ax1.set_ylabel('F1-Score')
    ax1.set_xticks(range(len(df_basic)))
    ax1.set_xticklabels(df_basic['配置名称'], rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. ⭐ Macro F1-Score 对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(df_basic)), df_basic['Macro_F1'], color='orange', alpha=0.7)
    ax2.set_title('Macro F1-Score Comparison', fontweight='bold')
    ax2.set_ylabel('F1-Score')
    ax2.set_xticks(range(len(df_basic)))
    ax2.set_xticklabels(df_basic['配置名称'], rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. ⭐ 精确率 vs 召回率散点图
    ax3 = axes[0, 2]
    scatter = ax3.scatter(df_basic['Micro_Precision'], df_basic['Micro_Recall'], 
                         c=df_basic['Micro_F1'], cmap='viridis', 
                         s=150, alpha=0.7, edgecolors='black')
    ax3.set_xlabel('Micro Precision')
    ax3.set_ylabel('Micro Recall')
    ax3.set_title('Precision vs Recall (colored by F1-Score)', fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # 添加对角线
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    for i, txt in enumerate(df_basic['配置名称']):
        ax3.annotate(txt, (df_basic['Micro_Precision'].iloc[i], df_basic['Micro_Recall'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter, ax=ax3, label='F1-Score')
    
    # 4. ⭐ 实体类型性能热力图
    ax4 = axes[1, 0]
    if not df_detailed.empty:
        # 创建透视表
        heatmap_data = df_detailed.pivot_table(
            index='实体类型', 
            columns='配置名称', 
            values='F1分数', 
            aggfunc='mean'
        ).fillna(0)
        
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=ax4, cbar_kws={'label': 'F1-Score'})
            ax4.set_title('F1-Score by Entity Type', fontweight='bold')
            ax4.set_xlabel('Model Configuration')
            ax4.set_ylabel('Entity Type')
    
    # 5. ⭐ 质量指标雷达图
    ax5 = axes[1, 1]
    
    # 选择质量指标进行雷达图展示
    metrics = ['Micro_F1', 'Macro_F1', 'Micro_Precision', 'Micro_Recall']
    metric_labels = ['Micro F1', 'Macro F1', 'Precision', 'Recall']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 完成圆形
    
    ax5 = plt.subplot(2, 3, 5, projection='polar')
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_basic)))
    
    for i, (_, row) in enumerate(df_basic.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]
        
        ax5.plot(angles, values, 'o-', linewidth=2, 
                label=row['配置名称'], color=colors[i])
        ax5.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(metric_labels)
    ax5.set_ylim(0, 1)
    ax5.set_title('Quality Metrics Radar Chart', fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 6. ⭐ 失败模式分析图
    ax6 = axes[1, 2]
    
    # 统计各模型的失败案例数量
    failure_counts = {}
    for config_name, result in results.items():
        if result is not None:
            failure_analysis = result.get("failure_analysis", {})
            detailed_cases = failure_analysis.get("detailed_cases", [])
            failure_counts[config_name] = len(detailed_cases)
    
    if failure_counts:
        configs = list(failure_counts.keys())
        counts = list(failure_counts.values())
        
        bars6 = ax6.bar(configs, counts, color='lightcoral', alpha=0.7)
        ax6.set_title('Failure Cases Count', fontweight='bold')
        ax6.set_ylabel('Number of Failures')
        ax6.tick_params(axis='x', rotation=45)
        
        for i, bar in enumerate(bars6):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # ⭐ 显示质量导向的Top 3 配置排名
    print(f"\n🏆 Quality-Focused Performance Ranking")
    print("=" * 60)
    
    # 按质量指标排名
    rankings = {
        "Overall F1-Score": df_basic.nlargest(3, 'Micro_F1'),
        "Precision": df_basic.nlargest(3, 'Micro_Precision'),
        "Recall": df_basic.nlargest(3, 'Micro_Recall'),
        "Macro F1": df_basic.nlargest(3, 'Macro_F1')
    }
    
    medals = ["🥇", "🥈", "🥉"]
    
    for ranking_type, top_3 in rankings.items():
        print(f"\n🎯 {ranking_type} Top 3:")
        for i, (_, row) in enumerate(top_3.iterrows()):
            if ranking_type == "Overall F1-Score":
                score = f"F1={row['Micro_F1']:.3f}"
            elif ranking_type == "Precision":
                score = f"Precision={row['Micro_Precision']:.3f}"
            elif ranking_type == "Recall":
                score = f"Recall={row['Micro_Recall']:.3f}"
            else:  # Macro F1
                score = f"Macro F1={row['Macro_F1']:.3f}"
            
            print(f"  {medals[i]} {row['模型描述']}")
            print(f"     {score}")

# 创建质量导向的可视化
if df_basic_enhanced is not None:
    create_quality_focused_visualizations(df_basic_enhanced, df_detailed_enhanced, enhanced_benchmark_results)

# %% [markdown]
# ## 6. 增强优化建议

# %%
def generate_quality_focused_recommendations(df_basic, df_detailed, results):
    """生成专注于质量的优化建议"""
    
    if df_basic.empty:
        print("❌ 没有数据可用于生成建议")
        return
    
    print(f"\n💡 质量导向的模型优化建议")
    print("=" * 80)
    
    # 分析最优配置
    best_f1_config = df_basic.loc[df_basic['Micro_F1'].idxmax()]
    best_precision_config = df_basic.loc[df_basic['Micro_Precision'].idxmax()]
    best_recall_config = df_basic.loc[df_basic['Micro_Recall'].idxmax()]
    best_macro_f1_config = df_basic.loc[df_basic['Macro_F1'].idxmax()]
    best_overall_config = df_basic.loc[df_basic['质量评分'].idxmax()]
    
    print(f"🎯 **最佳F1分数配置**: {best_f1_config['模型描述']}")
    print(f"   📊 Micro F1: {best_f1_config['Micro_F1']:.3f}")
    print(f"   📈 Precision: {best_f1_config['Micro_Precision']:.3f}")
    print(f"   📉 Recall: {best_f1_config['Micro_Recall']:.3f}")
    print(f"   🎯 发现实体: {best_f1_config['发现实体数']} 个")
    
    print(f"\n🎖️ **最佳精确率配置**: {best_precision_config['模型描述']}")
    print(f"   📈 Precision: {best_precision_config['Micro_Precision']:.3f}")
    print(f"   📊 F1: {best_precision_config['Micro_F1']:.3f}")
    print(f"   📉 Recall: {best_precision_config['Micro_Recall']:.3f}")
    
    print(f"\n🔍 **最佳召回率配置**: {best_recall_config['模型描述']}")
    print(f"   📉 Recall: {best_recall_config['Micro_Recall']:.3f}")
    print(f"   📊 F1: {best_recall_config['Micro_F1']:.3f}")
    print(f"   📈 Precision: {best_recall_config['Micro_Precision']:.3f}")
    
    print(f"\n🌟 **最佳宏平均F1配置**: {best_macro_f1_config['模型描述']}")
    print(f"   📊 Macro F1: {best_macro_f1_config['Macro_F1']:.3f}")
    print(f"   📊 Micro F1: {best_macro_f1_config['Micro_F1']:.3f}")
    
    print(f"\n🏆 **综合质量最优配置**: {best_overall_config['模型描述']}")
    print(f"   🎖️ 质量评分: {best_overall_config['质量评分']:.1f}")
    print(f"   📊 Micro F1: {best_overall_config['Micro_F1']:.3f}")
    print(f"   📈 Precision: {best_overall_config['Micro_Precision']:.3f}")
    print(f"   📉 Recall: {best_overall_config['Micro_Recall']:.3f}")
    
    # ⭐ 专业场景质量优化建议
    print(f"\n📋 **专业期货交易场景质量优化建议**:")
    print(f"=" * 60)
    
    scenarios = [
        {
            "scenario": "🎯 精确合约识别 (准确率优先)",
            "recommendation": best_precision_config['配置名称'],
            "rationale": "合约代码识别需要极高的精确率，避免误识别",
            "config": f"spacy.load('{best_precision_config['配置名称'].split('_')[0]}_core_web_{best_precision_config['配置名称'].split('_')[1]}', exclude={['parser','tagger','lemmatizer','attribute_ruler'] if 'ner_only' in best_precision_config['配置名称'] else []})",
            "use_cases": ["合约代码自动识别", "交易指令解析", "风险敞口计算"],
            "metrics": f"Precision={best_precision_config['Micro_Precision']:.3f}, F1={best_precision_config['Micro_F1']:.3f}"
        },
        {
            "scenario": "🔍 全面信息提取 (召回率优先)",
            "recommendation": best_recall_config['配置名称'],
            "rationale": "需要尽可能多地识别出所有相关实体，避免遗漏",
            "config": f"spacy.load('{best_recall_config['配置名称'].split('_')[0]}_core_web_{best_recall_config['配置名称'].split('_')[1]}', exclude={['parser','tagger','lemmatizer','attribute_ruler'] if 'ner_only' in best_recall_config['配置名称'] else []})",
            "use_cases": ["监管合规检查", "全量数据挖掘", "历史文档分析"],
            "metrics": f"Recall={best_recall_config['Micro_Recall']:.3f}, F1={best_recall_config['Micro_F1']:.3f}"
        },
        {
            "scenario": "⚖️ 平衡性能应用 (F1分数最优)",
            "recommendation": best_f1_config['配置名称'],
            "rationale": "在精确率和召回率之间找到最佳平衡点",
            "config": f"spacy.load('{best_f1_config['配置名称'].split('_')[0]}_core_web_{best_f1_config['配置名称'].split('_')[1]}', exclude={['parser','tagger','lemmatizer','attribute_ruler'] if 'ner_only' in best_f1_config['配置名称'] else []})",
            "use_cases": ["研究报告解析", "客户查询响应", "智能问答系统"],
            "metrics": f"F1={best_f1_config['Micro_F1']:.3f}, P={best_f1_config['Micro_Precision']:.3f}, R={best_f1_config['Micro_Recall']:.3f}"
        },
        {
            "scenario": "🌈 多类别均衡 (宏平均F1优先)",
            "recommendation": best_macro_f1_config['配置名称'],
            "rationale": "确保各种实体类型都有较好的识别效果",
            "config": f"spacy.load('{best_macro_f1_config['配置名称'].split('_')[0]}_core_web_{best_macro_f1_config['配置名称'].split('_')[1]}', exclude={['parser','tagger','lemmatizer','attribute_ruler'] if 'ner_only' in best_macro_f1_config['配置名称'] else []})",
            "use_cases": ["多元化信息提取", "跨类别分析", "完整性检查"],
            "metrics": f"Macro F1={best_macro_f1_config['Macro_F1']:.3f}, Micro F1={best_macro_f1_config['Micro_F1']:.3f}"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['scenario']}")
        print(f"   🔧 推荐配置: {scenario['recommendation']}")
        print(f"   💡 选择理由: {scenario['rationale']}")
        print(f"   📊 性能指标: {scenario['metrics']}")
        print(f"   💻 代码示例: {scenario['config']}")
        print(f"   📝 适用场景: {', '.join(scenario['use_cases'])}")
    
    # ⭐ 失败模式特定优化建议
    print(f"\n🛠️ **针对失败模式的质量提升策略**:")
    print(f"=" * 60)
    
    # 分析主要失败模式
    main_failure_types = set()
    for config_name, result in results.items():
        if result is not None:
            failure_analysis = result.get("failure_analysis", {})
            by_type = failure_analysis.get("by_type", {})
            main_failure_types.update(by_type.keys())
    
    optimization_strategies = {
        "CONTRACT_CODE": [
            "✅ 使用自定义规则匹配增强合约代码识别准确性",
            "✅ 构建专门的合约代码词典进行后处理验证",
            "✅ 训练针对期货合约代码的特化模型",
            "✅ 使用正则表达式预筛选候选实体，提高精确率",
            "✅ 建立合约代码格式验证机制"
        ],
        "EXCHANGE": [
            "✅ 建立交易所别名映射表，统一不同表述",
            "✅ 增加中英文混合表达的训练样本", 
            "✅ 使用基于规则的后处理纠正识别错误",
            "✅ 考虑上下文信息提高歧义消解能力",
            "✅ 维护交易所官方名称与简称对照表"
        ],
        "FUTURES_COMPANY": [
            "✅ 维护期货公司全称与简称对照表",
            "✅ 处理复杂的企业组织架构关系",
            "✅ 增强对子公司、分支机构的识别",
            "✅ 使用实体链接技术统一不同表述",
            "✅ 建立期货公司业务范围识别规则"
        ],
        "PRODUCT": [
            "✅ 区分期货品种与其他同名实体",
            "✅ 建立品种代码与中文名称映射关系",
            "✅ 处理品种名称的多种变体表达",
            "✅ 结合上下文判断实体的真实含义",
            "✅ 使用品种分类规则提高识别准确性"
        ]
    }
    
    for entity_type, strategies in optimization_strategies.items():
        if entity_type in main_failure_types:
            print(f"\n📋 {entity_type} 实体质量提升策略:")
            for strategy in strategies:
                print(f"   {strategy}")
    
    # ⭐ 质量优化代码实现
    print(f"\n💻 **质量优化代码实现建议**:")
    print(f"=" * 60)
    
#     code_examples = f"""
# # 1. 高精确率NER配置 (适用于关键业务)
# import spacy
# from typing import List, Tuple, Dict

# def setup_high_precision_ner():
#     '''设置高精确率的NER管道'''
#     # 使用精确率最高的配置
#     nlp = spacy.load("{best_precision_config['配置名称'].split('_')[0]}_core_web_{best_precision_config['配置名称'].split('_')[1]}")
    
#     # 精确率优先的后处理
#     def high_precision_postprocess(entities: List[Tuple], text: str, confidence_threshold: float = 0.8) -> List[Tuple]:
#         '''后处理：提高精确率，降低误报'''
#         validated_entities = []
        
#         for start, end, label in entities:
#             entity_text = text[start:end]
#             confidence = calculate_entity_confidence(entity_text, label)
            
#             # 只保留高置信度的实体
#             if confidence >= confidence_threshold:
#                 validated_entities.append((start, end, label))
        
#         return validated_entities
    
#     def calculate_entity_confidence(entity_text: str, label: str) -> float:
#         '''计算实体置信度'''
#         # 基于规则的置信度计算
#         confidence = 0.5  # 基础置信度
        
#         if label == "CONTRACT_CODE":
#             # 合约代码格式检查
#             import re
#             if re.match(r'^[A-Z]{{1,3}}\\d{{4}}

# # 生成质量导向的优化建议
# if df_basic_enhanced is not None:
#     generate_quality_focused_recommendations(df_basic_enhanced, df_detailed_enhanced, enhanced_benchmark_results)

# # ⭐ 保存质量分析结果
# def save_quality_focused_results(basic_results, detailed_results, benchmark_results):
#     """保存质量导向的测试结果"""
    
#     print(f"\n💾 保存质量分析结果")
#     print("=" * 60)
    
#     # 保存基本质量对比
#     if basic_results is not None and not basic_results.empty:
#         basic_results.to_csv("quality_focused_ner_comparison.csv", index=False, encoding="utf-8")
#         print(f"✅ 基本质量对比已保存到: quality_focused_ner_comparison.csv")
    
#     # 保存详细实体性能
#     if detailed_results is not None and not detailed_results.empty:
#         detailed_results.to_csv("detailed_entity_quality_metrics.csv", index=False, encoding="utf-8")
#         print(f"✅ 详细实体性能已保存到: detailed_entity_quality_metrics.csv")
    
#     # 保存原始基准测试结果
#     clean_results = {}
#     for config_name, result in benchmark_results.items():
#         if result is not None:
#             clean_result = result.copy()
#             # 移除不可序列化的对象
#             clean_result.pop("detailed_results", None)
#             # 简化失败分析数据
#             if "failure_analysis" in clean_result:
#                 failure_analysis = clean_result["failure_analysis"]
#                 if "detailed_cases" in failure_analysis:
#                     # 只保留前10个失败案例
#                     failure_analysis["detailed_cases"] = failure_analysis["detailed_cases"][:10]
#             clean_results[config_name] = clean_result
    
#     with open("quality_focused_benchmark_results.json", "w", encoding="utf-8") as f:
#         json.dump(clean_results, f, ensure_ascii=False, indent=2, default=str)
    
#     print(f"✅ 完整质量分析结果已保存到: quality_focused_benchmark_results.json")
#     print(f"📊 结果文件专注于识别质量分析，可在Excel中进一步分析")

# # 保存结果
# save_quality_focused_results(df_basic_enhanced, df_detailed_enhanced, enhanced_benchmark_results)

# # 总结
# print(f"\n🎉 **质量导向NER模型基准测试完成!**")
# print("=" * 80)
# print(f"📊 本次测试评估了 {len([r for r in enhanced_benchmark_results.values() if r is not None])} 个有效配置")
# print(f"📝 使用了 {len(enhanced_test_texts)} 个增强测试文本，包含专业期货合约代码测试")
# print(f"🎯 重点分析了精确率、召回率、F1分数等质量指标")
# print(f"🔍 提供了详细的失败模式分析和质量提升建议")
# print(f"💡 针对期货交易场景给出了专业的质量优化策略")
# print(f"\n🏆 主要发现:")
# if df_basic_enhanced is not None and not df_basic_enhanced.empty:
#     best_f1_model = df_basic_enhanced.loc[df_basic_enhanced['Micro_F1'].idxmax()]
#     best_precision_model = df_basic_enhanced.loc[df_basic_enhanced['Micro_Precision'].idxmax()]
#     best_recall_model = df_basic_enhanced.loc[df_basic_enhanced['Micro_Recall'].idxmax()]
#     print(f"   📈 最佳F1分数: {best_f1_model['模型描述']} (F1={best_f1_model['Micro_F1']:.3f})")
#     print(f"   🎯 最佳精确率: {best_precision_model['模型描述']} (Precision={best_precision_model['Micro_Precision']:.3f})")
#     print(f"   🔍 最佳召回率: {best_recall_model['模型描述']} (Recall={best_recall_model['Micro_Recall']:.3f})")
#     print(f"   🎓 专业合约代码识别能力得到重点测试和分析")
# print(f"\n💻 建议根据具体质量要求选择合适的模型配置进行部署!")
# print(f"🎯 重点关注precision/recall权衡，针对业务场景优化识别质量!")
# , entity_text):
#                 confidence += 0.4
#         elif label == "EXCHANGE":
#             # 交易所名称验证
#             exchange_keywords = ["交易所", "商所", "所"]
#             if any(keyword in entity_text for keyword in exchange_keywords):
#                 confidence += 0.3
#         elif label == "FUTURES_COMPANY":
#             # 期货公司名称验证
#             if "期货" in entity_text:
#                 confidence += 0.3
        
#         return min(confidence, 1.0)
    
#     return nlp, high_precision_postprocess

# # 2. 高召回率NER配置 (适用于信息收集)
# def setup_high_recall_ner():
#     '''设置高召回率的NER管道'''
#     # 使用召回率最高的配置
#     nlp = spacy.load("{best_recall_config['配置名称'].split('_')[0]}_core_web_{best_recall_config['配置名称'].split('_')[1]}")
    
#     # 多模式匹配增强召回率
#     from spacy.matcher import Matcher
#     matcher = Matcher(nlp.vocab)
    
#     # 添加更多匹配模式
#     patterns = {{
#         "CONTRACT_CODE": [
#             [{{"TEXT": {{"REGEX": r"[A-Z]{{1,3}}\\d{{4}}"}}}}],  # 标准合约代码
#             [{{"TEXT": {{"REGEX": r"[A-Za-z]{{1,3}}\\d{{4}}"}}}}],  # 包含小写字母
#         ],
#         "EXCHANGE_ALIAS": [
#             [{{"LOWER": "上期所"}}],
#             [{{"LOWER": "大商所"}}],
#             [{{"LOWER": "郑商所"}}],
#             [{{"LOWER": "中金所"}}],
#         ]
#     }}
    
#     for label, pattern_list in patterns.items():
#         matcher.add(label, pattern_list)
    
#     def high_recall_extract(text: str) -> List[Tuple]:
#         '''高召回率实体提取'''
#         doc = nlp(text)
#         entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        
#         # 添加规则匹配的结果
#         matches = matcher(doc)
#         for match_id, start, end in matches:
#             span = doc[start:end]
#             label = nlp.vocab.strings[match_id]
#             entities.append((span.start_char, span.end_char, label))
        
#         # 去重
#         entities = list(set(entities))
#         return sorted(entities)
    
#     return high_recall_extract

# # 3. 平衡F1分数的NER配置 (推荐用于生产环境)
# def setup_balanced_ner():
#     '''设置平衡F1分数的NER管道'''
#     # 使用F1分数最高的配置
#     nlp = spacy.load("{best_f1_config['配置名称'].split('_')[0]}_core_web_{best_f1_config['配置名称'].split('_')[1]}")
    
#     def balanced_ner_pipeline(text: str) -> Dict[str, any]:
#         '''平衡的NER管道，返回详细结果'''
#         doc = nlp(text)
        
#         entities = []
#         for ent in doc.ents:
#             entity_info = {{
#                 "text": ent.text,
#                 "label": ent.label_,
#                 "start": ent.start_char,
#                 "end": ent.end_char,
#                 "confidence": calculate_entity_confidence(ent.text, ent.label_)
#             }}
#             entities.append(entity_info)
        
#         return {{
#             "entities": entities,
#             "entity_count": len(entities),
#             "text_length": len(text),
#             "model_info": "{{}}".format("{best_f1_config['模型描述']}")
#         }}
    
#     return balanced_ner_pipeline

# # 4. 质量监控和评估函数
# def monitor_ner_quality(predictions: List[Tuple], ground_truth: List[Tuple]) -> Dict[str, float]:
#     '''监控NER质量'''
#     from sklearn.metrics import precision_recall_fscore_support
    
#     # 转换为标签序列进行评估
#     pred_labels = [label for _, _, label in predictions]
#     true_labels = [label for _, _, label in ground_truth]
    
#     if len(pred_labels) == 0 and len(true_labels) == 0:
#         return {{"precision": 1.0, "recall": 1.0, "f1": 1.0}}
    
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         true_labels, pred_labels, average='micro', zero_division=0
#     )
    
#     return {{
#         "precision": precision,
#         "recall": recall, 
#         "f1": f1,
#         "total_predictions": len(predictions),
#         "total_ground_truth": len(ground_truth)
#     }}

# # 5. 生产环境部署示例
# class ProductionNERService:
#     '''生产环境NER服务'''
    
#     def __init__(self, model_type: str = "balanced"):
#         self.model_type = model_type
#         self.quality_stats = {{
#             "total_processed": 0,
#             "total_entities": 0,
#             "avg_confidence": 0.0
#         }}
        
#         if model_type == "precision":
#             self.nlp, self.postprocess = setup_high_precision_ner()
#         elif model_type == "recall":
#             self.extract_func = setup_high_recall_ner()
#         else:  # balanced
#             self.pipeline = setup_balanced_ner()
    
#     def extract_entities(self, text: str) -> Dict[str, any]:
#         '''提取实体'''
#         if self.model_type == "balanced":
#             result = self.pipeline(text)
#         else:
#             # 其他类型的处理逻辑
#             entities = self.extract_func(text) if self.model_type == "recall" else []
#             result = {{"entities": entities}}
        
#         # 更新统计信息
#         self.quality_stats["total_processed"] += 1
#         self.quality_stats["total_entities"] += len(result["entities"])
        
#         return result
    
#     def get_quality_report(self) -> Dict[str, any]:
#         '''获取质量报告'''
#         avg_entities = (self.quality_stats["total_entities"] / 
#                        max(self.quality_stats["total_processed"], 1))
        
#         return {{
#             "model_type": self.model_type,
#             "total_processed": self.quality_stats["total_processed"],
#             "avg_entities_per_text": avg_entities,
#             "recommended_for": self._get_recommendation()
#         }}
    
#     def _get_recommendation(self) -> str:
#         '''获取使用建议'''
#         recommendations = {{
#             "precision": "关键业务场景，需要高准确率",
#             "recall": "信息收集场景，需要高覆盖率", 
#             "balanced": "一般业务场景，平衡准确率和覆盖率"
#         }}
#         return recommendations.get(self.model_type, "通用场景")

# # 使用示例
# if __name__ == "__main__":
#     # 测试文本
#     test_text = "郑商所AP2502期货怎么样了"
    
#     # 不同质量目标的处理
#     print("高精确率处理:")
#     nlp_precision, postprocess = setup_high_precision_ner()
#     doc = nlp_precision(test_text)
#     entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
#     validated_entities = postprocess(entities, test_text)
#     print(f"结果: {{validated_entities}}")
    
#     print("\\n高召回率处理:")
#     extract_recall = setup_high_recall_ner()
#     recall_entities = extract_recall(test_text)
#     print(f"结果: {{recall_entities}}")
    
#     print("\\n平衡处理:")
#     balanced_pipeline = setup_balanced_ner()
#     balanced_result = balanced_pipeline(test_text)
#     print(f"结果: {{balanced_result}}")
    
#     # 生产服务示例
#     service = ProductionNERService("balanced")
#     result = service.extract_entities(test_text)
#     quality_report = service.get_quality_report()
#     print(f"\\n生产服务结果: {{result}}")
#     print(f"质量报告: {{quality_report}}")
# """
    
#     print(code_examples)

# 生成增强优化建议
if df_basic_enhanced is not None:
    generate_enhanced_optimization_recommendations(df_basic_enhanced, df_detailed_enhanced, enhanced_benchmark_results)

# ⭐ 保存增强结果
def save_enhanced_results(basic_results, detailed_results, benchmark_results):
    """保存增强的测试结果"""
    
    print(f"\n💾 保存增强测试结果")
    print("=" * 60)
    
    # 保存基本性能对比
    if basic_results is not None and not basic_results.empty:
        basic_results.to_csv("enhanced_ner_performance_comparison.csv", index=False, encoding="utf-8")
        print(f"✅ 基本性能对比已保存到: enhanced_ner_performance_comparison.csv")
    
    # 保存详细实体性能
    if detailed_results is not None and not detailed_results.empty:
        detailed_results.to_csv("detailed_entity_performance.csv", index=False, encoding="utf-8")
        print(f"✅ 详细实体性能已保存到: detailed_entity_performance.csv")
    
    # 保存原始基准测试结果
    clean_results = {}
    for config_name, result in benchmark_results.items():
        if result is not None:
            clean_result = result.copy()
            # 移除不可序列化的对象
            clean_result.pop("detailed_results", None)
            # 简化失败分析数据
            if "failure_analysis" in clean_result:
                failure_analysis = clean_result["failure_analysis"]
                if "detailed_cases" in failure_analysis:
                    # 只保留前10个失败案例
                    failure_analysis["detailed_cases"] = failure_analysis["detailed_cases"][:10]
            clean_results[config_name] = clean_result
    
    with open("enhanced_ner_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ 完整基准测试结果已保存到: enhanced_ner_benchmark_results.json")
    print(f"📊 结果文件可在Excel或其他工具中进一步分析")

# 保存结果
save_enhanced_results(df_basic_enhanced, df_detailed_enhanced, enhanced_benchmark_results)

# 总结
print(f"\n🎉 **增强版NER模型基准测试完成!**")
print("=" * 80)
print(f"📊 本次测试评估了 {len([r for r in enhanced_benchmark_results.values() if r is not None])} 个有效配置")
print(f"📝 使用了 {len(enhanced_test_texts)} 个增强测试文本，包含专业期货合约代码测试")
print(f"🎯 新增了精确率、召回率、F1分数等详细性能指标")
print(f"🔍 提供了详细的失败模式分析和优化建议")
print(f"💡 针对期货交易场景给出了专业的模型选择和部署建议")
print(f"\n🏆 主要发现:")
if df_basic_enhanced is not None and not df_basic_enhanced.empty:
    best_f1_model = df_basic_enhanced.loc[df_basic_enhanced['Micro_F1'].idxmax()]
    fastest_model = df_basic_enhanced.loc[df_basic_enhanced['平均延迟(毫秒)'].idxmin()]
    print(f"   📈 最佳F1分数: {best_f1_model['模型描述']} (F1={best_f1_model['Micro_F1']:.3f})")
    print(f"   ⚡ 最快处理速度: {fastest_model['模型描述']} ({fastest_model['平均延迟(毫秒)']:.1f}ms)")
    print(f"   🎯 专业合约代码识别能力得到重点测试和分析")
print(f"\n💻 建议根据具体业务场景选择合适的模型配置进行部署!")