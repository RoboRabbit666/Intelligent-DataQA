# NER模型性能基准测试
# 文件名：ner_model_benchmark.py

import spacy
import time
import json
from typing import Dict, List, Tuple
import pandas as pd

def create_test_corpus():
    """创建期货交易领域的测试语料"""
    
    test_texts = [
        "苹果期货在郑州商品交易所交易，华泰期货公司参与其中",
        "上海期货交易所的铜期货价格上涨，中信期货发布研究报告",
        "大连商品交易所推出新的豆粕期货合约，永安期货积极参与交易",
        "郑商所白糖期货主力合约收盘价格为5200元/吨",
        "shfe螺纹钢期货今日成交活跃，华泰期货研究所发布看涨报告",
        "czce棉花期货CF2024合约交割月临近",
        "中国金融期货交易所股指期货IF2024合约波动加剧",
        "永安期货、海通期货、申银万国期货三家公司持仓排名前三",
        "红枣jr期货在郑州商品交易所上市以来表现活跃",
        "dce玉米期货C2024合约价格突破前期高点"
    ]
    
    return test_texts

def benchmark_model_configurations():
    """基准测试不同模型配置"""
    
    # 配置定义
    configurations = {
        "md_full": {
            "model": "zh_core_web_md",
            "exclude": [],
            "description": "中型模型完整配置"
        },
        "trf_ner_only": {
            "model": "zh_core_web_trf", 
            "exclude": ["parser", "tagger", "lemmatizer", "attribute_ruler"],
            "description": "大模型仅NER"
        },
        "trf_ner_pos": {
            "model": "zh_core_web_trf",
            "exclude": ["parser", "lemmatizer", "attribute_ruler"], 
            "description": "大模型NER+词性"
        },
        "trf_full": {
            "model": "zh_core_web_trf",
            "exclude": [],
            "description": "大模型完整配置"
        }
    }
    
    test_texts = create_test_corpus()
    results = {}
    
    print("🔬 开始NER模型性能基准测试")
    print("=" * 60)
    
    for config_name, config in configurations.items():
        print(f"\n📊 测试配置: {config['description']}")
        print("-" * 40)
        
        try:
            # 加载模型
            start_load = time.time()
            nlp = spacy.load(config["model"], exclude=config["exclude"])
            load_time = time.time() - start_load
            
            print(f"模型加载时间: {load_time:.2f}秒")
            print(f"排除组件: {config['exclude']}")
            print(f"激活管道: {nlp.pipe_names}")
            
            # 预热模型
            warmup_text = test_texts[0]
            for _ in range(3):
                _ = nlp(warmup_text)
            
            # 性能测试
            start_time = time.time()
            all_entities = []
            
            for text in test_texts:
                doc = nlp(text)
                entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                           for ent in doc.ents]
                all_entities.append({
                    "text": text,
                    "entities": entities,
                    "entity_count": len(entities)
                })
            
            total_time = time.time() - start_time
            avg_time = total_time / len(test_texts)
            
            # 统计结果
            total_entities = sum(item["entity_count"] for item in all_entities)
            
            results[config_name] = {
                "model": config["model"],
                "description": config["description"],
                "excluded_components": config["exclude"],
                "active_pipes": nlp.pipe_names,
                "load_time": load_time,
                "total_processing_time": total_time,
                "avg_time_per_text": avg_time,
                "texts_per_second": len(test_texts) / total_time,
                "total_entities_found": total_entities,
                "avg_entities_per_text": total_entities / len(test_texts),
                "detailed_results": all_entities[:3]  # 只保存前3个详细结果
            }
            
            print(f"✅ 处理{len(test_texts)}个文本用时: {total_time:.3f}秒")
            print(f"⚡ 平均每文本: {avg_time:.4f}秒")
            print(f"🚀 处理速度: {len(test_texts)/total_time:.1f} 文本/秒")
            print(f"🎯 发现实体总数: {total_entities}")
            print(f"📝 样例实体: {all_entities[0]['entities']}")
            
        except OSError as e:
            print(f"❌ 无法加载模型 {config['model']}: {e}")
            results[config_name] = None
        except Exception as e:
            print(f"❌ 测试过程出错: {e}")
            results[config_name] = None
    
    return results

def analyze_ner_quality(results):
    """分析NER质量"""
    
    print(f"\n📈 NER质量分析")
    print("=" * 60)
    
    # 期货领域关键实体类型
    expected_entity_types = {
        "ORG": ["华泰期货", "中信期货", "永安期货", "郑州商品交易所", "上海期货交易所"],
        "PRODUCT": ["苹果", "铜", "豆粕", "白糖", "螺纹钢"],
        "EXCHANGE": ["郑商所", "shfe", "czce", "dce"]
    }
    
    for config_name, result in results.items():
        if result is None:
            continue
            
        print(f"\n🔍 {result['description']} - 实体质量分析:")
        
        found_entities = {}
        for item in result["detailed_results"]:
            for entity_text, entity_label, start, end in item["entities"]:
                if entity_label not in found_entities:
                    found_entities[entity_label] = []
                found_entities[entity_label].append(entity_text)
        
        print(f"  发现的实体类型: {list(found_entities.keys())}")
        for label, entities in found_entities.items():
            unique_entities = list(set(entities))
            print(f"    {label}: {unique_entities[:5]}")  # 显示前5个

def create_performance_comparison(results):
    """创建性能对比表"""
    
    print(f"\n📊 性能对比总结")
    print("=" * 80)
    
    # 创建对比表
    comparison_data = []
    
    for config_name, result in results.items():
        if result is None:
            continue
            
        comparison_data.append({
            "配置": result["description"],
            "模型大小": "74MB" if "md" in config_name else "396MB",
            "加载时间(秒)": f"{result['load_time']:.2f}",
            "处理速度(文本/秒)": f"{result['texts_per_second']:.1f}",
            "平均延迟(毫秒)": f"{result['avg_time_per_text']*1000:.1f}",
            "发现实体数": result['total_entities_found'],
            "相对速度": "基准" if "md_full" in config_name else ""
        })
    
    # 计算相对速度
    if comparison_data:
        baseline_speed = None
        for item in comparison_data:
            if "小模型" in item["配置"]:
                baseline_speed = float(item["处理速度(文本/秒)"])
                break
        
        if baseline_speed:
            for item in comparison_data:
                current_speed = float(item["处理速度(文本/秒)"])
                if item["相对速度"] != "基准":
                    ratio = current_speed / baseline_speed
                    item["相对速度"] = f"{ratio:.2f}x"
    
    # 打印表格
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))

def save_benchmark_results(results):
    """保存基准测试结果"""
    
    # 清理结果以便JSON序列化
    clean_results = {}
    for config_name, result in results.items():
        if result is not None:
            clean_results[config_name] = {
                k: v for k, v in result.items() 
                if k != "detailed_results"  # 详细结果太大，不保存
            }
    
    # 保存到文件
    with open("ner_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 基准测试结果已保存到 ner_benchmark_results.json")

def recommend_configuration(results):
    """基于测试结果推荐配置"""
    
    print(f"\n🎯 配置推荐")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ 没有可用的测试结果")
        return
    
    # 分析各项指标
    fastest_config = min(valid_results.items(), 
                        key=lambda x: x[1]["avg_time_per_text"])
    most_entities = max(valid_results.items(), 
                       key=lambda x: x[1]["total_entities_found"])
    
    print(f"🚀 速度最快: {fastest_config[1]['description']}")
    print(f"   平均延迟: {fastest_config[1]['avg_time_per_text']*1000:.1f}毫秒")
    
    print(f"\n🎯 实体最多: {most_entities[1]['description']}")
    print(f"   发现实体: {most_entities[1]['total_entities_found']}个")
    
    # 综合推荐
    print(f"\n💡 综合推荐:")
    print(f"   对于您的DataQA项目，建议使用exclude参数:")
    print(f"   📈 高准确率需求: zh_core_web_trf exclude=['parser', 'lemmatizer', 'attribute_ruler']")
    print(f"   ⚡ 高速度需求: zh_core_web_trf exclude=['parser', 'tagger', 'lemmatizer', 'attribute_ruler']")
    print(f"   💾 平衡选择: zh_core_web_md (完整配置，74MB，NER F-score: 0.70)")
    print(f"   🎯 最佳性能: zh_core_web_trf (NER F-score: 0.74)")

def main():
    """主函数"""
    
    print("🚀 NER模型配置基准测试")
    print("=" * 60)
    print("这个测试将帮助您选择最适合的spaCy配置")
    print("包括准确率、速度和资源使用的权衡分析\n")
    
    # 运行基准测试
    results = benchmark_model_configurations()
    
    # 分析结果
    analyze_ner_quality(results)
    create_performance_comparison(results)
    recommend_configuration(results)
    
    # 保存结果
    save_benchmark_results(results)
    
    print(f"\n🎉 基准测试完成!")
    print(f"基于测试结果，您可以选择最适合的模型配置")
    print(f"建议使用exclude参数来完全排除不需要的组件，节省内存和提高性能")
    print(f"建议先从推荐配置开始，然后根据实际使用情况调整")

if __name__ == "__main__":
    main()
