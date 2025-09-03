# -*- coding: utf-8 -*-
"""
简化版批量测试工具脚本
"""

import re
import json
import numpy as np  # 新增导入
from pathlib import Path
from typing import List, Dict, Tuple
from app.core.data.workflow_JINGFANG import DataQaWorkflow
from app.core.components import embedder  # 新增导入


def load_faq_questions():
    """加载faq问题"""
    print("加载faq测试问题...")
    
    tables_dir = Path("../test_data/tables")
    if not tables_dir.exists():
        tables_dir = Path("test_data/tables")
    if not tables_dir.exists():
        tables_dir = Path("../../test_data/tables")
    
    if not tables_dir.exists():
        raise FileNotFoundError(f"找不到测试数据目录")
    
    questions = []
    question_id = 1
    
    for table_dir in tables_dir.iterdir():
        if not table_dir.is_dir():
            continue
        
        for sql_file in table_dir.glob("*sql*知识库.txt"):
            with open(sql_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            pattern = r'问题[:：](.*?)\n(?:--.*?\n)*((?:WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*?)(?=\n\n问题[:：]|\n\n$|$)'
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            
            for question_text, sql_text in matches:

                question_text = question_text.strip()
                sql_text = sql_text.strip()

                # 对FAQ问题进行实体识别增强
                enhanced_question_text = DataQaWorkflow.entity_recognition(self=None, query=question_text)

                questions.append({
                    'id': question_id,
                    'question': enhanced_question_text,
                    'expected_sql': sql_text,
                    'table_name': table_dir.name
                })
                question_id += 1
    
    print(f"加载了 {len(questions)} 个问题")
    return questions


def extract_sql(content: str) -> str:
    """提取SQL语句"""
    if not content:
        return ""
    
    # 提取SQL代码块（修正正则表达式错误）
    blocks = re.findall(r"```[ \t]*(?:sql)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if not blocks:
        blocks = re.findall(r"(?is)\b(?:WITH|SELECT)\b[\s\S]*?(?:;|$)", content)
    
    # 去重和清理
    seen, uniq = set(), []
    for b in (s.strip() for s in blocks):
        if b and b not in seen:
            seen.add(b)
            uniq.append(b)
    
    if uniq:
        return uniq[0]
    
    # 如果没有找到代码块，尝试直接提取SQL语句
    sql_match = re.search(r'((?:WITH|SELECT)[\s\S]*?)(?:\n\n|$)', content, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    return content.strip()


def cosine_similarity(a, b):
    """计算余弦相似度"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def compare_sql(expected: str, actual: str) -> Tuple[bool, str, float]:
    """
    基于语义搜索的SQL比较 - 使用BgeM3Embedder + 余弦相似度
    返回: (是否匹配, 匹配类型, 相似度分数)
    """
    if not expected or not actual:
        return False, "empty", 0.0
    
    # 快速完全匹配检查
    if expected.strip() == actual.strip():
        return True, "exact_match", 1.0
    
    try:
        # 标准化SQL语句
        def normalize_sql(sql):
            sql = re.sub(r'--.*?\n', '\n', sql)  # 移除注释
            sql = re.sub(r'\s+', ' ', sql.strip().upper())  # 标准化空白和大小写
            return sql.rstrip(';')
        
        norm_expected = normalize_sql(expected)
        norm_actual = normalize_sql(actual)
        
        # 标准化后的完全匹配
        if norm_expected == norm_actual:
            return True, "exact_match", 1.0
        
        # 使用BgeM3Embedder生成向量表示
        expected_embedding = np.array(embedder.get_embedding(norm_expected))
        actual_embedding = np.array(embedder.get_embedding(norm_actual))
        
        # 计算余弦相似度
        similarity_score = cosine_similarity(expected_embedding, actual_embedding)
        
        # 使用0.85阈值判断匹配
        if similarity_score >= 0.85:
            return True, "semantic_match", similarity_score
        else:
            return False, "semantic_nomatch", similarity_score
            
    except Exception as e:
        # 如果语义比较失败，返回低分数
        return False, "error", 0.0


def save_results(results: List[Dict], filename: str = "test_results.json"):
    """保存简化的测试结果"""
    simple_results = []
    for r in results:
        simple_results.append({
            'question_id': r['question_id'],
            'original_question': r['original_question'],
            'expected_sql': r['expected_sql'],
            'generated_sql': r['generated_sql'],
            'is_match': r['is_match'],
            'match_type': r['match_type'],
            'match_score': r['match_score'],
            'is_faq_path': r['is_faq_path']
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(simple_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {filename}")


def print_statistics(results: List[Dict]):
    """打印统计信息"""
    total = len(results)
    matched = sum(1 for r in results if r['is_match'])
    faq_path = sum(1 for r in results if r['is_faq_path'])
    
    print(f"\n测试统计:")
    print(f"   总问题数: {total}")
    print(f"   匹配数: {matched}")
    print(f"   总体准确率: {matched/total*100:.2f}%")
    print(f"   FAQ快速路径使用率: {faq_path/total*100:.2f}%")
    
    # 匹配类型统计
    match_types = {}
    for r in results:
        if r['is_match']:
            mt = r['match_type']
            match_types[mt] = match_types.get(mt, 0) + 1
    
    if match_types:
        print(f"   匹配类型分布:")
        for mt, count in match_types.items():
            print(f"     {mt}: {count}")
    
    # 语义相似度统计（新增）
    semantic_scores = [r['match_score'] for r in results if r['match_type'] in ['exact_match', 'semantic_match', 'semantic_nomatch']]
    if semantic_scores:
        avg_similarity = sum(semantic_scores) / len(semantic_scores)
        print(f"   平均语义相似度: {avg_similarity:.3f}")
    
    # FAQ vs 完整流程
    faq_results = [r for r in results if r['is_faq_path']]
    full_results = [r for r in results if not r['is_faq_path']]
    
    if faq_results:
        faq_matched = sum(1 for r in faq_results if r['is_match'])
        print(f"   FAQ快速路径准确率: {faq_matched/len(faq_results)*100:.2f}%")
    
    if full_results:
        full_matched = sum(1 for r in full_results if r['is_match'])
        print(f"   完整流程准确率: {full_matched/len(full_results)*100:.2f}%")