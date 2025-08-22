# -*- coding: utf-8 -*-
"""
简化版批量测试工具脚本
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple


def load_252_questions():
    """加载252个问题"""
    print("加载252个测试问题...")
    
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
                questions.append({
                    'id': question_id,
                    'question': question_text.strip(),
                    'expected_sql': sql_text.strip(),
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


def compare_sql(expected: str, actual: str) -> Tuple[bool, str, float]:
    """
    混合策略SQL比较
    返回: (是否匹配, 匹配类型, 分数)
    """
    if not expected or not actual:
        return False, "empty", 0.0
    
    # 标准化
    def normalize(sql):
        sql = re.sub(r'--.*?\n', '\n', sql)  # 移除注释
        sql = re.sub(r'\s+', ' ', sql.strip().upper())  # 标准化空白和大小写
        return sql.rstrip(';')
    
    norm_expected = normalize(expected)
    norm_actual = normalize(actual)
    
    # 策略1: 完全匹配
    if norm_expected == norm_actual:
        return True, "exact_match", 1.0
    
    # 策略2: 语义匹配（改进多行SQL处理）
    def get_components(sql):
        comp = {}
        # SELECT字段（处理多行格式）
        select_match = re.search(r'SELECT\s+([\s\S]*?)\s+FROM', sql, re.IGNORECASE)
        if select_match:
            fields_text = select_match.group(1)
            # 清理并分割字段，移除AS别名以便比较
            fields = []
            for field in fields_text.split(','):
                field = field.strip()
                # 移除AS别名，只保留核心字段名
                field = re.sub(r'\s+AS\s+\w+', '', field, flags=re.IGNORECASE)
                if field:
                    fields.append(field)
            comp['select'] = sorted(fields)
        
        # FROM表
        from_match = re.search(r'FROM\s+([^\s\(]+)', sql, re.IGNORECASE)
        if from_match:
            comp['from'] = from_match.group(1)
        
        # WHERE条件
        where_match = re.search(r'WHERE\s+([\s\S]*?)(?:\s+GROUP|\s+ORDER|\s+LIMIT|$)', sql, re.IGNORECASE)
        if where_match:
            comp['where'] = re.sub(r'\s+', ' ', where_match.group(1).strip())
        
        return comp
    
    exp_comp = get_components(norm_expected)
    act_comp = get_components(norm_actual)
    
    if exp_comp and act_comp:
        matches = sum(1 for k, v in exp_comp.items() if k in act_comp and act_comp[k] == v)
        score = matches / len(exp_comp) if exp_comp else 0
        if score >= 0.8:
            return True, "semantic_match", score
    
    # 策略3: 高相似度
    def similarity(s1, s2):
        if not s1 or not s2:
            return 0.0
        common = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
        return common / max(len(s1), len(s2))
    
    sim_score = similarity(norm_expected, norm_actual)
    if sim_score >= 0.95:
        return True, "high_similarity", sim_score
    
    return False, "no_match", sim_score


def save_results(results: List[Dict], filename: str = "test_results_252.json"):
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
    
    # FAQ vs 完整流程
    faq_results = [r for r in results if r['is_faq_path']]
    full_results = [r for r in results if not r['is_faq_path']]
    
    if faq_results:
        faq_matched = sum(1 for r in faq_results if r['is_match'])
        print(f"   FAQ快速路径准确率: {faq_matched/len(faq_results)*100:.2f}%")
    
    if full_results:
        full_matched = sum(1 for r in full_results if r['is_match'])
        print(f"   完整流程准确率: {full_matched/len(full_results)*100:.2f}%")

