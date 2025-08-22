# -*- coding: utf-8 -*-
"""
超简化版252问题自动化测试 - 混合策略SQL比较
保留核心功能，大幅简化代码
"""

import re
import json
from pathlib import Path
from datetime import datetime


def load_all_questions():
    """加载252个测试问题"""
    print("🔍 加载测试问题...")
    
    tables_dir = Path("../test_data/tables")
    if not tables_dir.exists():
        tables_dir = Path("test_data/tables")
    
    questions = []
    question_id = 1
    
    for table_dir in tables_dir.iterdir():
        if not table_dir.is_dir():
            continue
            
        for sql_file in table_dir.glob("*sql*知识库.txt"):
            with open(sql_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单正则解析
            matches = re.findall(r'问题[:：](.*?)\n(?:--.*?\n)*((?:WITH|SELECT).*?)(?=\n\n问题[:：]|\n\n$|$)', 
                               content, re.DOTALL | re.IGNORECASE)
            
            for question_text, sql_text in matches:
                questions.append({
                    'id': question_id,
                    'question': question_text.strip(),
                    'expected_sql': sql_text.strip(),
                    'table_name': table_dir.name
                })
                question_id += 1
    
    print(f"📊 加载了 {len(questions)} 个问题")
    return questions


def simple_sql_compare(expected, actual):
    """
    简化版混合策略SQL比较
    返回: (是否匹配, 匹配类型)
    """
    if not expected or not actual:
        return False, "empty"
    
    # 简单标准化
    def normalize(sql):
        sql = re.sub(r'--.*?\n', '\n', sql)  # 去注释
        sql = re.sub(r'\s+', ' ', sql.strip().upper())  # 标准化空白和大小写
        return sql.rstrip(';')
    
    norm_expected = normalize(expected)
    norm_actual = normalize(actual)
    
    # 完全匹配
    if norm_expected == norm_actual:
        return True, "exact"
    
    # 关键词匹配（简化语义检查）
    def extract_keywords(sql):
        # 提取主要SQL组件
        keywords = []
        keywords.extend(re.findall(r'SELECT\s+(.*?)\s+FROM', sql))
        keywords.extend(re.findall(r'FROM\s+([^\s\(]+)', sql))
        keywords.extend(re.findall(r'WHERE\s+(.*?)(?:\s+GROUP|\s+ORDER|$)', sql))
        return ' '.join(keywords)
    
    expected_keywords = extract_keywords(norm_expected)
    actual_keywords = extract_keywords(norm_actual)
    
    # 简单相似度检查
    if expected_keywords and actual_keywords:
        common_words = len(set(expected_keywords.split()) & set(actual_keywords.split()))
        total_words = len(set(expected_keywords.split()) | set(actual_keywords.split()))
        similarity = common_words / total_words if total_words > 0 else 0
        
        if similarity >= 0.7:  # 70%关键词匹配
            return True, "semantic"
    
    return False, "no_match"


def extract_sql_from_response(response):
    """从响应中提取SQL"""
    if not response or not response.choices:
        return ""
    
    content = response.choices[0].message.content
    
    # 提取SQL代码块
    sql_match = re.search(r"```(?:sql)?\s*\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # 直接查找SQL语句
    sql_match = re.search(r'\b(WITH|SELECT).*', content, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(0).strip()
    
    return content.strip()


def test_single_question(question, workflow):
    """测试单个问题"""
    try:
        # 构造请求
        messages = [ChatMessage(role="user", content=question['question'])]
        request = DataQACompletionRequest(
            messages=messages,
            model="test",
            created=int(datetime.now().timestamp()),
            follow_up_num=0,
            knowledge_base_ids=["3cc33ed2-21fb-4452-9e10-528867bd5f99"],
            use_reranker=True
        )
        
        # 执行工作流
        response = workflow.do_generate(request=request, enable_follow_up=False, thinking=False)
        
        # 提取结果
        actual_sql = extract_sql_from_response(response)
        is_faq_path = response.model == "faq"
        is_match, match_type = simple_sql_compare(question['expected_sql'], actual_sql)
        
        return {
            'id': question['id'],
            'question': question['question'],
            'table': question['table_name'],
            'expected_sql': question['expected_sql'],
            'actual_sql': actual_sql,
            'is_match': is_match,
            'match_type': match_type,
            'is_faq': is_faq_path,
            'error': None
        }
        
    except Exception as e:
        return {
            'id': question['id'],
            'question': question['question'],
            'table': question['table_name'],
            'expected_sql': question['expected_sql'],
            'actual_sql': "",
            'is_match': False,
            'match_type': "error",
            'is_faq': False,
            'error': str(e)
        }


def run_all_tests(workflow, max_questions=None):
    """运行所有测试"""
    print("🚀 开始批量测试...")
    
    # 加载问题
    questions = load_all_questions()
    if max_questions:
        questions = questions[:max_questions]
    
    results = []
    
    # 逐个测试
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] 测试问题 {question['id']}: {question['question'][:40]}...")
        
        result = test_single_question(question, workflow)
        results.append(result)
        
        # 显示结果
        if result['error']:
            print(f"  ❌ 错误: {result['error']}")
        else:
            status = "✅" if result['is_match'] else "❌"
            path = "FAQ" if result['is_faq'] else "完整"
            print(f"  {status} {result['match_type']} | {path}")
        
        # 每20个显示进度
        if i % 20 == 0:
            matched = sum(1 for r in results if r['is_match'])
            print(f"  📊 当前准确率: {matched/len(results)*100:.1f}%")
    
    return results


def generate_report(results):
    """生成简化报告"""
    total = len(results)
    matched = sum(1 for r in results if r['is_match'])
    errors = sum(1 for r in results if r['error'])
    faq_count = sum(1 for r in results if r['is_faq'])
    
    # 基础统计
    stats = {
        'total': total,
        'matched': matched,
        'accuracy': f"{matched/total*100:.2f}%" if total > 0 else "0%",
        'errors': errors,
        'faq_usage': f"{faq_count/total*100:.1f}%" if total > 0 else "0%"
    }
    
    # 匹配类型统计
    match_types = {}
    for r in results:
        if r['is_match']:
            match_types[r['match_type']] = match_types.get(r['match_type'], 0) + 1
    
    print("\n" + "="*50)
    print("📊 测试结果汇总")
    print("="*50)
    print(f"总问题数: {stats['total']}")
    print(f"匹配数: {stats['matched']}")
    print(f"总体准确率: {stats['accuracy']}")
    print(f"错误数: {stats['errors']}")
    print(f"FAQ路径使用率: {stats['faq_usage']}")
    print(f"匹配类型分布: {match_types}")
    
    return stats


def save_results(results, filename="test_results_252.json"):
    """保存测试结果"""
    # 构建保存数据
    report_data = {
        'test_info': {
            'test_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_questions': len(results),
            'description': '252问题自动化测试结果'
        },
        'test_results': [
            {
                'question_id': r['id'],
                'original_question': r['question'],
                'table_name': r['table'],
                'expected_sql': r['expected_sql'],
                'generated_sql': r['actual_sql'],
                'is_match': r['is_match'],
                'match_type': r['match_type'],
                'is_faq_path': r['is_faq'],
                'error': r['error']
            }
            for r in results
        ]
    }
    
    # 保存JSON
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"📄 测试结果已保存到: {filename}")
    
    # 保存简化CSV
    csv_filename = filename.replace('.json', '.csv')
    try:
        with open(csv_filename, 'w', encoding='utf-8-sig') as f:
            f.write("问题ID,原始问题,表名,期望SQL,生成SQL,是否匹配,匹配类型,FAQ路径,错误\n")
            for r in results:
                f.write(f'{r["id"]},"{r["question"]}","{r["table"]}","{r["expected_sql"]}","{r["actual_sql"]}",{r["is_match"]},{r["match_type"]},{r["is_faq"]},"{r["error"] or ""}"\n')
        print(f"📊 CSV结果已保存到: {csv_filename}")
    except:
        print("CSV保存失败，但JSON保存成功")


def execute_full_test(workflow, max_questions=None):
    """
    执行完整测试流程
    
    Args:
        workflow: 工作流实例
        max_questions: 限制测试问题数（调试用）
    """
    try:
        print("🎯 开始252问题自动化测试（简化版）")
        
        # 运行测试
        results = run_all_tests(workflow, max_questions)
        
        # 生成报告
        stats = generate_report(results)
        
        # 保存结果
        save_results(results)
        
        # 显示不匹配样例
        mismatched = [r for r in results if not r['is_match'] and not r['error']]
        if mismatched:
            print(f"\n🔍 前5个不匹配样例:")
            for i, r in enumerate(mismatched[:5], 1):
                print(f"{i}. ID:{r['id']} - {r['question'][:50]}...")
                print(f"   匹配类型:{r['match_type']} | FAQ路径:{r['is_faq']}")
        
        print(f"\n🎉 测试完成！总体准确率: {stats['accuracy']}")
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


# 使用说明
print("✅ 超简化版自动化测试代码已加载！")
print("💡 使用方法:")
print("   execute_full_test(workflow)          # 完整测试252个问题")
print("   execute_full_test(workflow, 10)      # 调试：仅测试前10个问题")
print("📊 结果自动保存为JSON和CSV两种格式")