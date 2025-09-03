# -*- coding: utf-8 -*-
"""
端到端测试脚本
测试工作流的所有路径：API快速路径、FAQ快速路径、完整SQL生成
"""

import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# 导入新的工作流
from app.core.data.workflow_0901_parallel import DataQaWorkflow, WorkflowConfig
from app.core.components import qwen3_llm, qwen3_thinking_llm, embedder
from czce_ai.llm.message import Message as ChatMessage
import numpy as np


class WorkflowTester:
    """工作流测试器"""
    
    def __init__(self, thinking: bool = False):
        """初始化测试器"""
        # 使用新的并行工作流
        self.workflow = DataQaWorkflow(
            ans_llm=qwen3_llm,
            ans_thinking_llm=qwen3_thinking_llm,
            query_llm=qwen3_llm,
        )
        self.thinking = thinking
        self.results = []
        
    def load_test_cases(self) -> List[Dict]:
        """加载测试用例（FAQ问题 + API测试）"""
        test_cases = []
        
        # 1. 加载FAQ测试问题
        print("加载FAQ测试问题...")
        tables_dir = Path("test_data/tables")
        if not tables_dir.exists():
            tables_dir = Path("../test_data/tables")
        if not tables_dir.exists():
            tables_dir = Path("../../test_data/tables")
        
        if tables_dir.exists():
            for table_dir in tables_dir.iterdir():
                if not table_dir.is_dir():
                    continue
                
                for sql_file in table_dir.glob("*sql*知识库.txt"):
                    with open(sql_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    pattern = r'问题[:：](.*?)\n(?:--.*?\n)*((?:WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*?)(?=\n\n问题[:：]|\n\n$|$)'
                    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                    
                    for question, expected_sql in matches:
                        test_cases.append({
                            'type': 'faq',
                            'question': question.strip(),
                            'expected': expected_sql.strip(),
                            'table_name': table_dir.name
                        })
        
        # 2. 添加API测试用例
        print("添加API测试用例...")
        api_test_cases = [
            {
                'type': 'api',
                'question': '查询品种总仓单变动',
                'expected': '品种总仓单变动',  # API名称
            },
            {
                'type': 'api', 
                'question': '获取单边市情况统计',
                'expected': '单边市情况统计',
            },
            {
                'type': 'api',
                'question': '查询套保报表明细',
                'expected': '查询套保报表明细',
            }
        ]
        test_cases.extend(api_test_cases)
        
        print(f"加载了 {len(test_cases)} 个测试用例")
        return test_cases
    
    def test_single_case(self, test_case: Dict) -> Dict:
        """测试单个用例"""
        start_time = time.time()
        
        # 构造消息
        messages = [ChatMessage(role="user", content=test_case['question'])]
        
        # 调用工作流
        try:
            response = self.workflow.do_generate(
                input_messages=messages,
                use_reranker=True,
                knowledge_base_ids=["test_kb_id"],  # 测试用知识库ID
                thinking=self.thinking
            )
            
            # 提取结果
            result = {
                'question': test_case['question'],
                'type': test_case['type'],
                'expected': test_case['expected'],
                'actual': '',
                'model': response.model,
                'path': self._get_path(response),
                'steps': len(response.steps) if response.steps else 0,
                'time': time.time() - start_time,
                'is_match': False,
                'match_score': 0.0
            }
            
            # 提取实际结果
            if response.choices and response.choices[0].message.content:
                result['actual'] = response.choices[0].message.content
            
            # 判断匹配
            if test_case['type'] == 'api':
                # API测试：检查是否返回正确的API
                result['is_match'] = (
                    response.model == 'api_call' and 
                    test_case['expected'] in result['actual']
                )
                result['match_score'] = 1.0 if result['is_match'] else 0.0
            else:
                # FAQ/SQL测试：比较SQL
                result['is_match'], result['match_score'] = self._compare_sql(
                    test_case['expected'], 
                    self._extract_sql(result['actual'])
                )
            
            return result
            
        except Exception as e:
            return {
                'question': test_case['question'],
                'type': test_case['type'],
                'error': str(e),
                'time': time.time() - start_time,
                'is_match': False
            }
    
    def _get_path(self, response) -> str:
        """识别执行路径"""
        if response.model == 'api_call':
            return 'api_fast'
        elif response.model == 'faq_sql':
            return 'faq_fast'
        else:
            return 'full_sql'
    
    def _extract_sql(self, content: str) -> str:
        """提取SQL语句"""
        if not content:
            return ""
        
        # 提取SQL代码块
        blocks = re.findall(r"```\s*sql?\s*([\s\S]*?)```", content, re.IGNORECASE)
        if blocks:
            return blocks[0].strip()
        
        # 直接提取SQL
        sql_match = re.search(r'((?:WITH|SELECT)[\s\S]*?)(?:\n\n|$)', content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        return content.strip()
    
    def _compare_sql(self, expected: str, actual: str) -> Tuple[bool, float]:
        """比较SQL（使用向量相似度）"""
        if not expected or not actual:
            return False, 0.0
        
        # 标准化
        def normalize(sql):
            sql = re.sub(r'--.*?\n', '\n', sql)
            sql = re.sub(r'\s+', ' ', sql.strip().upper())
            return sql.rstrip(';')
        
        norm_expected = normalize(expected)
        norm_actual = normalize(actual)
        
        if norm_expected == norm_actual:
            return True, 1.0
        
        # 向量相似度比较
        try:
            exp_embedding = np.array(embedder.get_embedding(norm_expected))
            act_embedding = np.array(embedder.get_embedding(norm_actual))
            
            # 余弦相似度
            similarity = np.dot(exp_embedding, act_embedding) / (
                np.linalg.norm(exp_embedding) * np.linalg.norm(act_embedding)
            )
            
            return similarity >= 0.85, float(similarity)
        except:
            return False, 0.0
    
    def run_batch_test(self, test_cases: Optional[List[Dict]] = None, limit: Optional[int] = None):
        """批量测试"""
        if test_cases is None:
            test_cases = self.load_test_cases()
        
        if limit:
            test_cases = test_cases[:limit]
        
        print(f"\n开始批量测试 {len(test_cases)} 个用例...")
        print("-" * 80)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] 测试: {test_case['question'][:50]}...")
            result = self.test_single_case(test_case)
            self.results.append(result)
            
            # 实时反馈
            if result.get('is_match'):
                print(f"  ✓ 匹配 (路径: {result.get('path', 'unknown')}, 耗时: {result.get('time', 0):.2f}s)")
            else:
                print(f"  ✗ 不匹配 (路径: {result.get('path', 'unknown')}, 分数: {result.get('match_score', 0):.2f})")
        
        print("-" * 80)
        self._print_statistics()
    
    def test_parallel_performance(self):
        """测试并行性能"""
        print("\n测试并行执行性能...")
        print("-" * 80)
        
        test_query = "白糖期货的成交量是多少？"
        messages = [ChatMessage(role="user", content=test_query)]
        
        # 测试10次取平均
        times = []
        for i in range(10):
            start = time.time()
            response = self.workflow.do_generate(
                input_messages=messages,
                use_reranker=True,
                knowledge_base_ids=["test_kb_id"],
                thinking=False
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  运行 {i+1}: {elapsed:.3f}s (路径: {response.model})")
        
        avg_time = sum(times) / len(times)
        print(f"\n平均执行时间: {avg_time:.3f}s")
        print(f"最快: {min(times):.3f}s, 最慢: {max(times):.3f}s")
    
    def _print_statistics(self):
        """打印统计信息"""
        if not self.results:
            print("无测试结果")
            return
        
        total = len(self.results)
        matched = sum(1 for r in self.results if r.get('is_match'))
        
        # 路径统计
        paths = {}
        for r in self.results:
            path = r.get('path', 'error')
            paths[path] = paths.get(path, 0) + 1
        
        # 类型统计
        types = {}
        for r in self.results:
            t = r.get('type', 'unknown')
            types[t] = types.get(t, 0) + 1
        
        print("\n测试统计")
        print(f"  总用例数: {total}")
        print(f"  匹配数: {matched}")
        print(f"  准确率: {matched/total*100:.2f}%")
        
        print("\n  执行路径分布:")
        for path, count in paths.items():
            print(f"    {path}: {count} ({count/total*100:.1f}%)")
        
        print("\n  测试类型分布:")
        for t, count in types.items():
            type_matched = sum(1 for r in self.results if r.get('type') == t and r.get('is_match'))
            print(f"    {t}: {count} (准确率: {type_matched/count*100:.1f}%)")
        
        # 性能统计
        times = [r.get('time', 0) for r in self.results if 'time' in r]
        if times:
            print(f"\n  平均执行时间: {sum(times)/len(times):.2f}s")
            print(f"  最快: {min(times):.2f}s, 最慢: {max(times):.2f}s")
    
    def save_results(self, filename: str = None):
        """保存测试结果"""
        if not filename:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {filename}")


# ========== 快速测试函数 ==========

def quick_test():
    """快速测试主要功能"""
    tester = WorkflowTester(thinking=False)
    
    # 测试用例
    test_cases = [
        {'type': 'api', 'question': '查询品种总仓单变动', 'expected': '品种总仓单变动'},
        {'type': 'faq', 'question': '白糖的成交量是多少？', 'expected': 'SELECT ...'},
        {'type': 'sql', 'question': '分析最近的期货价格趋势', 'expected': ''},
    ]
    
    print("快速测试...")
    for case in test_cases:
        result = tester.test_single_case(case)
        print(f"  {case['type']}: {result.get('path', 'error')} - {result.get('is_match', False)}")


def full_test(limit: Optional[int] = None):
    """完整测试"""
    tester = WorkflowTester(thinking=False)
    
    # 批量测试
    tester.run_batch_test(limit=limit)
    
    # 性能测试
    tester.test_parallel_performance()
    
    # 保存结果
    tester.save_results()


if __name__ == "__main__":
    # 快速测试
    quick_test()
    
    # 完整测试（限制100个用例）
    # full_test(limit=100)
    
    # 完整测试（所有用例）
    # full_test()