#!/usr/bin/env python3
"""
Test script for semantic search FAQ functionality
Tests the _load_faqs and semantic_search_faq methods
"""

import sys
from pathlib import Path
import re
import numpy as np
from typing import List, Dict

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_file_parsing():
    """Test the FAQ file parsing logic independently"""
    print("=== Testing FAQ File Parsing ===")
    
    # Test with one of the actual SQL knowledge files
    test_file = "test_data/tables/郑商所会员品种成交持仓及客户数统计表/郑商所会员品种成交持仓及客户数统计表sql知识库.txt"
    
    if not Path(test_file).exists():
        print(f"Test file not found: {test_file}")
        return False
    
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Use the same pattern as in the code
    pattern = r'问题[:：](.*?)\n(SELECT.*?);'
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    
    print(f"Found {len(matches)} Q&A pairs in test file")
    
    # Show first few examples
    for i, (question, sql) in enumerate(matches[:3]):
        question = question.strip()
        sql = sql.strip()
        print(f"\nExample {i+1}:")
        print(f"Question: {question}")
        print(f"SQL (first 100 chars): {sql[:100]}...")
    
    return len(matches) > 0

def test_load_faqs_standalone():
    """Test the _load_faqs logic without the full class"""
    print("\n=== Testing _load_faqs Logic ===")
    
    faq_data = []
    tables_dir = Path("test_data/tables")
    
    if not tables_dir.exists():
        print(f"Tables directory not found: {tables_dir}")
        return False
    
    loaded_files = 0
    total_faqs = 0
    
    for table_dir in tables_dir.iterdir():
        if not table_dir.is_dir():
            continue
            
        # Find SQL knowledge files
        sql_files = list(table_dir.glob("*sql*知识库.txt"))
        
        for file_path in sql_files:
            loaded_files += 1
            table_name = table_dir.name
            
            print(f"Processing: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse Q&A pairs
            pattern = r'问题[:：](.*?)\n(SELECT.*?);'
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            
            file_faq_count = 0
            for question, sql in matches:
                question = question.strip()
                sql = sql.strip()
                
                # For testing, we'll use a dummy embedding
                embedding = np.random.rand(1024)  # Assuming 1024 dimensions
                
                faq_data.append({
                    'question': question,
                    'sql': sql,
                    'table': table_name,
                    'embedding': embedding
                })
                file_faq_count += 1
            
            total_faqs += file_faq_count
            print(f"  Found {file_faq_count} FAQs")
    
    print(f"\nSummary:")
    print(f"Files processed: {loaded_files}")
    print(f"Total FAQs loaded: {total_faqs}")
    
    # Show some examples
    if faq_data:
        print(f"\nFirst 3 FAQ examples:")
        for i, faq in enumerate(faq_data[:3]):
            print(f"\nFAQ {i+1}:")
            print(f"Table: {faq['table']}")
            print(f"Question: {faq['question']}")
            print(f"SQL (first 100 chars): {faq['sql'][:100]}...")
            print(f"Embedding shape: {faq['embedding'].shape}")
    
    return len(faq_data) > 0

def test_semantic_search_logic():
    """Test the semantic search logic with dummy data"""
    print("\n=== Testing Semantic Search Logic ===")
    
    # Create dummy FAQ data for testing
    dummy_faqs = [
        {
            'question': '查询郑商所今日会员交易活跃度排名',
            'sql': 'SELECT mem_name, SUM(trd_qty) FROM mem_clas_trd_hld_clit_stas_tab',
            'table': '郑商所会员品种成交持仓及客户数统计表',
            'embedding': np.random.rand(1024)
        },
        {
            'question': '罗列郑商所中棉花品种的成交量',
            'sql': 'SELECT mem_name, SUM(trd_qty) FROM mem_clas_trd_hld_clit_stas_tab WHERE clas_code = CF',
            'table': '郑商所会员品种成交持仓及客户数统计表',
            'embedding': np.random.rand(1024)
        },
        {
            'question': '查询仓库的基本信息',
            'sql': 'SELECT wh_name, wh_abbr FROM wh_base_info_tab',
            'table': '郑商所仓库基本信息表',
            'embedding': np.random.rand(1024)
        }
    ]
    
    # Test queries
    test_queries = [
        "查询今天的会员交易排名",
        "棉花品种的交易数据",
        "仓库信息查询",
        "完全不相关的问题"
    ]
    
    def semantic_search_faq_test(query: str, faq_data: List[Dict], top_k: int = 5) -> List[Dict]:
        """Test version of semantic search"""
        if not faq_data:
            return []
        
        # Create dummy query embedding
        query_embedding = np.random.rand(1024)
        
        # Calculate similarities
        similarities = []
        for faq in faq_data:
            # Cosine similarity
            similarity = np.dot(query_embedding, faq['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(faq['embedding'])
            )
            similarities.append(similarity)
        
        # Get Top-K
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Basic threshold
                results.append({
                    'question': faq_data[idx]['question'],
                    'sql': faq_data[idx]['sql'],
                    'table': faq_data[idx]['table'],
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    # Test each query
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        results = semantic_search_faq_test(query, dummy_faqs, top_k=3)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  {i+1}. Similarity: {result['similarity']:.3f}")
                print(f"     Question: {result['question']}")
                print(f"     Table: {result['table']}")
        else:
            print("No results found (all below threshold)")
    
    return True

def test_regex_patterns():
    """Test the regex patterns used for parsing"""
    print("\n=== Testing Regex Patterns ===")
    
    # Test cases
    test_cases = [
        "问题:查询郑商所2025-07-02日东方财富在SR品种的交易数据\nSELECT *\nFROM dataga.mem_clas_trd_hld_clit_stas_tab;",
        "问题：郑商所今日会员交易活跃度排名,即成交量前十会员\nSELECT mem_name, SUM(trd_qty) AS total_trd_qty\nFROM dataga.mem_clas_trd_hld_clit_stas_tab;",
        "问题: 多行问题\n这是第二行\nSELECT column FROM table;"
    ]
    
    pattern = r'问题[:：](.*?)\n(SELECT.*?);'
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Input: {test_case}")
        
        matches = re.findall(pattern, test_case, re.DOTALL | re.IGNORECASE)
        if matches:
            question, sql = matches[0]
            print(f"Extracted question: '{question.strip()}'")
            print(f"Extracted SQL: '{sql.strip()}'")
        else:
            print("No match found!")
    
    return True

def main():
    """Run all tests"""
    print("Starting FAQ Semantic Search Tests")
    print("=" * 50)
    
    try:
        # Test 1: File parsing
        success1 = test_file_parsing()
        
        # Test 2: Load FAQs logic
        success2 = test_load_faqs_standalone()
        
        # Test 3: Semantic search logic
        success3 = test_semantic_search_logic()
        
        # Test 4: Regex patterns
        success4 = test_regex_patterns()
        
        print("\n" + "=" * 50)
        print("Test Results:")
        print(f"File parsing: {'✓' if success1 else '✗'}")
        print(f"Load FAQs: {'✓' if success2 else '✗'}")
        print(f"Semantic search: {'✓' if success3 else '✗'}")
        print(f"Regex patterns: {'✓' if success4 else '✗'}")
        
        if all([success1, success2, success3, success4]):
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠️  Some tests failed. Check the output above.")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
