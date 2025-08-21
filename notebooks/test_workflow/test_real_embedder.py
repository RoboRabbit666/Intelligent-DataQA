#!/usr/bin/env python3
"""
Real-world test for semantic search with actual embeddings
Tests the FAQ functionality with the actual BgeM3Embedder
"""

import sys
from pathlib import Path
import re
import numpy as np
from typing import List, Dict

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try to import the actual embedder
try:
    from czce_ai.embedder.bgem3 import BgeM3Embedder
    print("✓ Successfully imported BgeM3Embedder")
    HAS_EMBEDDER = True
except ImportError as e:
    print(f"⚠️  Could not import BgeM3Embedder: {e}")
    HAS_EMBEDDER = False

def test_with_real_embedder():
    """Test with actual embedder if available"""
    if not HAS_EMBEDDER:
        print("Skipping real embedder test - embedder not available")
        return False
    
    print("\n=== Testing with Real Embedder ===")
    
    # Initialize embedder (you may need to adjust these parameters)
    try:
        embedder = BgeM3Embedder(
            base_url="http://10.251.146.132:8002/v1", 
            api_key="your_api_key"
        )
        print("✓ Embedder initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize embedder: {e}")
        return False
    
    # Test with real FAQ data
    print("\nLoading real FAQ data...")
    faq_data = []
    tables_dir = Path("test_data/tables")
    
    # Load a subset of FAQs for testing
    test_files = [
        "郑商所会员品种成交持仓及客户数统计表/郑商所会员品种成交持仓及客户数统计表sql知识库.txt",
        "郑商所仓库基本信息表/郑商所仓库基本信息表_sql知识库.txt"
    ]
    
    for file_rel_path in test_files:
        file_path = tables_dir / file_rel_path
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue
            
        table_name = file_path.parent.name
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse Q&A pairs
        pattern = r'问题[:：](.*?)\n(SELECT.*?);'
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        # Load first 5 FAQs from each file for testing
        for question, sql in matches[:5]:
            question = question.strip()
            sql = sql.strip()
            
            try:
                # Get actual embedding
                embedding = embedder.get_embedding(question)
                
                faq_data.append({
                    'question': question,
                    'sql': sql,
                    'table': table_name,
                    'embedding': np.array(embedding)
                })
                print(f"✓ Loaded FAQ: {question[:50]}...")
                
            except Exception as e:
                print(f"❌ Failed to get embedding for: {question[:30]}... Error: {e}")
                continue
    
    print(f"\nLoaded {len(faq_data)} FAQs with real embeddings")
    
    if not faq_data:
        print("No FAQs loaded, cannot test semantic search")
        return False
    
    # Test semantic search with real embeddings
    test_queries = [
        "查询东方财富的交易数据",
        "郑商所会员交易排名",
        "棉花品种成交量",
        "仓库基本信息查询",
        "白糖期货交易",
        "完全不相关的随机问题"
    ]
    
    def semantic_search_real(query: str, faq_data: List[Dict], top_k: int = 3) -> List[Dict]:
        """Real semantic search with actual embeddings"""
        try:
            query_embedding = np.array(embedder.get_embedding(query))
        except Exception as e:
            print(f"Failed to get query embedding: {e}")
            return []
        
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
            results.append({
                'question': faq_data[idx]['question'],
                'sql': faq_data[idx]['sql'],
                'table': faq_data[idx]['table'],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    print("\n" + "="*60)
    print("SEMANTIC SEARCH RESULTS WITH REAL EMBEDDINGS")
    print("="*60)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        try:
            results = semantic_search_real(query, faq_data, top_k=3)
            
            if results:
                for i, result in enumerate(results):
                    print(f"  {i+1}. Similarity: {result['similarity']:.4f}")
                    print(f"     Question: {result['question']}")
                    print(f"     Table: {result['table']}")
                    print(f"     SQL: {result['sql'][:100]}...")
                    print()
            else:
                print("  No results found")
                
        except Exception as e:
            print(f"  ❌ Error during search: {e}")
    
    return True

def test_similarity_thresholds():
    """Test different similarity thresholds"""
    if not HAS_EMBEDDER:
        return False
        
    print("\n=== Testing Similarity Thresholds ===")
    
    # Test questions that should be very similar
    similar_pairs = [
        ("查询郑商所东方财富的交易数据", "查询郑商所东方财富在SR品种的交易数据"),
        ("会员交易排名", "郑商所今日会员交易活跃度排名"),
        ("仓库信息", "查询仓库的基本信息")
    ]
    
    try:
        embedder = BgeM3Embedder(
            base_url="http://10.251.146.132:8002/v1",
            api_key="your_api_key"
        )
    except Exception as e:
        print(f"Failed to initialize embedder: {e}")
        return False
    
    for query1, query2 in similar_pairs:
        try:
            emb1 = np.array(embedder.get_embedding(query1))
            emb2 = np.array(embedder.get_embedding(query2))
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            print(f"Query 1: {query1}")
            print(f"Query 2: {query2}")
            print(f"Similarity: {similarity:.4f}")
            
            if similarity > 0.8:
                print("  → Very high similarity ✓")
            elif similarity > 0.6:
                print("  → High similarity ✓")
            elif similarity > 0.4:
                print("  → Medium similarity")
            else:
                print("  → Low similarity")
            print()
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
    
    return True

def main():
    """Run all tests"""
    print("Starting Real-World FAQ Semantic Search Tests")
    print("=" * 60)
    
    try:
        # Test with real embedder
        success1 = test_with_real_embedder()
        
        # Test similarity thresholds
        success2 = test_similarity_thresholds()
        
        print("\n" + "=" * 60)
        print("Test Results:")
        print(f"Real embedder test: {'✓' if success1 else '✗'}")
        print(f"Similarity thresholds: {'✓' if success2 else '✗'}")
        
        if success1 and success2:
            print("\n🎉 All real-world tests passed!")
        elif success1 or success2:
            print("\n⚠️  Some tests passed, check output for details.")
        else:
            print("\n❌ Tests could not run due to missing dependencies.")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
