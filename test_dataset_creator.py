# NER测试数据集创建工具
import json
from typing import List, Dict

def create_test_dataset() -> List[Dict]:
    """
    创建NER测试数据集
    基于期货交易领域的真实查询场景
    """
    
    test_cases = [
        # === 交易所相关查询 ===
        {
            "id": 1,
            "query": "苹果期货在哪个交易所上市的？",
            "category": "exchange_query",
            "expected_entities": {
                "product": ["苹果"],
                "exchange": ["郑州商品交易所"]  # 基于PDF中的信息
            },
            "difficulty": "easy"
        },
        {
            "id": 2, 
            "query": "shfe有哪些主要期货品种？",
            "category": "exchange_query",
            "expected_entities": {
                "exchange": ["上海期货交易所"]  # shfe的全称
            },
            "difficulty": "medium"
        },
        {
            "id": 3,
            "query": "czce最新的交易规则是什么？",
            "category": "exchange_query", 
            "expected_entities": {
                "exchange": ["郑州商品交易所"]  # czce的全称
            },
            "difficulty": "medium"
        },
        {
            "id": 4,
            "query": "大连商品交易所和郑州商品交易所有什么区别？",
            "category": "exchange_comparison",
            "expected_entities": {
                "exchange": ["大连商品交易所", "郑州商品交易所"]
            },
            "difficulty": "easy"
        },
        
        # === 期货品种相关查询 ===
        {
            "id": 5,
            "query": "白糖期货最新价格是多少？",
            "category": "product_query",
            "expected_entities": {
                "product": ["白糖"]
            },
            "difficulty": "easy"
        },
        {
            "id": 6,
            "query": "苹果ap2024合约的交割规则",
            "category": "contract_query",
            "expected_entities": {
                "product": ["苹果"],
                "contract": ["ap2024"]
            },
            "difficulty": "medium"
        },
        {
            "id": 7,
            "query": "红枣jr主力合约走势如何？",
            "category": "contract_query",
            "expected_entities": {
                "product": ["红枣"],
                "contract": ["jr"]
            },
            "difficulty": "medium"
        },
        {
            "id": 8,
            "query": "棉花cf期货的保证金是多少？",
            "category": "product_query",
            "expected_entities": {
                "product": ["棉花"],
                "contract": ["cf"]
            },
            "difficulty": "medium"
        },
        
        # === 机构相关查询 ===
        {
            "id": 9,
            "query": "华泰期货的研究报告怎么样？",
            "category": "institution_query",
            "expected_entities": {
                "institution": ["华泰期货"]
            },
            "difficulty": "easy"
        },
        {
            "id": 10,
            "query": "中信期货在苹果期货上的持仓情况",
            "category": "institution_product_query",
            "expected_entities": {
                "institution": ["中信期货"],
                "product": ["苹果"]
            },
            "difficulty": "hard"
        },
        
        # === 复合查询（多实体） ===
        {
            "id": 11,
            "query": "华泰期货在郑商所的白糖持仓是多少？",
            "category": "complex_query",
            "expected_entities": {
                "institution": ["华泰期货"],
                "exchange": ["郑商所", "郑州商品交易所"], 
                "product": ["白糖"]
            },
            "difficulty": "hard"
        },
        {
            "id": 12,
            "query": "上期所cu2024合约的主要持仓机构有哪些？",
            "category": "complex_query",
            "expected_entities": {
                "exchange": ["上期所", "上海期货交易所"],
                "contract": ["cu2024"]
            },
            "difficulty": "hard"
        },
        
        # === 边界情况测试 ===
        {
            "id": 13,
            "query": "Apple公司的股价走势如何？",
            "category": "boundary_test",
            "expected_entities": {
                "company": ["Apple"]  # 应该识别为公司而非苹果期货
            },
            "difficulty": "hard",
            "note": "测试苹果公司vs苹果期货的区分"
        },
        {
            "id": 14,
            "query": "今天苹果很甜很好吃",
            "category": "boundary_test", 
            "expected_entities": {},  # 应该不识别任何金融实体
            "difficulty": "hard",
            "note": "测试日常语境中的苹果"
        },
        {
            "id": 15,
            "query": "大豆期货在DCE交易所的情况",
            "category": "abbreviation_test",
            "expected_entities": {
                "product": ["大豆"],
                "exchange": ["DCE", "大连商品交易所"]
            },
            "difficulty": "medium",
            "note": "测试交易所简称识别"
        },
        
        # === 拼写错误容错测试 ===
        {
            "id": 16,
            "query": "苹果期货在那个交易所？",  # "哪"写成"那"
            "category": "typo_test",
            "expected_entities": {
                "product": ["苹果"]
            },
            "difficulty": "medium"
        },
        {
            "id": 17,
            "query": "华太期货的持仓报告",  # "华泰"写成"华太"
            "category": "typo_test", 
            "expected_entities": {
                "institution": ["华泰期货"]  # 应该能容错识别
            },
            "difficulty": "hard"
        },
        
        # === 长文本测试 ===
        {
            "id": 18,
            "query": "请问您能告诉我一下，关于郑州商品交易所上市的苹果期货品种，华泰期货公司的最新持仓数据和分析报告在哪里可以查到吗？",
            "category": "long_text_test",
            "expected_entities": {
                "exchange": ["郑州商品交易所"],
                "product": ["苹果"],
                "institution": ["华泰期货"]
            },
            "difficulty": "medium"
        }
    ]
    
    # 扩展到100个测试用例
    extended_cases = extend_test_cases(test_cases)
    
    return extended_cases

def extend_test_cases(base_cases: List[Dict]) -> List[Dict]:
    """
    扩展基础测试用例到100个
    通过模板化和变量替换
    """
    
    # 定义替换变量
    products = ["苹果", "白糖", "棉花", "红枣", "豆粕", "菜籽油", "玻璃", "动力煤"]
    institutions = ["华泰期货", "中信期货", "永安期货", "国泰君安", "海通期货", "申银万国"]
    exchanges = {
        "郑州商品交易所": ["郑商所", "czce"],
        "上海期货交易所": ["上期所", "shfe"], 
        "大连商品交易所": ["大商所", "dce"],
        "中国金融期货交易所": ["中金所", "cffex"]
    }
    
    extended_cases = base_cases.copy()
    current_id = len(base_cases) + 1
    
    # 模板1：产品价格查询
    price_template = "{product}期货今日价格如何？"
    for product in products:
        if current_id <= 100:
            extended_cases.append({
                "id": current_id,
                "query": price_template.format(product=product),
                "category": "product_price",
                "expected_entities": {"product": [product]},
                "difficulty": "easy"
            })
            current_id += 1
    
    # 模板2：机构持仓查询
    position_template = "{institution}在{product}期货上的持仓情况"
    for institution in institutions[:3]:  # 限制数量
        for product in products[:3]:
            if current_id <= 100:
                extended_cases.append({
                    "id": current_id,
                    "query": position_template.format(institution=institution, product=product),
                    "category": "institution_position", 
                    "expected_entities": {
                        "institution": [institution],
                        "product": [product]
                    },
                    "difficulty": "medium"
                })
                current_id += 1
    
    # 模板3：交易所简称测试
    for exchange_full, abbreviations in exchanges.items():
        for abbr in abbreviations:
            if current_id <= 100:
                extended_cases.append({
                    "id": current_id,
                    "query": f"{abbr}有哪些交易品种？",
                    "category": "exchange_abbreviation",
                    "expected_entities": {
                        "exchange": [abbr, exchange_full]
                    },
                    "difficulty": "medium"
                })
                current_id += 1
    
    return extended_cases[:100]  # 确保恰好100个

def save_test_dataset(dataset: List[Dict], filename: str = "ner_test_dataset.json"):
    """保存测试数据集"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"测试数据集已保存到 {filename}")
    print(f"总计 {len(dataset)} 个测试用例")
    
    # 统计信息
    categories = {}
    difficulties = {}
    
    for case in dataset:
        cat = case.get('category', 'unknown')
        diff = case.get('difficulty', 'unknown')
        
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print("\n类别分布:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
    
    print("\n难度分布:")
    for diff, count in difficulties.items():
        print(f"  {diff}: {count}")

def load_test_dataset(filename: str = "ner_test_dataset.json") -> List[Dict]:
    """加载测试数据集"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    # 创建并保存测试数据集
    dataset = create_test_dataset()
    save_test_dataset(dataset)
    
    # 显示前几个样例
    print("\n前5个测试样例:")
    for case in dataset[:5]:
        print(f"ID: {case['id']}")
        print(f"查询: {case['query']}")
        print(f"预期实体: {case['expected_entities']}")
        print(f"难度: {case['difficulty']}")
        print("-" * 50)