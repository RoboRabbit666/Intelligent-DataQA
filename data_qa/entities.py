from enum import Enum, __all__

class WorkflowStepType(Enum):
    """DataQA 工作流步骤类型"""
    FOLLOW_UP = "follow_up"
    MODIFY_QUERY = "modify_query"
    ENTITY_RECOGNITION = "entity_recognition"
    SEMANTIC_SEARCH_FAQ = "semantic_search_faq"
    LOCATE_TABLE = "locate_table"
    GENERATE_PROMPT = "generate_prompt"
    GENERATE_SQL = "generate_sql"

__all__ = ["WorkflowStepType"]