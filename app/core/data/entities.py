# coding=utf-8
from enum import Enum


class WorkflowStepType(str, Enum):
    """
    工作流步骤类型枚举类
    """
    FOLLOW_UP = "follow_up"
    MODIFY_QUERY = "modify_query"
    ENTITY_RECOGNITION = "entity_recognition"
    SEMANTIC_SEARCH_FAQ = "semantic_search_faq"
    LOCATE_TABLE = "locate_table"
    GENERATE_PROMPT = "generate_single_table_prompt"
    GENERATE_SQL = "generate_sql"