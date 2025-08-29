# coding=utf-8
from enum import Enum


class WorkflowStepType(str, Enum):
    """
    工作流步骤类型枚举类
    """
    FOLLOW_UP = "follow_up"
    MODIFY_QUERY = "modify_query"
    ENTITY_RECOGNITION = "query_entity_recognition"
    LOCATE_API = "locate_api"
    SEARCH_FAQ = "search_faq"
    LOCATE_TABLE = "locate_table"
    GENERATE_SQL = "generate_sql"