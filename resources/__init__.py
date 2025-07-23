from pathlib import Path

def get_absolute_path():
    """
    获取当前文件的绝对路径

    Returns:
        str: 当前文件的绝对路径
    """
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_absolute_path() / "resources"
USER_DICT_PATH = BASE_DIR / "user_dict.txt"
STOP_WORDS_PATH = BASE_DIR / "stopword.txt"
SYNONYM_DICT_PATH = BASE_DIR / "synonym.json"
NER_PATTERNs_PATH = BASE_DIR / "ner_patterns.jsonl"
KNOWLEDGE__CONF_PATH = BASE_DIR / "conf/knowledge_conf.yml"