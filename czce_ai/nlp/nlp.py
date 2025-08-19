import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import jieba
import spacy
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc

@dataclass
class NLPToolkit:
    """
    # 配置路径
    """
    user_dict_path: Optional[Path] = None
    syn_dict_path: Optional[Path] = None
    stop_words_path: Optional[Path] = None  # 停用词词典路径
    patterns_path: Optional[Path] = None # patterns.jsonl 路径

    _syn_dictionary: Dict[str, List[str]] = field(default_factory=dict, init=False)
    _stop_words: set = field(default_factory=set, init=False)

    def __post_init__(self):
        """
        # 初始化 jieba
        """
        jieba.initialize()
        if self.user_dict_path:
            jieba.load_userdict(str(self.user_dict_path))
        if self.syn_dict_path:
            self._syn_dictionary = json.load(self.syn_dict_path.open("r"))
        if self.stop_words_path:
            self._loadStopWord(self.stop_words_path)

        """
        # 初始化 Spacy 中文模型
        """
        self.nlp = spacy.load("zh_core_web_sm")
        """
        # 替换 tokenizer
        """
        self.nlp.tokenizer = self._create_jieba_tokenizer()
        """
        # 初始化 EntityRuler
        """
        self.ruler = self._init_entity_ruler()

    def _loadStopWord(self, file_path: Path = None) -> None:
        """加载停用词词典，包含空格

        Args:
            file_path (Path, optional): 停用词词典路径. Defaults to None.
        """
        if file_path:
            stop_words = [word.strip() for word in file_path.open("r").readlines()]
            self._stop_words = set(stop_words)

    def _create_jieba_tokenizer(self):

        def custom_tokenizer(text):
            words = list(jieba.cut(text))
            spaces = [False] * len(words)
            return Doc(self.nlp.vocab, words=words, spaces=spaces)

        return custom_tokenizer

    def _init_entity_ruler(self):
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")
        
        cfg = {
            "phrase_matcher_attr": "LOWER",  # 让字符串模式大小写不敏感（针对英文缩写/代码）
            "overwrite_ents": True           # 优先用规则识别实体（覆盖spaCy模型识别）
        }

        if "ner" in self.nlp.pipe_names:
            # 规则覆盖模型
            ruler = self.nlp.add_pipe("entity_ruler", after="ner", config=cfg)
        else:
            ruler = self.nlp.add_pipe("entity_ruler", config=cfg)

        # 从jsonl文件加载patterns
        if self.patterns_path and self.patterns_path.exists():
            ruler.from_disk(str(self.patterns_path))
        return ruler

    def recognize(self, text: str, filter_labels: Optional[List[str]] = None) -> List[Dict[str, str]]:
        doc = self.nlp(text)
        results = []
        for ent in doc.ents:
            if filter_labels is None or ent.label_ in filter_labels:
                results.append(
                    {
                        "text": ent.text,
                        "id": ent.ent_id_,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )
        return results

    def tokenize(self, text: str, for_search: bool = False, rm_sw: bool = False) -> List[str]:
        """统一分词接口

        Args:
            text: 输入文本
            for_search: 是否启用搜索引擎分词
            rm_stopwords: 是否去除停用词
        """
        text = self.str_f2h(text.lower())
        if for_search:
            seg_list = jieba.cut_for_search(text)
        else:
            seg_list = jieba.cut(text)
        tokens = list(seg_list)
        if rm_sw:
            tokens = self.remove_stopwords(tokens)
        return [w for w in tokens if w.strip()] # 清除空字符

    def question_parse(self, line: str, for_search: bool = False):
        """对提问进行分词,并构建同义词序列

        Args:
            line (str): description_
            for_search (bool, optional): _description_. Defaults to False.
        """
        seg_list = self.tokenize(line, for_search, True)

        def syn_extension_v1(self, seg_list: List[str], max_size: int = 4) -> List[List[str]]:
            """直接添加同义词

            Args:
                seg_list (List[str]): 分词后的question
                max_size (int, optional): 支持的最大同义词替换次数. Defaults to 8.

            Returns:
                List[List[str]]: _description_
            """
            result = [seg_list]
            for word in seg_list:
                for seg in self._syn_dictionary.get(word, []):
                    result.extend(self.tokenize(word))
            return result
        return syn_extension_v1(seg_list)

    def add_entities(self, entities: List[Dict[str, str]]):
        """
        entities: List of {"label": str, "pattern": str, "id": str}
        """
        self.ruler.add_patterns(entities)

    def remove_stopwords(self, seg_list: List[str]) -> List[str]:
        """去除停用词"""
        return [w for w in seg_list if w not in self._stop_words]

    def str_f2h(self, ustring: str) -> str:
        """将全角字符转换为半角字符

        Args:
            line (str): 待处理字符串
        """
        result = []
        for uchar in ustring:
            # 获取unicode编码
            code = ord(uchar)
            # 全角空格
            if code == 0x3000:
                result.append(chr(0x0020))
            # 其他全角
            elif 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            else:
                result.append(uchar)
        return "".join(result)