# coding: utf-8
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np
import requests
from jinja2 import Template

from czce_ai.document import Chunk
from czce_ai.reranker.base import Reranker
from czce_ai.utils.log import logger


@dataclass
class MxbaiReranker(Reranker):
    model: str = "mxbai-rerank-large"
    custom_template: str = (
        "<|im_start|>system\nYou are Owen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nquery: {{ query }}\ndocument: {{ document"
    )
    estimated_max: float = 12.0

    def prepare_inputs(self, query: str, documents: List[Chunk]) -> List[str]:
        """处理准备模型调用数据"""
        template = Template(self.custom_template)
        inputs = []
        with ThreadPoolExecutor() as executor:
            inputs = list(
                executor.map(
                    lambda doc: template.render(
                        query=query, document=doc.data.embed_rerank_content
                    ),
                    documents,
                )
            )
        return inputs

    def invoke(self, prompts: List[str]) -> List[float]:
        """调用rerank模型,返回按数据顺序的rerank分数,多线程加速HTTP请求"""
        headers = {"accept": "application/json", "Content-Type": "application/json"}

        def _process_batch(batch_prompts):
            """处理单个批次返回原始logits"""
            data = {"model": self.model, "input": batch_prompts}
            response = requests.post(self.base_url, headers=headers, json=data)
            return [item["data"] for item in response.json()["data"]]

        # 并行获取所有logits数据,分批次按顺序处理
        prompts_num = len(prompts)
        batch_indices = range(0, prompts_num, self.batch_size)
        all_logits = []
        with ThreadPoolExecutor() as executor:
            # 使用map保证结果顺序与输入顺序一致
            batch_logits = executor.map(
                _process_batch,
                [prompts[i: min(i + self.batch_size, prompts_num)] for i in batch_indices],
            )
            for logits in batch_logits:
                all_logits.extend(logits)

        logits_array = np.array(all_logits)
        return self.sigmoid_normalize(logits_array)

    def rerank(
        self, query: str, documents: List[Chunk], alpha: float = 0.3
    ) -> List[Chunk]:
        try:
            prompts = self.prepare_inputs(query, documents)
            scores = self.invoke(prompts)
            for chunk, score in zip(documents, scores):
                chunk.reranking_score = alpha * chunk.reranking_score + (1 - alpha) * (
                    max(0.3, chunk.reranking_score) * score
                )
            documents.sort(key=lambda x: x.reranking_score, reverse=True)
            return documents
        except Exception as e:
            logger.error(f"Error reranking documents: {e}.")
            raise e

    def sigmoid_normalize(self, logits: np.ndarray) -> np.ndarray:
        """使用sigmoid函数处理logits"""
        logit_gaps = logits[:, 1] - logits[:, 0]
        logit_gaps = logit_gaps - self.estimated_max / 2
        return 1 / (1 + np.exp(-logit_gaps))