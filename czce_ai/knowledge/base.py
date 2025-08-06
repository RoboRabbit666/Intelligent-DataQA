# coding=utf-8
from abc import ABC, abstractmethod

from concurrent.futures import ThreadPoolExecutor, TimeoutError, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import yaml
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    WeightedRanker,
)

from czce_ai.document import BaseChunkData, Chunk
from czce_ai.embedder import Embedder
from czce_ai.knowledge.search import SearchType
from czce_ai.nlp import NLPToolkit
from czce_ai.reranker import Reranker
from czce_ai.utils.log import logger
from resources import KNOWLEDGE_CONF


class BaseKnowledge(ABC):
    # 定义知识库类型,类属性
    collection_type: str
    config_path: Path = KNOWLEDGE_CONF

    def __init__(
        self,
        tokenizer: NLPToolkit,
        embedder: Embedder,
        uri: str,
        reranker: Optional[Reranker] = None,
        token: Optional[str] = None,
    ):
        """初始化 Milvus 管理器
        Args:
            tokenizer: 分词器实例
            embedder: 嵌入模型实例
            uri: Milvus连接URI
            reranker: 重排序模型(可选)
            token: Milvus认证token
        """
        if not hasattr(self, "collection_type") or not self.collection_type:
            raise ValueError("子类必须定义 `collection_type`")
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.reranker = reranker
        self.client = MilvusClient(uri=uri, token=token)
        self.config = self._load_config_file()
        # 验证配置
        if not self._validate_config():
            raise ValueError("配置验证失败!")

    def _load_config_file(self) -> Dict:
        try:
            with self.config_path.open("r") as f:
                all_config = yaml.safe_load(f)
            config = all_config.get(self.collection_type)
            if not config:
                logger.error(
                    f"Collection type {self.collection_type} not found in config."
                )
                raise ValueError(f"Collection type `{self.collection_type}` not found.")
            return config
        except Exception as e:
            logger.error(f"Failed to load YAML config: {e}")
            raise

    def _parse_field_config(self, field_config: Dict) -> Dict:
        """将字段配置转换为 Milvus 字段定义"""
        dtype = getattr(DataType, field_config["type"].upper())
        field_def = {
            "field_name": field_config["name"],
            "datatype": dtype,
            "description": field_config.get("description", ""),
        }
        if dtype == DataType.VARCHAR:
            field_def["max_length"] = field_config.get("max_length", 65535)
        elif dtype == DataType.FLOAT_VECTOR:
            field_def["dim"] = field_config["dim"]

        for flag in [
            "is_primary",
            "auto_id",
            "nullable",
            "default_value",
            "enable_analyzer",
        ]:
            if flag in field_config:
                field_def[flag] = field_config[flag]
        return field_def

    def _parse_function_config(self, function_config: Dict) -> Dict:
        """将字段配置转换为 Milvus 字段定义"""
        dtype = getattr(FunctionType, function_config["type"].upper())
        func_def = {
            "name": function_config.get("name"),
            "function_type": dtype,
            "input_field_names": function_config.get("input_field_names"),
            "output_field_names": function_config.get("output_field_names"),
        }
        return func_def

    def _build_schema_from_config(self, config: Dict):
        """从配置构建 Collection Schema"""
        schema = self.client.create_schema(
            enable_dynamic_field=config.get("enable_dynamic_field", False)
        )
        for field_config in config.get("fields", []):
            schema.add_field(**self._parse_field_config(field_config))
        for function_config in config.get("functions", []):
            func_def = self._parse_function_config(function_config)
            schema.add_function(Function(**func_def))
        return schema

    def _build_index_config_from_config(self, config: Dict):
        """从配置构建索引参数"""
        index_configs = config.get("indexes", [])
        index_params = self.client.prepare_index_params()
        for index_config in index_configs:
            index_params.add_index(
                field_name=index_config["field_name"],
                index_type=index_config.get("index_type", "IVF_FLAT"),
                metric_type=index_config.get("metric_type", "L2"),
                params=index_config.get("params", {}),
            )
        return index_params

    def create_collection(self, collection: str, force: bool = False):
        """从配置类型创建 Collection 如果不存在
        param collection_name: Collection 名称
        """
        has_collection = self.client.has_collection(collection_name=collection)
        if has_collection and force:
            logger.info(f"开始删除名称为{collection}的collection")
            self.client.drop_collection(collection_name=collection)
        elif has_collection:
            logger.info(f"名称为{collection}的collection已存在")
            return
        schema = self._build_schema_from_config(self.config)
        index_params = self._build_index_config_from_config(self.config)
        self.client.create_collection(
            collection,
            schema=schema,
            index_params=index_params,
            consistency_level=self.config.get("consistency_level", "Session"),
        )
        logger.info(f"Collection `{collection}` created successfully.")

    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        limit: int = 20,
        search_type: SearchType = SearchType.hybrid,
        dense_weight: float = 0.6,
        use_reranker: bool = True,
        dense_sub_weights: Optional[List[float]] = None,
        sparse_sub_weights: Optional[List[float]] = None,
        knowledge_ids: Optional[List[str]] = None,
        filters: Optional[str] = None,
        post_process: Optional[Callable[[Chunk], Chunk]] = None,
    ) -> List[Chunk]:
        """在向量数据库中检索chunk数据
        Args:
            collection (str): _description_
            query (str): 搜索的query
            search_type (SearchType): 检索类型,全文检索、向量检索和混合检索
            filters (Optional[Dict[str, Any]]): 搜索时的过滤
            limit (int, optional): _description_. Defaults to 20.
            dense_weight (float): 向量检索权重. Defaults to 0.6.
            sparse_sub_weights (List[float]): 多路全文检索权重。
            knowledge_ids (List[str], optional): 检索知识库的范围.Default to None,检索所有知识库
            filter (str, optional): TODO,与knowledge_ids整合为过滤模板,该参数暂时无效
        Returns:
            List[Chunk]: _description_
        """
        # 验证参数
        self._validate_search_weights(
            search_type, dense_sub_weights, sparse_sub_weights
        )
        # 获取输出字段
        output_fields = self.config["search_params"]["output_fields"]
        # 构建搜索请求
        if isinstance(knowledge_ids, str):
            knowledge_ids = [knowledge_ids]
        reqs = self._build_search_requests(query, search_type, limit, knowledge_ids)
        # 构建混合排序器
        dense_sub_weights = (
            dense_sub_weights or self.config["search_params"]["dense_sub_weights"]
        )
        sparse_sub_weights = (
            sparse_sub_weights or self.config["search_params"]["sparse_sub_weights"]
        )
        ranker = self._build_ranker(
            search_type, dense_weight, dense_sub_weights, sparse_sub_weights
        )
        # 执行搜索
        candidate_num = min(200, limit * 10)
        res = self.client.hybrid_search(
            collection_name=collection,
            reqs=reqs,
            ranker=ranker,
            limit=candidate_num,
            output_fields=output_fields,
        )
        # 构造结果Chunks
        chunks = [
            self._process_hit_result(hit, post_process)
            for hit in res[0][: min(max(top_k * 2, 20), len(res[0]))]
        ]
        # 重排序
        if use_reranker and self.reranker and len(chunks) > 1:
            self.reranker.rerank(query, chunks)
        
        for chunk in chunks:
            chunk.embedding_content = None
            
        return chunks[:top_k]

    def _build_search_requests(
        self, query: str, search_type: SearchType, limit: int, knowledge_ids: List[str]
    ) -> List:
        """构建搜索请求"""
        requests = []
        candidate_num = min(200, limit * 10)
        if search_type in [SearchType.hybrid, SearchType.dense]:
            requests.extend(
                self._construct_dense_req(query, candidate_num, knowledge_ids)
            )
        if search_type in [SearchType.hybrid, SearchType.sparse]:
            requests.extend(
                self._construct_sparse_req(query, candidate_num, knowledge_ids)
            )
        return requests

    def _build_ranker(
        self,
        search_type: SearchType,
        dense_weight: float = 0.6,
        dense_sub_weights: Optional[List[float]] = None,
        sparse_sub_weights: Optional[List[float]] = None,
    ) -> Optional[WeightedRanker]:
        """构建混合排序器支持多路dense/sparse权重
        Args:
            search_type: 搜索类型
            dense_weight: 总向量检索权重 hybrid模式下有效
            dense_sub_weights: 多路dense权重列表如[0.4, 0.6]
            sparse_sub_weights: 多路sparse权重列表如[0.3, 0.7]
        Returns:
            WeightedRanker实例或None
        """
        # 处理默认值
        dense_sub_weights = dense_sub_weights or [1.0]
        sparse_sub_weights = sparse_sub_weights or [1.0]
        # 标准化权重(确保总和为1)
        def normalize_weights(weights: List[float]) -> List[float]:
            total = sum(weights)
            return [w / total for w in weights] if total > 0 else weights

        dense_sub_weights = normalize_weights(dense_sub_weights)
        sparse_sub_weights = normalize_weights(sparse_sub_weights)
        
        if search_type == SearchType.dense:
            return WeightedRanker(*dense_sub_weights)
        elif search_type == SearchType.sparse:
            return WeightedRanker(*sparse_sub_weights)
        elif search_type == SearchType.hybrid:
            sparse_weight = 1 - dense_weight
            # [dense_weight * dense_sub_weights] + [sparse_weight * sparse_sub_weights]
            dense_part = [dense_weight * w for w in dense_sub_weights]
            sparse_part = [sparse_weight * w for w in sparse_sub_weights]
            return WeightedRanker(*(dense_part + sparse_part))
        
        return None

    def _process_hit_result(
        self, hit: Dict, post_process: Optional[Callable[[Chunk], Chunk]] = None
    ) -> Chunk:
        """处理单个命中结果"""
        entity = hit["entity"]
        # 构造ChunkData
        data = self._construct_chunk_data(entity)
        # 构造完整Chunk
        chunk = Chunk(
            chunk_id=entity["chunk_id"],
            data=data,
            doc_id=entity.get("doc_id"),
            knowledge_id=entity.get("knowledge_id"),
            reranking_score=float(hit["distance"]),
        )
        # 应用后处理
        return post_process(chunk) if post_process else chunk

    @abstractmethod
    def _construct_chunk_data(self, entity: Dict) -> BaseChunkData:
        """根据类型构造对应的ChunkData对象"""
        pass

    def insert(
        self,
        collection: str,
        chunks: List[Chunk],
        batch_size: int = 32,
        max_workers: int = 4,
    ) -> None:
        """插入函数
        Args:
            collection (str): 插入collection名称
            chunks (List[Chunk]): 需要插入的chunk
            batch_size (int): 每次插入的数量
        """
        self._batch_operation("insert", collection, chunks, batch_size, max_workers)

    def upsert(
        self,
        collection: str,
        chunks: List[Chunk],
        batch_size: int = 32,
        max_workers: int = 4,
    ) -> None:
        """更新函数,删除并插入
        Args:
            collection (str): 插入collection名称
            chunks (List[Chunk]): 需要插入的chunk
            batch_size (int): 每次插入的数量
        """
        self._batch_operation("upsert", collection, chunks, batch_size, max_workers)

    def _batch_operation(
        self,
        operation: str,
        collection: str,
        chunks: List[Chunk],
        batch_size: int = 32,
        max_workers: int = 5,
        timeout: float = 60,
    ):
        """insert与upsert批量操作"""
        
        def process_batch(batch_chunks: List[Chunk]) -> None:
            data = [self._standard_format(chunk) for chunk in batch_chunks]
            try:
                if operation == "insert":
                    self.client.insert(collection_name=collection, data=data)
                elif operation == "upsert":
                    self.client.upsert(collection_name=collection, data=data)
            except Exception as e:
                logger.error(f"Batch upsert or insert failed: {e}")
                raise

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_batch, chunks[i : i + batch_size])
                for i in range(0, len(chunks), batch_size)
            ]
            try:
                done, not_done = wait(futures, timeout=timeout)
                if not_done:
                    raise TimeoutError(f"{len(not_done)} batches timed out")
                
                for future in done:
                    future.result()  # 触发可能的异常
            except Exception as e:
                # 取消未完成的任务
                for f in futures:
                    f.cancel()
                logger.error(f"Bulk upsert failed: {e}")
                raise

    @abstractmethod
    def _standard_format(self, chunk: Chunk) -> Dict[str, Any]:
        """转化为schema对应的数据
        Args:
            chunk (Chunk): chunk数据
        Returns:
            Dict[str, Any]: schema格式数据
        """
        pass

    def delete_with_ids(
        self,
        collection: str,
        chunk_id: Optional[Union[str, List[str]]] = None,
        doc_id: Optional[Union[str, List[str]]] = None,
    ):
        """根据chunk_id或者doc_id删除数据
        Args:
            collection (str): collection名称
            chunk_id (Union[str, List[str]], optional): chunk_id 支持单个和多个. Defaults to None.
            doc_id (Union[str, List[str]], optional): doc_id 支持单个和多个. Defaults to None.
        """
        expr, filter_params = self._build_filter_expr_and_params(chunk_id, doc_id)
        if not expr:
            return
        self.client.delete(collection, filter=expr, params=filter_params)

    def get_by_ids(
        self,
        collection: str,
        chunk_id: Optional[Union[str, List[str]]] = None,
        doc_id: Optional[Union[str, List[str]]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> Optional[List[Chunk]]:
        """根据chunk_id或者doc_id查询
        Args:
            collection (str): 表明名称
            chunk_id (Optional[Union[str, List[str]]], optional): chunk id. Defaults to None.
            doc_id (Optional[Union[str, List[str]]], optional): doc_id. Defaults to None.
            output_fields (Optional[List[str]], optional): 召回的字段列表. Defaults to None.
        Returns:
            Optional[List[Chunk]]: _description_
        """
        if output_fields is None:
            output_fields = self.config["search_params"]["output_fields"]
        expr, filter_params = self._build_filter_expr_and_params(chunk_id, doc_id)
        if not expr:
            return None
        
        res = self.client.query(
            collection,
            filter=expr,
            output_fields=output_fields,
            params=filter_params,
        )
        # 处理查询结果
        chunks = []
        for item in res:
            try:
                # 使用_process_hit_result统一处理结果
                chunk = self._process_hit_result({"entity": item, "distance": 0})
                chunks.append(chunk)
            except Exception as e:
                logger.error(f"Failed to process chunk {item.get('chunk_id')}: {e}")
                raise
        return chunks

    def _build_filter_expr_and_params(
        self,
        chunk_id: Optional[Union[str, List[str]]] = None,
        doc_id: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[str, Dict]:
        """构建filter表达式和参数"""
        if chunk_id is None and doc_id is None:
            return "", {}
        if isinstance(chunk_id, str):
            chunk_id = [chunk_id]
        if isinstance(doc_id, str):
            doc_id = [doc_id]
            
        if chunk_id is None:
            expr = "doc_id IN {doc_id}"
            filter_params = {"doc_id": doc_id}
        elif doc_id is None:
            expr = "chunk_id IN {chunk_id}"
            filter_params = {"chunk_id": chunk_id}
        else:
            expr = "chunk_id IN {chunk_id} OR doc_id IN {doc_id}"
            filter_params = {"chunk_id": chunk_id, "doc_id": doc_id}
        return expr, filter_params

    def doc_id_exists(self, collection: str, doc_id: str) -> bool:
        """根据doc_id判断文档是否存在
        Args:
            collection (str): collection名称
            doc_id (str): 文档ID
        Returns:
            bool:
        """
        expr = f"doc_id == '{doc_id}'"
        scroll_result = self.client.query(
            collection_name=collection,
            filter=expr,
            limit=1,
        )
        return len(scroll_result) > 0

    def get_count(self, collection: str) -> int:
        """TODO
        由于Milvus问题,删除数据后统计数量不准确
        Args:
            collection (str): _description_
        Returns:
            int: _description_
        """
        return self.client.get_collection_stats(collection_name=collection)["row_count"]

    def _construct_dense_req(
        self, query: str, limit: int = 20, knowledge_ids: Optional[List[str]] = None
    ) -> List[AnnSearchRequest]:
        """功能函数,用来组装向量查询的AnnSearchRequest
        Args:
            query (str): 查询语句
            limit (int, optional): 返回数量. Defaults to 20.
            filters (str, optional): 过滤条件. Defaults to None.
        Returns:
            AnnSearchRequest: _description_
        """
        if knowledge_ids is None:
            expr = None
            filter_params = None
        else:
            expr = "knowledge_id IN {knowledge_ids}"
            filter_params = {"knowledge_ids": knowledge_ids}
            
        search_params = {"params": {"ef": 10 * 50}}
        query_dense_embedding = self.embedder.get_embedding(query)
        dense_fields = self.config["search_params"]["dense_fields"]
        dense_reqs = []
        for field in dense_fields:
            dense_reqs.append(
                AnnSearchRequest(
                    data=[query_dense_embedding],
                    anns_field=field,
                    param=search_params,
                    limit=limit,
                    expr=expr,
                    params=filter_params,
                )
            )
        return dense_reqs

    def _construct_sparse_req(
        self, query: str, limit: int = 20, knowledge_ids: Optional[List[str]] = None
    ) -> List[AnnSearchRequest]:
        """功能函数,用来组装多路全文检索的AnnSearchRequest列表
        Args:
            query (str):
            limit (int, optional): 返回数量. Defaults to 20.
            knowledge_ids (list[str], optional): 知识库ID. Defaults to None.
        Returns:
            List[AnnSearchRequest]: _description_
        """
        if knowledge_ids is None:
            expr = None
            filter_params = None
        else:
            expr = "knowledge_id IN {knowledge_ids}"
            filter_params = {"knowledge_ids": knowledge_ids}
            
        query_sparse_embedding = " ".join(
            self.tokenizer.question_parse(query, for_search=True)
        )
        search_param = {
            "metric_type": "BM25",
        }
        sparse_fields = self.config["search_params"]["sparse_fields"]
        sparse_reqs = []
        for field in sparse_fields:
            sparse_reqs.append(
                AnnSearchRequest(
                    [query_sparse_embedding],
                    field,
                    param=search_param,
                    limit=limit,
                    expr=expr,
                    params=filter_params,
                )
            )
        return sparse_reqs

    def _validate_config(self, raise_on_error: bool = True) -> bool:
        """验证配置"""
        errors = []
        # 检查必要字段
        required_sections = ["fields", "indexes", "search_params"]
        for section in required_sections:
            if section not in self.config:
                errors.append(f"缺失必要字段:{section}")
                
        # 检查search_params是否正确
        field_names = set([field.get("name") for field in self.config.get("fields", [])])
        search_params = self.config.get("search_params", {})
        if "dense_fields" in search_params:
            for field in search_params.get("dense_fields", []):
                if field not in field_names:
                    errors.append(f"dense_fields中的`{field}`未在fields中定义")
        if "sparse_fields" in search_params:
            for field in search_params.get("sparse_fields", []):
                if field not in field_names:
                    errors.append(f"sparse_fields中的`{field}`未在fields中定义")
        if "output_fields" in search_params:
            for field in search_params.get("output_fields", []):
                if field not in field_names:
                    errors.append(f"output_fields中的`{field}`未在fields中定义")
                    
        # 检查向量维度
        for field in self.config.get("fields", []):
            if (
                field.get("name") in self.config.get("search_params", {}).get("dense_fields", [])
                and field.get("dim") != self.embedder.dimensions
            ):
                errors.append(
                    f"向量维度不匹配: 配置{field.get('dim')} vs 模型{self.embedder.dimensions}"
                )
                
        # 检查权重匹配
        search_params = self.config.get("search_params", {})
        if "dense_fields" in search_params and "dense_sub_weights" in search_params:
            dense_fields_len = len(search_params["dense_fields"])
            dense_sub_weights_len = len(search_params["dense_sub_weights"])
            if dense_fields_len != dense_sub_weights_len:
                errors.append(
                    f"dense_fields和dense_sub_weights的长度不匹配: {dense_fields_len} vs {dense_sub_weights_len}"
                )
        if "sparse_fields" in search_params and "sparse_sub_weights" in search_params:
            sparse_fields_len = len(search_params["sparse_fields"])
            sparse_sub_weights_len = len(search_params["sparse_sub_weights"])
            if sparse_fields_len != sparse_sub_weights_len:
                errors.append(
                    f"sparse_fields和sparse_sub_weights的长度不匹配: {sparse_fields_len} vs {sparse_sub_weights_len}"
                )
        
        if errors and raise_on_error:
            error_msg = "配置检查失败:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return len(errors) == 0

    def _validate_search_weights(
        self,
        search_type: SearchType,
        dense_sub_weights: Optional[List[float]],
        sparse_sub_weights: Optional[List[float]],
    ) -> None:
        """验证搜索权重是否有效
        Args:
            search_type (SearchType): 搜索类型
            dense_sub_weights (Optional[List[float]]): 密集向量子权重
            sparse_sub_weights (Optional[List[float]]): 稀疏向量子权重
        """
        if not dense_sub_weights and not sparse_sub_weights:
            return
            
        search_params = self.config["search_params"]
        # 验证dense权重
        if search_type in [SearchType.hybrid, SearchType.dense]:
            dense_fields = search_params["dense_fields"]
            expected_len = len(dense_fields)
            if dense_sub_weights and len(dense_sub_weights) != expected_len:
                raise ValueError(
                    f"dense_sub_weights长度({len(dense_sub_weights)})"
                    f"与配置的dense_fields数量({expected_len})不匹配"
                )
        # 验证sparse权重
        if search_type in [SearchType.hybrid, SearchType.sparse]:
            sparse_fields = search_params["sparse_fields"]
            expected_len = len(sparse_fields)
            if sparse_sub_weights and len(sparse_sub_weights) != expected_len:
                raise ValueError(
                    f"sparse_sub_weights长度({len(sparse_sub_weights)})"
                    f"与配置的sparse_fields数量({expected_len})不匹配"
                )