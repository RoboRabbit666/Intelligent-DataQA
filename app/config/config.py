#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

from czce_ai.utils.log import logger


load_dotenv()


class AppSettings(BaseSettings):
    """app 整体配置"""
    host: str = Field(default="0.0.0.0", validation_alias="APP_HOST")
    port: int = Field(default=8000, validation_alias="APP_PORT")
    reload: bool = Field(default=False, validation_alias="APP_RELOAD")
    works: int = Field(default=1, validation_alias="APP_WORKS")


class RAGWorkflowSettings(BaseSettings):
    """RAG 相关配置"""
    history_round: int = Field(default=3, validation_alias="RAG_WORKFLOW_HISTORY_ROUND")
    router_prefix: str = Field(
        default="/knoq", validation_alias="RAG_WORKFLOW_ROUTER_PREFIX"
    )
    reranking_threshold: float = Field(
        default=0.2, validation_alias="RAG_WORKFLOW_RERANKING_THRESHOLD"
    )
    milvus_collection: str = Field(
        default="rag_dev",
        validation_alias="RAG_WORKFLOW_MILVUS_COLLECTION",
    )


class DataQaWorkflowSettings(BaseSettings):
    """DataQa 相关配置"""
    history_round: int = Field(default=3, validation_alias="DATA_WORKFLOW_HISTORY_ROUND")
    router_prefix: str = Field(
        default="/data", validation_alias="DATA_WORKFLOW_ROUTER_PREFIX"
    )
    reranking_threshold: float = Field(
        default=0.2, validation_alias="DATA_WORKFLOW_RERANKING_THRESHOLD"
    )
    milvus_collection: str = Field(
        default="data_dev",
        validation_alias="DATA_WORKFLOW_MILVUS_COLLECTION",
    )


class MilvusSettings(BaseSettings):
    """milvus 相关配置"""
    uri: str = Field(
        validation_alias="MILVUS_URI", default="http://10.251.146.131:19530"
    )


class EmbedderSettings(BaseSettings):
    """Embedder 相关配置"""
    base_url: str = Field(
        default="http://10.251.146.132:8002/v1", validation_alias="EMBEDDER_BASE_URL"
    )
    api_key: str = Field(default="your_api_key", validation_alias="EMBEDDER_API_KEY")


class MxbaiRerankerSettings(BaseSettings):
    """Embedder 相关配置"""
    base_url: str = Field(
        default="http://10.251.146.132:8008/pooling",
        validation_alias="MXBAI_RERANKER_BASE_URL",
    )
    api_key: str = Field(
        default="your_api_key", validation_alias="MXBAI_RERANKER_API_KEY"
    )


class QWen3MOELLMSettings(BaseSettings):
    """QWQ LLM 相关配置"""
    base_url: str = Field(
        default="http://10.251.146.132:8001/v1",
        validation_alias="Qwen3_MOE_LLM_BASE_URL",
    )
    model_name: str = Field(
        default="Qwen3-30B-A3B", validation_alias="QWEN3_MOE_LLM_MODEL"
    )
    api_key: str = Field(
        default="your_api_key", validation_alias="QWEN3_MOE_LLM_API_KEY"
    )
    temperature: float = Field(
        default=0.7, validation_alias="QWEN3_MOE_LLM_TEMPERATURE"
    )


class Qwen14BLLMSettings(BaseSettings):
    """QWQ LLM 相关配置"""
    base_url: str = Field(
        default="http://10.251.146.132:8007/v1",
        validation_alias="Qwen3_14B_LLM_BASE_URL",
    )
    model_name: str = Field(default="Qwen3-14B", validation_alias="QWEN3_14B_LLM_MODEL")
    api_key: str = Field(
        default="your_api_key", validation_alias="QWEN3_14B_LLM_API_KEY"
    )
    temperature: float = Field(
        default=0.7, validation_alias="QWEN3_14B_LLM_TEMPERATURE"
    )


class QWen3LLMSettings(BaseSettings):
    """QWen3 LLM 相关配置"""
    base_url: str = Field(
        default="http://10.251.146.132:8006/v1", validation_alias="QWEN3_LLM_BASE_URL"
    )
    model_name: str = Field(default="Qwen3-32B", validation_alias="QWEN3_LLM_MODEL")
    api_key: str = Field(default="your_api_key", validation_alias="QWEN3_LLM_API_KEY")
    temperature: float = Field(default=0.7, validation_alias="QWEN3_LLM_TEMPERATURE")


class MINIOSettings(BaseSettings):
    """MINIO 相关配置"""
    endpoint: str = Field(
        validation_alias="MINIO_ENDPOINT", default="10.251.146.131:9000"
    )
    access_key: str = Field(
        validation_alias="MINIO_ACCESS_KEY", default="XaXRgf0xK10VSgtU"
    )
    secret_key: str = Field(
        validation_alias="MINIO_SECRET_KEY", default="rVRoiNvIEZ17Vap90arQolxTtDaP5jy1"
    )


class DocInspectionSettings(BaseSettings):
    """DocInspection 相关配置"""
    local_doc_root: str = Field(
        validation_alias="DOI_LOCAL_DOC_ROOT",
        default="/host_folder/out_folder/doi_folder/",
    )
    doc_bucket_name: str = Field(
        validation_alias="DOI_DOC_BUCKET_NAME", default="bricket"
    )
    mt_workers: int = Field(default=3, validation_alias="DOI_MT_WORKERS")


class Settings(BaseSettings):
    app: AppSettings = AppSettings()
    rag_workflow: RAGWorkflowSettings = RAGWorkflowSettings()
    milvus: MilvusSettings = MilvusSettings()
    embedder: EmbedderSettings = EmbedderSettings()
    mxbai_reranker: MxbaiRerankerSettings = MxbaiRerankerSettings()
    qwen3_moe_llm: QWen3MOELLMSettings = QWen3MOELLMSettings()
    qwen3_14b_llm: Qwen14BLLMSettings = Qwen14BLLMSettings()
    qwen3_llm: QWen3LLMSettings = QWen3LLMSettings()
    minio: MINIOSettings = MINIOSettings()
    doi: DocInspectionSettings = DocInspectionSettings()
    dataqa_workflow: DataQaWorkflowSettings = DataQaWorkflowSettings()  # Added this line to include DataQaWorkflowSettings
    # Add other settings classes here as needed

    def log(self):
        logger.info(f"Settings: \n {self.model_dump_json(indent=2)}")


settings = Settings()