from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
from openai import OpenAI as OpenAIClient

from app.core.data.workflow_faq import DataWorkflow
from czce_ai.llm.message import Message as ChatMessage
from app.config.config import settings
from app.core.components import qwen3_llm, qwen3_thinking_llm

knowledge_id = "3cc33ed2-21fb-4452-9e10-528867bd5f99"
bucket_name = "czce-ai-dev"
collection_name = "hybrid_sql"

rag = DataWorkflow(
    ans_llm = qwen3_llm,
    ans_thinking_llm = qwen3_thinking_llm,
    query_llm = qwen3_llm,
    reranking_threshold = settings.rag_workflow.reranking_threshold,
    collection = collection_name
)

questions = [
    ChatMessage(role = 'user', content = '白糖的成交量是多少？')
]

knowledge_ids = [knowledge_id]
rag.query_table('白糖的成交量是多少？', knowledge_ids = knowledge_ids)
rag.do_generate(input_messages = questions, knowledge_base_ids = knowledge_ids, thinking = False)