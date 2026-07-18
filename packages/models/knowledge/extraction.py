"""
知识抽取运行记录
来源：数据库结构设计 V1.1 §12.2
"""

from pydantic import Field
from ..common.base import ContractModel

from ..common.enums import WorkerStatus

class KnowledgeExtractionRun(ContractModel):
    """GraphRAG / 受限 LLM 抽取运行"""

    extraction_run_id: str
    idempotency_key: str
    source_document_version_ids: list[str] = Field(default_factory=list)
    extraction_model: str
    prompt_version: str
    schema_version: str
    status: WorkerStatus = WorkerStatus.PENDING
    raw_result_uri: str | None = None
    entities_found: int = 0
    relations_found: int = 0
