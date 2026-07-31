"""
Transactional Outbox 事件与同步状态
"""

from ..common.base import ContractModel

from datetime import datetime

from ..common.enums import SyncStatus, VectorSyncEventType

class VectorSyncOutboxEvent(ContractModel):
    """Outbox 事件 — 与 knowledge_chunks 在同一 PostgreSQL 事务提交"""

    event_id: str
    aggregate_type: str = "knowledge_chunks"
    aggregate_id: str
    event_type: VectorSyncEventType
    collection_alias: str
    embedding_model_code: str
    payload_json: dict
    status: SyncStatus = SyncStatus.PENDING
    attempts: int = 0
    max_attempts: int = 8
    next_retry_at: datetime | None = None
    locked_by: str | None = None
    locked_at: datetime | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None
    created_at: datetime | None = None
    processed_at: datetime | None = None
