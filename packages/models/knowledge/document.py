"""
文档、版本与 Chunk 模型
"""

from ..common.base import ContractModel

from datetime import datetime

from ..common.enums import SyncStatus

class KnowledgeDocument(ContractModel):
    """文档主记录 — 稳定文档身份"""

    document_id: str
    document_code: str
    name: str
    document_type: str
    language: str = "zh-CN"
    current_version_code: str | None = None
    status: str = "ACTIVE"

class KnowledgeDocumentVersion(ContractModel):
    """文档版本 — 不可覆盖，最多一个 is_active=true"""

    document_version_id: str
    document_id: str
    version_code: str
    original_file_uri: str | None = None
    normalized_content_uri: str | None = None
    tables_json_uri: str | None = None
    content_hash: str
    parse_status: str = "PENDING"
    vector_sync_status: str = "PENDING"
    extraction_status: str = "PENDING"
    is_active: bool = False
    effective_date: datetime | None = None
    expiry_date: datetime | None = None

class KnowledgeChunk(ContractModel):
    """Chunk 权威文本 — 保存在 PostgreSQL，不保存正式向量"""

    chunk_id: str
    document_version_id: str
    chunk_index: int
    section_path: str | None = None
    heading: str | None = None
    page_number: int | None = None
    chunk_text: str
    token_count: int = 0
    content_hash: str
    active: bool = True
    qdrant_point_id: str | None = None  # 固定 = chunk_id
    vector_sync_status: SyncStatus = SyncStatus.PENDING
    payload_schema_version: int | None = None
    metadata_json: dict | None = None
