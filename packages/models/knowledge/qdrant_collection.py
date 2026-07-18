"""
Qdrant Collection 版本登记
来源：数据库结构设计 V1.1 §11.6, Qdrant 技术方案 V1.0.1 §6
"""

from ..common.base import ContractModel

from datetime import datetime

class QdrantCollectionVersion(ContractModel):
    """物理 Collection 与逻辑 Alias"""

    collection_name: str
    alias_name: str | None = None
    embedding_model_code: str
    embedding_model_version: str
    dense_vector_dimension: int
    distance_metric: str = "Cosine"
    dense_enabled: bool = True
    sparse_enabled: bool = False
    multivector_enabled: bool = False
    payload_schema_version: int = 1
    shard_number: int = 1
    replication_factor: int = 1
    status: str = "CREATED"
    indexed_point_count: int = 0
    snapshot_uri: str | None = None
    created_at: datetime | None = None
    activated_at: datetime | None = None
