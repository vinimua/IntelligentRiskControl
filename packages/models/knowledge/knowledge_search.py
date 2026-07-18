"""
Qdrant 文档/案例检索模型
"""

from pydantic import Field
from ..common.base import ContractModel

class VectorSearchRequest(ContractModel):
    """向量检索请求"""

    query: str
    model_id: str | None = None
    model_family: str | None = None
    document_types: list[str] = Field(default_factory=list)
    root_cause_codes: list[str] = Field(default_factory=list)
    method_codes: list[str] = Field(default_factory=list)
    top_k: int = 10

class VectorSearchResult(ContractModel):
    """向量检索结果 — 只用于解释和案例参考"""

    chunk_id: str
    score: float
    chunk_text: str
    document_id: str
    document_version_id: str
    section_path: str | None = None
    page_number: int | None = None
    content_hash: str
    retrieval_method: str = "DENSE"
