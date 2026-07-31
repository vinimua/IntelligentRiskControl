"""数据快照服务 — Parquet 生成 + MinIO 上传 + Hash"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile

import structlog
from minio import Minio
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..repositories.snapshot_repo import SnapshotRepo

logger = structlog.get_logger(__name__)


def compute_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_dataframe_hash(data: list[dict]) -> str:
    """对数据行列表计算确定性 SHA-256。按所有列排序后序列化。"""
    if not data:
        return hashlib.sha256(b"empty").hexdigest()

    columns = sorted(data[0].keys())
    canonical = []
    for row in data:
        ordered = tuple(row.get(col) for col in columns)
        canonical.append(ordered)

    serialized = repr(canonical).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class SnapshotService:
    """生成 Parquet 快照 → MinIO → PostgreSQL 元数据。"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = SnapshotRepo(session)
        self.minio = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    async def create_snapshot(
        self,
        records: list[dict],
        model_id: str | None = None,
        window_id: str | None = None,
        data_track: str = "NATURAL",
    ) -> dict:
        """从 dict 列表创建快照：写 Parquet → MinIO → 落库。"""
        content_hash = compute_dataframe_hash(records)
        row_count = len(records)
        column_count = len(records[0]) if records else 0

        # 检查是否已存在相同 hash 的快照
        existing = await self.repo.get_by_hash(content_hash)
        if existing:
            logger.info("snapshot_exists", hash=content_hash)
            return dict(existing)

        # 写 Parquet 到临时文件
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(records)
        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        parquet_bytes = buf.read()

        # 上传 MinIO
        file_hash = compute_hash(parquet_bytes)
        object_name = f"snapshots/{model_id or 'unknown'}/{file_hash}.parquet"

        self._ensure_bucket()
        self.minio.put_object(
            bucket_name=settings.minio_bucket,
            object_name=object_name,
            data=io.BytesIO(parquet_bytes),
            length=len(parquet_bytes),
            content_type="application/octet-stream",
        )

        storage_uri = f"s3://{settings.minio_bucket}/{object_name}"

        # 落库
        result = await self.repo.insert_snapshot(
            storage_uri=storage_uri,
            content_hash=content_hash,
            model_id=model_id,
            window_id=window_id,
            data_track=data_track,
            row_count=row_count,
            column_count=column_count,
        )
        logger.info("snapshot_created", hash=content_hash, uri=storage_uri)
        return result

    def read_snapshot(self, snapshot_id: str, storage_uri: str) -> list[dict]:
        """从 MinIO 读取 Parquet 并返回 dict 列表。"""
        import pyarrow.parquet as pq

        object_name = storage_uri.replace(f"s3://{settings.minio_bucket}/", "")
        response = self.minio.get_object(settings.minio_bucket, object_name)
        table = pq.read_table(io.BytesIO(response.read()))
        return table.to_pylist()

    def verify_snapshot_hash(self, storage_uri: str, expected_hash: str) -> bool:
        """校验 MinIO 上文件的 SHA-256。"""
        object_name = storage_uri.replace(f"s3://{settings.minio_bucket}/", "")
        response = self.minio.get_object(settings.minio_bucket, object_name)
        actual_bytes = response.read()
        actual_hash = compute_hash(actual_bytes)
        return actual_hash == expected_hash

    def _ensure_bucket(self) -> None:
        if not self.minio.bucket_exists(settings.minio_bucket):
            self.minio.make_bucket(settings.minio_bucket)
