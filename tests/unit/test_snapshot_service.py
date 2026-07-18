"""SnapshotService 单元测试 — Hash 可复现性"""

from apps.modelops_api.services.snapshot_service import (
    compute_dataframe_hash,
    compute_hash,
)


class TestSnapshotHash:
    """测试快照 Hash 确定性。"""

    def test_snapshot_hash_is_reproducible(self):
        """同一数据重复计算得到相同 Hash。"""
        data = [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"},
        ]
        h1 = compute_dataframe_hash(data)
        h2 = compute_dataframe_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256

    def test_snapshot_hash_detects_changes(self):
        """不同数据产生不同 Hash。"""
        data1 = [{"a": 1}]
        data2 = [{"a": 2}]
        assert compute_dataframe_hash(data1) != compute_dataframe_hash(data2)

    def test_snapshot_hash_order_independent(self):
        """列顺序不影响 Hash — 按列名排序。"""
        data1 = [{"a": 1, "b": 2}]
        data2 = [{"b": 2, "a": 1}]
        assert compute_dataframe_hash(data1) == compute_dataframe_hash(data2)

    def test_empty_dataframe_hash(self):
        """空数据有固定 Hash。"""
        h = compute_dataframe_hash([])
        assert len(h) == 64

    def test_file_hash_deterministic(self):
        """文件级 SHA-256 也是确定性的。"""
        content = b"hello world"
        assert compute_hash(content) == compute_hash(content)
