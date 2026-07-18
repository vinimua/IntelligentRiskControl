"""OpenAPI 生成验证 — 只验证 Pydantic → OpenAPI 链路正常工作。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_openapi.py"


def test_openapi_can_be_generated():
    """OpenAPI 从 FastAPI app 自动生成不报错（最基本健康检查）。"""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_openapi_artifacts_are_current():
    """提交的 OpenAPI 产物与当前代码一致（防漂移）。"""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, f"OpenAPI 产物过期: {result.stderr}"
