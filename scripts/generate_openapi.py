"""直接从 FastAPI app 导出 OpenAPI 规范。

用法：
    python scripts/generate_openapi.py             # 生成
    python scripts/generate_openapi.py --check     # 检查是否过期（CI 用）

输出：
    contracts/generated/openapi/contracts.openapi.json
    contracts/generated/openapi/contracts.openapi.yaml

规则：
    - Pydantic 模型 + FastAPI 路由 = 唯一真相源
    - 无中间 YAML 层、无手工登记、无治理状态机
    - 前端和第三方只读这两个生成文件
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "contracts" / "generated" / "openapi"
JSON_OUT = OUT_DIR / "contracts.openapi.json"
YAML_OUT = OUT_DIR / "contracts.openapi.yaml"

# 把仓库根加入 sys.path，确保 app 可以导入
sys.path.insert(0, str(ROOT))


def generate() -> dict:
    from apps.modelops_api.main import app

    spec = app.openapi()
    spec["info"]["title"] = "RiskItem ModelOps API"
    spec["info"]["description"] = (
        "信贷风控模型智能监测与自主迭代系统 API。"
        "由 scripts/generate_openapi.py 从 FastAPI app 自动生成，不手工维护。"
    )
    return spec


def write_json(spec: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(spec, indent=2, ensure_ascii=False, default=str)
    JSON_OUT.write_text(text, encoding="utf-8")


def write_yaml(spec: dict) -> None:
    import yaml

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(spec, allow_unicode=True, sort_keys=False, default_flow_style=False)
    YAML_OUT.write_text(text, encoding="utf-8")


def check() -> bool:
    """返回 True 表示产物与当前代码一致，False 表示过期。"""
    if not JSON_OUT.exists() or not YAML_OUT.exists():
        return False

    current = generate()
    stored_json = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    return current == stored_json


def main() -> None:
    if "--check" in sys.argv:
        ok = check()
        if not ok:
            print("ERROR: OpenAPI 产物已过期，请运行 python scripts/generate_openapi.py 刷新")
            sys.exit(1)
        print("OK: OpenAPI 产物与代码一致")
        return

    spec = generate()
    write_json(spec)
    write_yaml(spec)
    print(f"OK: JSON → {JSON_OUT}")
    print(f"OK: YAML → {YAML_OUT}")


if __name__ == "__main__":
    main()
