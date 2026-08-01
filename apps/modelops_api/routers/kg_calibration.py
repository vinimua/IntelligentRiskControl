"""KG 权重校准管理 API。

提供校准运行、权重快照、Neo4j 同步任务、观测记录的查询与触发。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import NotFoundError, request_trace_id
from ..database import get_db

router = APIRouter(prefix="/api/kg", tags=["kg-calibration"])


@router.get("/ping")
def ping():
    return {"ok": True}


# ── Pydantic 请求/响应模型 ──

class TriggerCalibrationRequest(BaseModel):
    data_track: str = Field(default="NATURAL", pattern="^(NATURAL|SCENARIO)$")
    rule_version: str = Field(default="BETA_BINOMIAL_V2", min_length=1, max_length=200)
    weight_version: str = Field(default="KG_WEIGHT_BETA_V2", min_length=1, max_length=200)


class ApplyToNeo4jRequest(BaseModel):
    weight_version: str | None = None


# ── 统一包络 ──

def _envelope(request: Request, data, message: str = "success") -> dict:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": data,
        "trace_id": request_trace_id(request),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 校准运行
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/calibration-runs")
async def list_calibration_runs(
    request: Request,
    data_track: str | None = None,
    status: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """列出校准运行记录，支持按 data_track / status 过滤。"""
    where = ["1=1"]
    params: dict = {"limit": limit, "offset": offset}

    if data_track:
        where.append("data_track = :data_track")
        params["data_track"] = data_track.upper()
    if status:
        where.append("status = :status")
        params["status"] = status.upper()

    result = await db.execute(
        text(f"""
            SELECT * FROM knowledge.kg_calibration_runs
            WHERE {' AND '.join(where)}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = [dict(r) for r in result.mappings()]

    # 总数
    count_result = await db.execute(
        text(f"""
            SELECT COUNT(*) AS cnt FROM knowledge.kg_calibration_runs
            WHERE {' AND '.join(where)}
        """),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )
    total = count_result.scalar()

    return _envelope(request, {"items": rows, "total": total})


@router.get("/calibration-runs/{calibration_run_id}")
async def get_calibration_run(
    calibration_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """获取校准运行详情，包含快照列表和同步任务。"""
    run_result = await db.execute(
        text("SELECT * FROM knowledge.kg_calibration_runs WHERE calibration_run_id = :rid"),
        {"rid": calibration_run_id},
    )
    run = run_result.mappings().first()
    if not run:
        raise NotFoundError(f"校准运行 {calibration_run_id} 不存在")

    snapshots_result = await db.execute(
        text("""
            SELECT * FROM knowledge.kg_relation_weight_snapshots
            WHERE calibration_run_id = :rid
            ORDER BY relation_key
        """),
        {"rid": calibration_run_id},
    )
    snapshots = [dict(r) for r in snapshots_result.mappings()]

    sync_jobs_result = await db.execute(
        text("""
            SELECT * FROM knowledge.kg_sync_jobs
            WHERE calibration_run_id = :rid
            ORDER BY created_at DESC
        """),
        {"rid": calibration_run_id},
    )
    sync_jobs = [dict(r) for r in sync_jobs_result.mappings()]

    return _envelope(request, {
        "run": dict(run),
        "snapshots": snapshots,
        "sync_jobs": sync_jobs,
    })


@router.post("/calibration-runs")
async def trigger_calibration(
    request: Request,
    body: TriggerCalibrationRequest = TriggerCalibrationRequest(),
    db: AsyncSession = Depends(get_db),
):
    """触发一次 KG 权重校准。

    内部调用 Beta-Binomial V2 算法聚合观测并生成权重快照。
    同步执行并返回 calibration_run_id。
    """
    from ..services.kg_calibration_service import KgCalibrationService

    svc = KgCalibrationService(db)
    run_id = await svc.run_calibration(
        data_track=body.data_track.upper(),
        rule_version=body.rule_version,
        weight_version=body.weight_version,
    )
    await db.commit()
    return _envelope(
        request,
        {"calibration_run_id": run_id},
        "calibration completed",
    )


@router.post("/calibration-runs/{calibration_run_id}/apply-to-neo4j")
async def apply_calibration_to_neo4j(
    calibration_run_id: str,
    request: Request,
    body: ApplyToNeo4jRequest = ApplyToNeo4jRequest(),
    db: AsyncSession = Depends(get_db),
):
    """将指定校准运行的权重快照同步到 Neo4j。"""
    # 验证校准运行存在
    run_result = await db.execute(
        text("SELECT * FROM knowledge.kg_calibration_runs WHERE calibration_run_id = :rid"),
        {"rid": calibration_run_id},
    )
    if not run_result.mappings().first():
        raise NotFoundError(f"校准运行 {calibration_run_id} 不存在")

    from ..services.kg_calibration_service import KgCalibrationService

    svc = KgCalibrationService(db)
    result = await svc.apply_to_neo4j(
        calibration_run_id=calibration_run_id,
        weight_version=body.weight_version,
    )
    await db.commit()
    return _envelope(
        request,
        result,
        "weights applied to Neo4j",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Neo4j 同步任务
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/sync-jobs")
async def list_sync_jobs(
    request: Request,
    calibration_run_id: str | None = None,
    relation_type: str | None = None,
    status: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """列出 Neo4j 同步任务。"""
    where = ["1=1"]
    params: dict = {"limit": limit, "offset": offset}

    if calibration_run_id:
        where.append("calibration_run_id = :rid")
        params["rid"] = calibration_run_id
    if relation_type:
        where.append("relation_type = :rtype")
        params["rtype"] = relation_type.upper()
    if status:
        where.append("status = :st")
        params["st"] = status.upper()

    result = await db.execute(
        text(f"""
            SELECT * FROM knowledge.kg_sync_jobs
            WHERE {' AND '.join(where)}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = [dict(r) for r in result.mappings()]

    count_result = await db.execute(
        text(f"""
            SELECT COUNT(*) AS cnt FROM knowledge.kg_sync_jobs
            WHERE {' AND '.join(where)}
        """),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )
    total = count_result.scalar()

    return _envelope(request, {"items": rows, "total": total})


# ═══════════════════════════════════════════════════════════════════════════════
# 观测记录
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/observations")
async def list_observations(
    request: Request,
    lifecycle_run_id: str | None = None,
    relation_key: str | None = None,
    direction: str | None = None,
    data_track: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """列出 KG 观测记录。"""
    where = ["1=1"]
    params: dict = {"limit": limit, "offset": offset}

    if lifecycle_run_id:
        where.append("lifecycle_run_id = :lrid")
        params["lrid"] = lifecycle_run_id
    if relation_key:
        where.append("relation_key = :rk")
        params["rk"] = relation_key
    if direction:
        where.append("direction = :dir")
        params["dir"] = direction.upper()
    if data_track:
        where.append("data_track = :track")
        params["track"] = data_track.upper()

    result = await db.execute(
        text(f"""
            SELECT * FROM knowledge.kg_relation_observations
            WHERE {' AND '.join(where)}
            ORDER BY observed_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = [dict(r) for r in result.mappings()]

    count_result = await db.execute(
        text(f"""
            SELECT COUNT(*) AS cnt FROM knowledge.kg_relation_observations
            WHERE {' AND '.join(where)}
        """),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )
    total = count_result.scalar()

    return _envelope(request, {"items": rows, "total": total})


# ═══════════════════════════════════════════════════════════════════════════════
# 权重快照与趋势
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/weight-snapshots")
async def list_weight_snapshots(
    request: Request,
    relation_key: str | None = None,
    weight_version: str | None = None,
    applied_to_neo4j: bool | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """列出权重快照。"""
    where = ["1=1"]
    params: dict = {"limit": limit, "offset": offset}

    if relation_key:
        where.append("relation_key = :rk")
        params["rk"] = relation_key
    if weight_version:
        where.append("weight_version = :wv")
        params["wv"] = weight_version
    if applied_to_neo4j is not None:
        where.append("applied_to_neo4j = :applied")
        params["applied"] = applied_to_neo4j

    result = await db.execute(
        text(f"""
            SELECT * FROM knowledge.kg_relation_weight_snapshots
            WHERE {' AND '.join(where)}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = [dict(r) for r in result.mappings()]

    count_result = await db.execute(
        text(f"""
            SELECT COUNT(*) AS cnt FROM knowledge.kg_relation_weight_snapshots
            WHERE {' AND '.join(where)}
        """),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )
    total = count_result.scalar()

    return _envelope(request, {"items": rows, "total": total})


@router.get("/weight-trend/{relation_key}")
async def get_weight_trend(
    relation_key: str,
    request: Request,
    limit: int = Query(default=30, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """获取某个 relation_key 的权重变化趋势（用于趋势图）。

    返回按时间排序的权重快照列表，包含 old/new weight、置信区间。
    """
    result = await db.execute(
        text("""
            SELECT
                snapshot_id,
                calibration_run_id,
                new_effective_weight,
                old_effective_weight,
                confidence_lower_bound,
                confidence_upper_bound,
                evidence_case_count,
                support_count,
                against_count,
                neutral_count,
                weight_version,
                created_at
            FROM knowledge.kg_relation_weight_snapshots
            WHERE relation_key = :rk
            ORDER BY created_at ASC
            LIMIT :limit
        """),
        {"rk": relation_key, "limit": limit},
    )
    rows = [dict(r) for r in result.mappings()]

    # 附加每个快照的校准运行信息
    run_ids = list({r["calibration_run_id"] for r in rows if r.get("calibration_run_id")})
    runs_map: dict = {}
    if run_ids:
        run_result = await db.execute(
            text("""
                SELECT calibration_run_id, data_track, status, started_at, completed_at
                FROM knowledge.kg_calibration_runs
                WHERE calibration_run_id = ANY(:ids)
            """),
            {"ids": run_ids},
        )
        runs_map = {r["calibration_run_id"]: dict(r) for r in run_result.mappings()}

    for row in rows:
        row["calibration_run"] = runs_map.get(row["calibration_run_id"])

    if not rows:
        raise NotFoundError(f"关系 {relation_key} 没有权重快照记录")

    return _envelope(request, {
        "relation_key": relation_key,
        "trend": rows,
    })


@router.get("/relation-keys")
async def list_relation_keys(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """列出所有已知的 relation_key 及其最新权重。"""
    result = await db.execute(
        text("""
            SELECT DISTINCT ON (relation_key)
                relation_key,
                new_effective_weight,
                confidence_lower_bound,
                confidence_upper_bound,
                evidence_case_count,
                support_count,
                against_count,
                neutral_count,
                weight_version,
                created_at
            FROM knowledge.kg_relation_weight_snapshots
            ORDER BY relation_key, created_at DESC
        """),
    )
    rows = [dict(r) for r in result.mappings()]
    return _envelope(request, {"items": rows})


@router.get("/stats")
async def get_kg_stats(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """KG 校准总体统计信息。"""
    stats: dict = {}

    r = await db.execute(text("SELECT COUNT(*) AS cnt FROM knowledge.kg_relation_observations"))
    stats["total_observations"] = r.scalar() or 0

    r = await db.execute(text("SELECT direction, COUNT(*) AS cnt FROM knowledge.kg_relation_observations GROUP BY direction ORDER BY direction"))
    stats["observations_by_direction"] = {row["direction"]: row["cnt"] for row in r.mappings()}

    r = await db.execute(text("SELECT status, COUNT(*) AS cnt FROM knowledge.kg_calibration_runs GROUP BY status ORDER BY status"))
    stats["calibration_runs_by_status"] = {row["status"]: row["cnt"] for row in r.mappings()}

    r = await db.execute(text("SELECT status, COUNT(*) AS cnt FROM knowledge.kg_sync_jobs GROUP BY status ORDER BY status"))
    stats["sync_jobs_by_status"] = {row["status"]: row["cnt"] for row in r.mappings()}

    r = await db.execute(text("SELECT COUNT(DISTINCT relation_key) AS cnt FROM knowledge.kg_relation_observations"))
    stats["unique_relation_keys"] = r.scalar() or 0

    r = await db.execute(text("""
        SELECT calibration_run_id, data_track, status, started_at,
               completed_at, target_weight_version
        FROM knowledge.kg_calibration_runs
        WHERE status = 'SUCCEEDED'
        ORDER BY completed_at DESC NULLS LAST LIMIT 1
    """))
    row = r.mappings().first()
    if row:
        latest = dict(row)
        latest["calibration_run_id"] = str(latest["calibration_run_id"])
        stats["latest_calibration"] = latest
    else:
        stats["latest_calibration"] = None

    return _envelope(request, stats)
