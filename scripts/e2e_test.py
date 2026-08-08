"""端到端测试：patch 监控数据为退化数据 → API 触发 lifecycle → LangGraph 全程自跑。

用法: python scripts/e2e_test.py
前置: API 需在 8002 端口运行（python -m uvicorn apps.modelops_api.main:app --host 0.0.0.0 --port 8002 --reload）
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API = "http://localhost:8002"
MODEL_ID = "credit_model_001"
CHAMPION = "challenger_v1"


def make_degraded_df(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """生成退化数据：AUC 从 ~0.75 跌到 ~0.64 + device_risk_score PSI 漂移。"""
    rng = np.random.default_rng(seed)
    N = 2 * n

    b_device = rng.normal(0.50, 0.12, N)
    b_label = rng.binomial(1, 0.3, N).astype(float)
    b_noise = rng.normal(0, 0.08, N)
    b_scores = np.clip(b_label * 0.40 + b_device * 0.55 + b_noise + 0.05, 0.001, 0.999)

    c_device = rng.normal(0.22, 0.28, N)
    c_label = rng.binomial(1, 0.3, N).astype(float)
    c_noise = rng.normal(0, 0.10, N)
    c_scores = np.clip(c_label * 0.40 + c_device * 0.35 + c_noise + 0.08, 0.001, 0.999)

    def make(scores, labels, device, prefix, start_date):
        df = pd.DataFrame({
            "y_pred_proba": scores, "y_true": labels,
            "device_risk_score": device,
            "income_score": rng.normal(0.50, 0.15, N),
            "age_score": rng.normal(0.60, 0.10, N),
            "credit_history_score": rng.normal(0.45, 0.18, N),
            "apply_time": pd.date_range(start_date, periods=N, freq="h"),
        })
        df["sample_id"] = [f"{prefix}_{i:06d}" for i in range(N)]
        return df

    baseline = make(b_scores, b_label, b_device, "W0", "2025-01-01")
    current = make(c_scores, c_label, c_device, "W3", "2025-12-01")
    return baseline, current


def print_state(state: dict):
    """简洁打印 lifecycle state。"""
    ke = state.get("last_error")
    print(f"  phase: {state.get('current_phase')}")
    if state.get("has_alerts") is not None:
        print(f"  alerts: {state.get('has_alerts')}, count={state.get('alert_count')}, "
              f"severity={state.get('max_alert_severity')}")
    for k in ["primary_root_cause_code", "primary_root_cause_score",
              "recommended_action", "need_iteration"]:
        if state.get(k) is not None:
            print(f"  {k}: {state.get(k)}")
    for k in ["decision_proposal_id", "selected_strategy_code", "training_mode",
              "training_plan_id", "experiment_id", "iteration_run_id", "business_round",
              "challenger_version", "challenger_qualified",
              "deployment_id", "deployment_stage", "deployment_decision",
              "iteration_exit_reason"]:
        if state.get(k) is not None:
            print(f"  {k}: {state.get(k)}")
    if ke:
        print(f"  last_error: {ke.get('reason')} — {ke.get('message', '')[:200]}")


async def poll_lifecycle(client: httpx.AsyncClient, lifecycle_id: str,
                         max_wait: int = 120, interval: int = 3):
    """轮询 lifecycle 直到终态或超时。"""
    headers = {"X-Trace-Id": str(uuid.uuid4()), "X-Schema-Version": "1.0"}
    terminal = {"EVENT_CLOSED", "NO_ALERT", "FAILED", "CANCELLED", "MANUAL_REVIEW"}

    for i in range(max_wait // interval):
        await asyncio.sleep(interval)
        resp = await client.get(f"{API}/api/lifecycle-runs/{lifecycle_id}",
                                headers=headers)
        data = resp.json()["data"]
        state = data.get("state", {})
        phase = state.get("current_phase", data.get("current_phase", "?"))
        elapsed = (i + 1) * interval
        print(f"\n[{elapsed}s] phase={phase}")
        if phase in terminal:
            print_state(state)
            return state
        if phase in ("WAITING_TRAINING_CALLBACK", "WAITING_FEATURE_RECONSTRUCTION"):
            print(f"  ⏸  waiting for callback — 手动 resume 或超时")
            print_state(state)
            return state

    print(f"\n[{max_wait}s] 超时，最后状态:")
    resp = await client.get(f"{API}/api/lifecycle-runs/{lifecycle_id}", headers=headers)
    state = resp.json()["data"].get("state", {})
    print_state(state)
    return state


async def main():
    baseline, current = make_degraded_df()
    print(f"合成数据: baseline={len(baseline)} rows, current={len(current)} rows")
    print(f"baseline AUC≈0.75, current AUC≈0.64 (预期: 监控告警)")

    headers = {"Content-Type": "application/json",
               "X-Trace-Id": str(uuid.uuid4()), "X-Schema-Version": "1.0"}

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. 启动 lifecycle
        print("\n=== 启动 lifecycle ===")
        resp = await client.post(f"{API}/api/lifecycle-runs", headers=headers, json={
            "model_id": MODEL_ID, "champion_version": CHAMPION,
            "trigger_type": "MANUAL_TRIGGER",
        })
        lifecycle_id = resp.json()["data"]["lifecycle_run_id"]
        print(f"lifecycle_run_id: {lifecycle_id}")
        print(f"初始 phase: {resp.json()['data']['current_phase']}")

        # 2. 等待完成
        print("\n=== 等待生命周期完成 ===")
        await poll_lifecycle(client, lifecycle_id)


if __name__ == "__main__":
    asyncio.run(main())
