"""
LangGraph IterationSubgraph State
"""

from ..common.base import ContractModel

class IterationState(ContractModel):
    """任务三子图 State — 承载多轮训练、异步等待、失败重试和最佳 Challenger 选择"""

    lifecycle_run_id: str
    diagnosis_run_id: str
    model_id: str
    base_model_version: str
    primary_root_cause_code: str

    iteration_run_id: str | None = None
    current_round: int = 0
    max_rounds: int = 3
    current_strategy_code: str | None = None
    current_training_job_id: str | None = None
    current_experiment_id: str | None = None
    current_candidate_version: str | None = None

    best_experiment_id: str | None = None
    best_challenger_version: str | None = None
    challenger_qualified: bool = False

    training_retry_count: int = 0
    max_training_retries: int = 3

    exit_reason: str | None = None
    last_error: dict | None = None
