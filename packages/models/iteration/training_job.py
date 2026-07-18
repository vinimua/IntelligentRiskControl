"""
Training Worker 合同
来源：技术开发文档 V1.4.2 §8.5, 接口总汇 V1.1 §7.3
"""

from ..common.base import ContractModel

from ..common.enums import WorkerStatus

class TrainingJobInput(ContractModel):
    """Training Worker 输入 — 训练任务参数"""

    training_job_id: str
    iteration_run_id: str
    strategy_code: str
    training_window: str
    base_model_version: str
    seed: int

class TrainingJobOutput(ContractModel):
    """Training Worker 输出 — 训练结果"""

    candidate_version: str
    experiment_id: str
    metrics: dict
    artifact_uri: str
    status: WorkerStatus
