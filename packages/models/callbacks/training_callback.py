"""
Training Worker Callback
"""

from ..common.base import ContractModel

from ..common.enums import WorkerStatus

class TrainingCallback(ContractModel):
    """
    训练完成/失败后恢复 IterationSubgraph
    幂等键：training_job_id
    """

    training_job_id: str
    status: WorkerStatus
    candidate_version: str | None = None
    experiment_id: str | None = None
    metrics: dict | None = None
    artifact_uri: str | None = None
    error_code: str | None = None
    error_message: str | None = None
