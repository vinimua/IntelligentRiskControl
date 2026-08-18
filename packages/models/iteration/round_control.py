"""三轮业务迭代与技术重试合同。"""

from pydantic import Field

from ..common.base import ContractModel
from ..common.enums import IterationExitReason


class RetryIdentity(ContractModel):
    training_job_id: str
    experiment_id: str
    # 最大业务轮次统一为 2（A7 定稿 §5）
    business_round: int = Field(ge=1, le=2)
    technical_retry_count: int = Field(ge=0)


class RoundTransition(ContractModel):
    allowed: bool
    next_business_round: int | None = None
    training_job_id: str | None = None
    experiment_id: str | None = None
    exit_reason: IterationExitReason | None = None
    explanation: str
