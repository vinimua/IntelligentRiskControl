"""严格区分业务轮次与 Worker 技术重试。"""

from uuid import uuid4

from packages.models.common.enums import IterationExitReason
from packages.models.iteration.round_control import RetryIdentity, RoundTransition

from .config_loader import IterationConfigBundle, load_iteration_config


class IterationRoundController:
    def __init__(self, config: IterationConfigBundle | None = None) -> None:
        self.config = config or load_iteration_config()

    def technical_retry(self, identity: RetryIdentity) -> RoundTransition:
        next_retry = identity.technical_retry_count + 1
        if next_retry > self.config.iteration.max_technical_retries:
            return RoundTransition(
                allowed=False,
                exit_reason=IterationExitReason.TECHNICAL_FAILURE,
                explanation="技术重试次数已耗尽，业务轮次不增加",
            )
        return RoundTransition(
            allowed=True,
            next_business_round=identity.business_round,
            training_job_id=identity.training_job_id,
            experiment_id=identity.experiment_id,
            explanation="技术重试复用 training_job_id 和 experiment_id",
        )

    def next_business_round(self, current_business_round: int) -> RoundTransition:
        if current_business_round >= self.config.iteration.max_iteration_rounds:
            return RoundTransition(
                allowed=False,
                exit_reason=IterationExitReason.MAX_BUSINESS_ROUNDS_REACHED,
                explanation="三轮业务策略均未通过，停止自动迭代",
            )
        return RoundTransition(
            allowed=True,
            next_business_round=current_business_round + 1,
            training_job_id=str(uuid4()),
            experiment_id=str(uuid4()),
            explanation="业务策略发生变化，创建新的 experiment_id",
        )
