"""模型注册 Service"""

from __future__ import annotations

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import NotFoundError
from ..repositories.model_repo import ModelRepo

logger = structlog.get_logger(__name__)


class ModelRegistryService:
    """模型、版本和部署状态管理。"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = ModelRepo(session)

    async def register_model(
        self, model_id: str, model_name: str, model_type: str = "CREDIT_RISK"
    ) -> dict:
        existing = await self.repo.get_model(model_id)
        if existing:
            return existing
        return await self.repo.insert_model(model_id, model_name, model_type)

    async def register_version(
        self,
        model_id: str,
        version_code: str,
        role: str = "CHALLENGER",
    ) -> dict:
        model = await self.repo.get_model(model_id)
        if not model:
            raise NotFoundError(f"模型 {model_id} 不存在")

        existing = await self.repo.get_version(model_id, version_code)
        if existing:
            return existing

        result = await self.repo.insert_version(
            model_id=model_id,
            version_code=version_code,
            role=role,
        )
        logger.info(
            "version_registered",
            model_id=model_id,
            version_code=version_code,
            role=role,
        )
        return result

    async def promote_champion(self, model_id: str, version_code: str) -> None:
        version = await self.repo.get_version(model_id, version_code)
        if not version:
            raise NotFoundError(f"版本 {version_code} 不存在")
        await self.repo.set_champion(model_id, version_code)
        logger.info("champion_promoted", model_id=model_id, version=version_code)

    async def get_model_detail(self, model_id: str) -> dict | None:
        model = await self.repo.get_model(model_id)
        if not model:
            return None
        versions = await self.repo.list_versions(model_id)
        model["versions"] = versions
        return model
