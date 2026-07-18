"""统一配置加载与环境变量校验"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── 环境 ──
    env: str = "development"

    # ── API ──
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # ── PostgreSQL ──
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "riskitem"
    postgres_user: str = "riskitem"
    postgres_password: str = "riskitem_dev"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def asyncpg_dsn(self) -> str:
        """asyncpg 只接受 postgresql:// 或 postgres:// 协议。"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ──
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── MinIO ──
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "riskitem"
    minio_secure: bool = False

    # ── MLflow ──
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"
    mlflow_artifact_root: str = "s3://riskitem/mlflow/"

    # ── Neo4j ──
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "riskitem_dev"

    # ── Qdrant ──
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_alias: str = "risk_knowledge_current"

    # ── 日志 ──
    log_level: str = "INFO"
    log_format: str = "json"

    # ── 测试 ──
    test_database_url: str = (
        "postgresql+asyncpg://riskitem:riskitem_dev@localhost:5432/riskitem_test"
    )
    skip_integration: bool = False


settings = Settings()
