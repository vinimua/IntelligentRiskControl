"""结构化日志配置

所有模块使用 trace_id、lifecycle_run_id 等结构化字段。
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
lifecycle_run_id_var: ContextVar[str] = ContextVar("lifecycle_run_id", default="")


def add_trace_context(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """将 contextvars 中的 trace_id 等注入到每条日志。"""
    trace_id = trace_id_var.get()
    if trace_id:
        event_dict["trace_id"] = trace_id
    lifecycle_run_id = lifecycle_run_id_var.get()
    if lifecycle_run_id:
        event_dict["lifecycle_run_id"] = lifecycle_run_id
    return event_dict


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """初始化 structlog，绑定到 Python 标准 logging。"""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_trace_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 抑制第三方库的噪音日志
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
