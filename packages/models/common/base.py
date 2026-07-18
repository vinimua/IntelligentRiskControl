"""
通用响应包装与错误模型
来源：技术开发文档 V1.4.2 §13.3-§13.4, 知识图谱接口 V1.1 §2.3-§2.4
"""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """所有契约模型的严格基类。"""

    model_config = ConfigDict(extra="forbid")


class BaseResponse(ContractModel):
    """通用成功响应"""
    success: bool = True
    code: str = "OK"
    message: str = "success"
    data: dict | list | None = None
    trace_id: str | None = None


class ErrorDetail(ContractModel):
    """错误响应"""
    success: bool = False
    code: str
    message: str
    data: None = None
    trace_id: str | None = None
    retryable: bool = False
    details: dict | None = None
