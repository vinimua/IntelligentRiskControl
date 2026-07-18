"""基础异常类与全局异常处理器"""

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def request_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", request.headers.get("X-Trace-Id", ""))


class AppException(Exception):
    """业务异常基类 — 对应统一错误响应格式。"""

    def __init__(
        self,
        code: str = "INTERNAL_ERROR",
        message: str = "内部错误",
        status_code: int = 500,
        retryable: bool = False,
        details: dict | None = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}


class NotFoundError(AppException):
    def __init__(self, message: str = "资源不存在", code: str = "NOT_FOUND"):
        super().__init__(code=code, message=message, status_code=404)


class ValidationAppError(AppException):
    def __init__(self, code: str, message: str):
        super().__init__(code=code, message=message, status_code=400)


class ConflictError(AppException):
    def __init__(self, message: str = "状态冲突", code: str = "CONFLICT"):
        super().__init__(code=code, message=message, status_code=409)


class ForbiddenError(AppException):
    def __init__(self, message: str = "无权限", code: str = "FORBIDDEN"):
        super().__init__(code=code, message=message, status_code=403)


class ServiceUnavailableError(AppException):
    def __init__(self, message: str = "依赖服务不可用", code: str = "INTERNAL_ERROR"):
        super().__init__(code=code, message=message, status_code=503, retryable=True)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "code": exc.code,
                "message": exc.message,
                "data": None,
                "trace_id": request_trace_id(request),
                "retryable": exc.retryable,
                "details": exc.details,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "code": "VALIDATION_ERROR",
                "message": "请求参数校验失败",
                "data": None,
                "trace_id": request_trace_id(request),
                "retryable": False,
                "details": {"errors": jsonable_encoder(exc.errors())},
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "code": "INTERNAL_ERROR",
                "message": "内部服务错误",
                "data": None,
                "trace_id": request_trace_id(request),
                "retryable": False,
                "details": {},
            },
        )
