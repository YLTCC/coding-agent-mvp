"""schemas.py —— 请求 / 响应模型（pydantic v2）

⚠️ 这里**只有"名字"，没有"路径"**：`CreateSessionRequest.workspace` 是**工作区名字**，
服务端 resolve 后校验它在 `WEB_WORKSPACE_ROOT` 之下（决策 D）。
允许前端传任意路径 = 把 4.6 辛苦立起来的边界交给匿名请求。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    workspace: str | None = Field(
        default=None,
        description="工作区**名字**（相对 WEB_WORKSPACE_ROOT）。留空 = 用根目录本身。",
    )


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, description="用户任务原文")


class ApprovalRequest(BaseModel):
    allow: bool = Field(description="true=允许，false=拒绝")


class SessionInfo(BaseModel):
    sid: str
    workspace: str
    shared: bool
    shared_with: list[str]
    inflight: bool
    consumers: int
    created_at: float
    pending_approvals: int


class CreatedSession(BaseModel):
    sid: str
    workspace: str
    shared: bool
    shared_with: list[str]
    log_path: str | None = None
