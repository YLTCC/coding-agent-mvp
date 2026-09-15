"""app.py —— FastAPI app + 路由 + lifespan（阶段三·b 的接线处）

目标只有一句：**给已经 async 的主循环接一根线**——
输入 `input()` → HTTP，输出 `print` → SSE，审批 stdin → 按钮。

文件里几处"看起来多余"的写法都是有意为之的，改动前请先读注释：

  * `POST .../approvals/{aid}` **必须 `async def`**（决策 C.2）。
    写成 `def`，FastAPI 会把它丢进 threadpool → `future.set_result()` 变成跨线程
    → `set_result` 内部的 `loop.call_soon()` 不写 self-pipe → **唤醒丢失**，
    loop 空闲时 await 侧一直不醒。**类型签名上看不出来**，W-T3b 拿测试钉住它。
  * SSE 的 `data:` 必须是单行 JSON（`sse_frame` 里有断言）。
  * SSE 响应头显式 `charset=utf-8`：否则前端 `EventSource` 会把中文解码错。
  * **必须 `--workers 1`**：内存 registry 是进程内状态，多 worker 会
    "建会话落在 A 进程、发消息打到 B 进程" → 404，**且不报错**（经典的
    "不报错的错误结论"）。见 run_web.py 的注释。
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Callable

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.config import Config

from .events import DISK_EVENT_TYPES, HEARTBEAT, WebEvent, parse_last_event_id, record_to_event, sse_frame
from .schemas import ApprovalRequest, CreateSessionRequest, MessageRequest
from .session_registry import (
    SessionNotFound,
    SessionRegistry,
    WebSession,
    prepare_web_config,
)

# SSE 空闲心跳间隔（秒）：防中间层掐掉空闲连接
HEARTBEAT_SECONDS = 15.0


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SessionNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


async def _run_task(session: WebSession, text: str) -> None:
    """把一个任务跑完。**不 await**（`POST /messages` 返回 202 就撒手）。"""
    try:
        await session.agent.run_task(text)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 —— 任务炸了也要让浏览器**看见**
        # 绝不能静默：浏览器那边只会看到"消息流停住了"，那是最难查的形态
        session.out("error", error=f"{type(e).__name__}: {e}")
    finally:
        session.inflight = False


async def _sse_stream(session: WebSession, last_event_id: int | None):
    """SSE 事件流（决策 F / I / J）。

    续传语义被钉死为：**只能从"最后一个落盘记录"续，不能从"最后一个 delta"续**
    （决策 I：delta 没有 seq，本来就不落盘）。
    """
    session.consumer_attached()
    try:
        # ① 续传起点（决策 F / I / W4）
        #    - 全新连接（没有 Last-Event-ID）→ 先给"已提交态"快照，
        #      水位直接设成快照覆盖到的最大 seq（快照已含全部落盘记录，
        #      再逐条重放一遍只会让前端看到重复）
        #    - 重连（带 Last-Event-ID）→ 从磁盘补它之后**已提交**的内容
        #    两条路都**只从落盘记录续，不能从 delta 续**（delta 根本没有 seq）
        if last_event_id is None:
            payload, sent_upto = session.replay_snapshot()
            yield sse_frame(WebEvent(kind="replay", fields=payload))
        else:
            sent_upto = last_event_id
            try:
                for event in session.disk_events_after(sent_upto):
                    yield sse_frame(event)
                    sent_upto = event.seq
            except Exception as e:  # noqa: BLE001
                # 日志中间损坏（note3 I6 不允许静默跳过）→ 大声告诉用户，
                # 但不把整条 SSE 连接打死（否则连"为什么坏了"都看不到）
                yield sse_frame(WebEvent(kind="error", fields={
                    "error": f"读取会话日志失败：{type(e).__name__}: {e}"}))

        # ② 断线期间被丢掉的 delta 要**显式记账**：换 API 只把失败从"静默"变成"可见"，
        #    "可见但没处理 = 还是丢"。所以这里给一条 `lagged`
        dropped = session.events.dropped_deltas
        if dropped:
            session.events.dropped_deltas = 0
            yield sse_frame(WebEvent(kind="lagged", fields={
                "dropped_deltas": dropped,
                "detail": f"你断线期间跳过了 {dropped} 个片段"
                          f"（定稿文本已落盘，明细见 {session.log_path_or_placeholder()}）",
            }))

        # ③ 队列里已经攒下的（跳过磁盘已经覆盖过的 seq）
        while True:
            event = session.events.try_get()
            if event is None:
                break
            if event.seq is not None:
                if event.seq <= sent_upto:
                    continue
                sent_upto = event.seq
            yield sse_frame(event)

        # ⑤ 实时
        while True:
            try:
                event = await asyncio.wait_for(session.events.get(), HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield HEARTBEAT
                continue
            if event.seq is not None:
                if event.seq <= sent_upto:
                    continue
                sent_upto = event.seq
            yield sse_frame(event)
    finally:
        # 消费者消失 → 脱钩（决策 J）+ 唤醒并拒绝所有待决审批（决策 C.3）
        session.consumer_detached()


def create_app(cfg: Config | None = None,
               llm_factory: Callable[[], Any] | None = None) -> FastAPI:
    """建一个独立的 app（测试要隔离，所以做成工厂；模块底部再懒加载一个默认 app）。"""
    cfg = cfg if cfg is not None else Config()
    # 决策 K 第 1 条：审计目录必须在工作区之外，否则**启动失败**
    prepare_web_config(cfg)

    registry = SessionRegistry(cfg)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # 关服：排干并 close 所有 logger（丢尾部记录 = 丢 commit 边界，
        # 正是 note3 §4.4 的意义）。Windows 下 Ctrl+C 与 CLI 的 KeyboardInterrupt
        # 处理不同，必须在这里统一收口。
        await registry.close_all()

    app = FastAPI(title="Coding Agent Web", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.registry = registry

    # ---------- 鉴权（决策 G）----------
    async def require_token(request: Request) -> None:
        token = cfg.WEB_TOKEN
        if not token:
            return          # 没配 token = 不鉴权（默认只绑 127.0.0.1，别改成 0.0.0.0）
        supplied = None
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
        if not supplied:
            # SSE：EventSource **不能设 header**，只能走 query
            supplied = request.query_params.get("token")
        if supplied != token:
            raise HTTPException(status_code=401, detail="缺少或错误的 token")

    guard = Depends(require_token)

    @app.exception_handler(SessionNotFound)
    async def _not_found(_request: Request, exc: SessionNotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    api = APIRouter(prefix="/api", dependencies=[guard])

    # ---------- 会话 ----------

    @api.post("/sessions", status_code=201)
    async def create_session(req: CreateSessionRequest | None = None):
        name = req.workspace if req is not None else None
        try:
            session = registry.create(
                name, llm_client=(llm_factory() if llm_factory is not None else None)
            )
        except Exception as e:  # WorkspaceNotAllowed / SessionNotFound
            raise _http_error(e) from e
        return {
            "sid": session.sid,
            "workspace": str(session.workspace),
            "shared": bool(session.shared_with),
            "shared_with": list(session.shared_with),
            "log_path": str(session.logger.path) if session.logger is not None else None,
        }

    @api.get("/sessions")
    async def list_sessions():
        return {"sessions": [s.info() for s in registry.sessions.values()]}

    @api.get("/sessions/{sid}")
    async def get_session(sid: str):
        return registry.get(sid).info()

    @api.get("/sessions/{sid}/replay")
    async def replay(sid: str):
        return registry.get(sid).replay_payload()

    @api.delete("/sessions/{sid}")
    async def close_session(sid: str):
        ok = await registry.close(sid)
        if not ok:
            raise HTTPException(status_code=404, detail=f"会话不存在：{sid}")
        return {"status": "closed", "sid": sid}

    # ---------- 任务 ----------

    @api.post("/sessions/{sid}/messages", status_code=202)
    async def send_message(sid: str, req: MessageRequest):
        session = registry.get(sid)
        if session.inflight:
            # 决策 H：每会话同时只允许 1 个 in-flight 任务。比"调大线程池"对的理由是
            # run_task 会改 self.messages —— 同一会话并发跑两个任务，两条主线往
            # 同一个 messages 里插消息**必然错乱**，而且**不报错**。
            raise HTTPException(
                status_code=409,
                detail="该会话已有一个任务在运行（每会话同时只允许 1 个 in-flight 任务）",
            )
        session.inflight = True
        # **不 await**：流式输出走 SSE，这个请求 202 就返回
        session.task = asyncio.create_task(_run_task(session, req.text))
        return {"status": "accepted", "sid": sid}

    # ---------- SSE ----------

    @api.get("/sessions/{sid}/stream")
    async def stream(sid: str, request: Request, last_event_id: str | None = None):
        session = registry.get(sid)
        resid = parse_last_event_id(
            # EventSource 重连时自动带 `Last-Event-ID` 头；手写客户端可用 query
            request.headers.get("last-event-id") or last_event_id
        )
        return StreamingResponse(
            _sse_stream(session, resid),
            media_type="text/event-stream; charset=utf-8",   # 中文靠它才对
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ---------- 审批 ----------
    # ⚠️⚠️ **必须 async def**（决策 C.2）。
    # 写成 `def` → FastAPI 用 run_in_threadpool 执行 → 下面的 set_result 变成跨线程
    # → `loop.call_soon()` 不写 self-pipe → **loop 空闲时唤醒丢失**（低负载才复现，
    # 高负载被回程流量捎带唤醒 → 生产里表现为"偶发卡住"）。
    # 这个坑从类型签名上看不出来，所以 W-T3b 专门断言"处理器与 loop 同线程"。
    @api.post("/sessions/{sid}/approvals/{aid}")
    async def resolve_approval(sid: str, aid: str, req: ApprovalRequest):
        session = registry.get(sid)
        granted = session.approvals.resolve(aid, req.allow)
        if not granted:
            # 晚到 / 重复点击：**绝不静默改判**（决策 C.3）。
            # 超时已拒之后又点"允许"，如果这里返回 200，用户会以为批准生效了，
            # 而工具早已按拒绝返回——那是最典型的"不报错的错误结论"。
            raise HTTPException(
                status_code=409,
                detail="该审批已超时或被处理，本次点击不生效（结论不会改变）",
            )
        return {"status": "ok", "aid": aid, "allow": req.allow}

    app.include_router(api)

    # ---------- 静态前端（同源部署，**不开** allow_origins=["*"]，决策 G）----------
    if cfg.WEB_STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(cfg.WEB_STATIC_DIR), html=True),
                  name="static")

    return app


def __getattr__(name: str):
    """懒加载模块级 `app`（PEP 562）。

    为什么不直接 `app = create_app()`：`create_app` 会做**启动断言**（决策 K），
    而默认配置下 `SESSION_DIR` 就在工作区里 → 一 import 就抛。测试只想
    `from web.app import create_app` 也会跟着炸。所以让"启动"这件事只在真的
    要被当作 WSGI/ASGI 应用取用时发生：

        uvicorn web.app:app        # ← 只有这时才触发断言
        run_web.py                 # ← 或干脆显式 create_app(cfg)
    """
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
