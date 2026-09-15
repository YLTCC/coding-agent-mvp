"""session_registry.py —— 会话注册表：`sid → WebSession`（决策 D / E / H / K）

这里回答四个问题：

1. **会话长什么样**：WebSession 把"一个浏览器会话需要的东西"聚在一起
   （logger / agent / SSE 队列 / 审批桥 / in-flight 标志）。装配仍然走
   `main.build_session` —— **不新写"web 版 agent"**，那是本次改造"比看起来小"的原因。

2. **workspace 从哪来**（决策 D）：**服务端**派生，前端只能提交**工作区名字**。
   若允许 `POST /api/sessions {"workspace": "C:\\\\Windows"}`，等于把 4.6 辛苦立起来的
   边界交给匿名请求。校验在 `resolve_workspace`。

3. **同名 workspace 怎么办**（决策 D 末尾，选 C：**显式共享**）：
   两个会话选同一个 workspace 名 → 共享同一批业务文件（`sessions/` 分开了、
   业务文件却没有）。这在文件层**不可能隔离**（每会话 mkdir 专属子目录对
   coding agent 不可实现：建空目录则 agent 看不到用户项目）。所以选择把它
   **摆到明面上**：`session_start` 记 `shared: true` + `shared_with: [...]`，UI 常驻标记。
   📌 **key 必须先 `resolve()` 再比**：否则 Windows 大小写不敏感 + 短名/长名
   （`PROGRA~1`）+ 相对路径会把"同一个目录"判成两个 → **静默共享照旧**，
   而 UI 却标着"独占"（比不标记更坏：标记变成了谎话）。

4. **审计目录放哪**（决策 K 第 1 条）：web 模式下 `SESSION_DIR` **必须**在
   `WEB_WORKSPACE_ROOT` 之外，启动时**断言**，不满足就启动失败。
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any

from agent.session_log import SessionLogger, new_session_id

# 落盘记录类型的白名单在 events 里；这里只用作过滤条件（避免循环 import）
from . import events as events_mod

import main as main_mod

from .approvals import ApprovalBridge
from .events import EventQueue, WebEvent, record_to_event


class WorkspaceNotAllowed(Exception):
    """前端提交的工作区名不合法（越界 / 不存在）。路由层转成 HTTP 400。"""


class SessionNotFound(Exception):
    """sid 不存在（或已被关）。路由层转成 HTTP 404。"""


# ============================================================
# 启动期配置收口（决策 K 第 1 条：把审计目录移出沙箱）
# ============================================================

def prepare_web_config(cfg: Any) -> Path:
    """把 `SESSION_DIR` 放到 `WEB_WORKSPACE_ROOT` **之外**，并断言成功。

    为什么这是**事故级**而不是"小瑕疵"（决策 K）：
      1. **跨会话读取**：会话 A 的 agent 一句 `read_file sessions/<B>.jsonl`
         就能拿到 B 的 system prompt、全部对话与工具输出——"不串台"当场失效
         （而且这种走法很自然：模型看到目录里有 jsonl，会想去读）；
      2. **审计篡改**：`write_file` / `edit_file` 能改自己或别人的审计日志，
         而审计的价值恰恰是"事后不可抵赖"；
      3. `sessions/` 会污染 `list_dir` / `glob`（如 `**/*.jsonl`）的结果。

    ⚠️ **只打 warning 等于没做**（警告会被忽略、日志会滚走）→ 这里直接抛，
    由启动流程转成"服务器起不来 + 明确修法"。

    只在 web 启动时调用：CLI 的 `sessions/` 仍在 workspace 内，那条路径靠
    `tools._is_protected_audit_path`（决策 K 第 2 条）兜底。
    """
    if os.getenv("AGENT_SESSION_DIR"):
        # 用户显式配了就尊重它（但仍然要过断言——配错了一样会出事）
        target = Path(cfg.SESSION_DIR).resolve()
    else:
        local = os.getenv("LOCALAPPDATA")
        if local:
            target = (Path(local) / "coding-agent" / "sessions").resolve()
        else:
            # 没有 LOCALAPPDATA（非 Windows / 精简环境）→ 退到沙箱根的**兄弟目录**
            root = Path(cfg.WEB_WORKSPACE_ROOT).resolve()
            target = (root.parent / f"{root.name}_sessions").resolve()

    root = Path(cfg.WEB_WORKSPACE_ROOT).resolve()
    if target.is_relative_to(root):
        raise RuntimeError(
            f"❌ 拒绝启动：会话日志目录 {target} 位于工作区根 {root} 之内。\n"
            f"   web 模式下这会让**另一个会话**的 agent 直接读到本会话的全部对话，\n"
            f"   并且可以用 write_file/edit_file 篡改审计日志（事后不可抵赖就没了）。\n"
            f"   修法（任选其一）：\n"
            f"     1) 删掉 .env 里的 AGENT_SESSION_DIR，让 web 用默认的\n"
            f"        %LOCALAPPDATA%\\coding-agent\\sessions；\n"
            f"     2) 把 AGENT_SESSION_DIR 指到工作区根**之外**的绝对路径。\n"
        )
    cfg.SESSION_DIR = target
    return target


# ============================================================
# WebSession
# ============================================================

class WebSession:
    """一个浏览器会话的全部状态。

    ⚠️ **两个线程都在碰它**：SSE 队列的变更只在 loop 线程（`_push_ui`），
    但事件的来源既有 loop 线程（agent 的 delta）也有 threadpool 线程
    （tools 的审批）。所以所有入队都走 `_dispatch_threadsafe`。
    """

    def __init__(self, sid: str, workspace: Path, cfg: Any,
                 loop: asyncio.AbstractEventLoop, shared_with: list[str] | None = None):
        self.sid = sid
        self.workspace = workspace
        self.cfg = cfg
        self.loop = loop
        # 建会话时所在的线程 = loop 线程。审批桥的线程约束（决策 C.2）
        # 需要一个"loop 线程是谁"的参照物，W-T3b 就是拿它比的
        self.loop_thread_id = threading.get_ident()
        self.created_at = time.time()
        # 同名 workspace 的其它活跃会话（决策 D 选 C：显式共享，不假装隔离）
        self.shared_with: list[str] = list(shared_with or ())
        self.events = EventQueue(maxsize=getattr(cfg, "WEB_QUEUE_MAXSIZE", 256))
        self.approvals = ApprovalBridge(self)
        self.agent: Any = None
        self.logger: SessionLogger | None = None
        self.inflight = False
        self.task: asyncio.Task | None = None
        # SSE 消费者计数：建/断连接时 ±1。0 = **没有人在看**
        self.consumers = 0

    # ---------- 状态 ----------

    @property
    def attached(self) -> bool:
        return self.consumers > 0

    @property
    def detached(self) -> bool:
        """没有人看这个会话（浏览器关了 / 从没连上）。

        用途一：delta 不入队（决策 J 的省内存 + 省序列化）。
        用途二：审批**不进入等待**，立即拒绝（决策 J：丢的是"功能性"事件，
        任务会白等满超时，用户看到的是"卡住 5 分钟然后失败"）。
        """
        return self.consumers <= 0

    def info(self) -> dict:
        return {
            "sid": self.sid,
            "workspace": str(self.workspace),
            "shared": bool(self.shared_with),
            "shared_with": list(self.shared_with),
            "inflight": self.inflight,
            "consumers": self.consumers,
            "created_at": self.created_at,
            "pending_approvals": self.approvals.pending_count,
        }

    # ---------- 入队：两条来源，一个出口 ----------

    def _dispatch_threadsafe(self, event: WebEvent) -> None:
        """把事件从**任意线程**送到 loop 线程再入队（决策 A）。

        `asyncio.Queue` / 我们的 EventQueue 都**不是线程安全的**。
        而事件确实来自两个线程：
          - `on_record` 由 `SessionLogger.emit` 触发，而 emit 的调用方可能是
            threadpool（审批事件）；
          - `out` 由 agent 的 `_out` 触发，主循环线程与 threadpool 都有。
        所以统一走 `call_soon_threadsafe`（它对任意线程都安全，包括 loop 自己的
        线程，因此不必判断当前线程，一条路径）。

        ⚠️ 失败形态是"**静默丢事件**"而不是崩溃：关服竞态下
        `call_soon_threadsafe` 会抛 `RuntimeError`，而它的上游
        `_emit_event` 有 `except Exception: pass` → 现象是"偶尔少一条工具卡片"，
        不报错、不复现。所以这里显式吞掉的同时**必须**只吞"loop 已关"这一种。
        """
        loop = self.loop
        if loop.is_closed():
            # 关服竞态：UI 事件丢掉是可以接受的（磁盘上还有）。
            # 绝不向上抛——那会把 run_task 带崩。
            return
        try:
            loop.call_soon_threadsafe(self._push_ui, event)
        except RuntimeError:
            # is_closed() 与 call_soon_threadsafe 之间 loop 被关掉了
            pass

    def _push_ui(self, event: WebEvent) -> None:
        """⚠️ **这个函数跑在 loop 线程**——背压策略在这里做（决策 A / J）。"""
        if event.kind in ("delta",) and self.detached:
            # 没人看：delta 根本不入队。省内存，也省掉每 token 一次的 JSON 序列化
            self.events.dropped_deltas += 1
            return
        self.events.push(event)

    # ---------- agent / tools 的两个注入点 ----------

    def out(self, kind: str, **fields) -> None:
        """W1 输出通道的 web 实现：agent 与 tools 的"给人看"的输出都从这里进 SSE。

        ⚠️ **可能在任何线程被调用**（tools 的审批提示在 threadpool 里）。
        """
        self._dispatch_threadsafe(WebEvent(kind=kind, fields=dict(fields), seq=None))

    def sink(self, rtype: str, fields: dict) -> None:
        """审计事件外出通道，注入给 `ToolContext.event_sink`。

        ⚠️ 签名是**位置参数** `(rtype, fields)`——`tools._emit_event` 就是这么调的
        （`sink(rtype, fields)`）。写成 `sink(rtype, **fields)` **不报错**：
        异常会被 `_emit_event` 的 `except: pass` 吞掉，结果是**审计静默消失**。
        W-T11 抓的就是这一条。

        职责：落盘（有 logger 时）+ 推 SSE。落盘那半会经由 `logger.listener`
        回调 `on_record`，所以这里**不要**重复推一次。
        """
        if self.logger is not None:
            try:
                self.logger.emit(rtype, **fields)
                return          # listener 已经把事件推给 SSE 了
            except Exception:   # noqa: BLE001 —— 日志已关（关服竞态）
                pass
        self._dispatch_threadsafe(WebEvent(kind=rtype, fields=dict(fields), seq=None))

    def on_record(self, seq: int, rtype: str, fields: dict) -> None:
        """`SessionLogger.listener`：每条**落盘**记录都带 seq → SSE 的 `id:`（决策 I）。"""
        self._dispatch_threadsafe(WebEvent(kind=rtype, fields=dict(fields), seq=seq))

    # ---------- 审批相关的 UI 事件 ----------

    def push_approval_request(self, aid: str, action: str, detail: str) -> None:
        """让浏览器渲染"⚠️ 请求写入 ×××" + 允许/拒绝按钮。

        这条**不带 seq**：它没有落盘记录（磁盘上的 `approval` 行是**决定做出之后**
        才写的，而且不带 aid）。所以它**不可续传**——重连后拿不到它，
        此时桥的超时/断连策略会兜住（fail-closed）。
        """
        self._dispatch_threadsafe(WebEvent(
            kind="approval_request",
            fields={"aid": aid, "action": action, "detail": detail,
                    "timeout": self.approvals.timeout},
            seq=None,
        ))

    def push_approval_missed(self, action: str, detail: str, reason: str) -> None:
        """审批因为"没人看"被拒 → 必须让用户**事后**知道发生了什么。

        注意磁盘上的 `approval` 行记的是 `source="user", granted=false`
        （由 tools 写），它看不出"因为断连"。所以这条 UI 事件是唯一的解释来源。
        """
        self._dispatch_threadsafe(WebEvent(
            kind="approval_missed",
            fields={"action": action, "detail": detail, "reason": reason,
                    "granted": False},
            seq=None,
        ))

    # ---------- 断连 ----------

    def consumer_attached(self) -> None:
        self.consumers += 1

    def consumer_detached(self) -> None:
        self.consumers = max(0, self.consumers - 1)
        if self.detached:
            # 消费者消失 = 审批请求再也没人看 → 唤醒并**拒绝**所有待决项。
            # 不这么做的话 threadpool 线程会一直占着（决策 H：十几个会话就把池子吃光），
            # 而且用户回来时看到的是一个 5 分钟前的"待审批"按钮。
            self.approvals.reject_all()

    # ---------- 重连 ----------

    def log_path_or_placeholder(self) -> str:
        return str(self.logger.path) if self.logger is not None else "(未开启会话日志)"

    def disk_events_after(self, seq: int) -> list[WebEvent]:
        """磁盘上 `seq` 之后的**落盘记录**（还原成 SSE 事件）。

        断线重连靠它做到"不丢已提交内容"：只看内存队列的话，断线期间落到磁盘、
        却被 detach 策略丢掉的事件就找不回来了。而磁盘本来就有全部真相——
        重新读一遍比维护一份内存影子状态便宜得多，也不会漂移。

        ⚠️ 中间行损坏时 `read_records` 会抛 `SessionLogError`（note3 I6：
        不允许静默跳过）。这里**故意不吞**——调用方（SSE 生成器）会把它转成
        一条显式的 `error` 事件；吞掉就等于"读到半截日志却假装完整"。
        """
        from agent.session_log import read_records

        if self.logger is None:
            return []
        events: list[WebEvent] = []
        for record in read_records(self.logger.path):
            if record.get("type") not in events_mod.DISK_EVENT_TYPES:
                continue
            if int(record.get("seq") or 0) <= seq:
                continue
            events.append(record_to_event(record))
        return events

    def replay_payload(self) -> dict:
        """已提交态（从磁盘重放）。没有 logger 时给一个空壳。"""
        payload, _ = self.replay_snapshot()
        return payload

    def replay_snapshot(self) -> tuple[dict, int]:
        """已提交态 **+ 它覆盖到的最大 seq**（决策 F / I 的续传基础）。

        📌 两样东西**必须一起返回**。分两次读（先读快照、再单独读一次
        "当前最大 seq"）会在两次读之间留一个窗口：那个窗口里写下的记录
        **既在快照里、又会被内存队列重新送来** → 前端看到重复消息，
        而且这种重复只在并发时偶发（又一个"偶发 + 不报错"的形态）。
        一次读、一次取 max，窗口就不存在了。
        """
        from agent.session_log import read_records, replay

        if self.logger is None:
            return ({"session_id": self.sid, "workspace": str(self.workspace),
                     "model": None, "messages": [], "summary": None,
                     "compressed_turns": 0, "dropped_upto_seq": 0,
                     "task_ends": [], "approvals": [], "unknown_lines": 0}, 0)
        records = read_records(self.logger.path)
        max_seq = max((int(r.get("seq") or 0) for r in records), default=0)
        return events_mod.replay_payload(replay(records)), max_seq


# ============================================================
# 注册表
# ============================================================

class SessionRegistry:
    """`sid → WebSession`，外加 `workspace_path → [active_sids]`（决策 D 的 C 方案）。

    内存态（决策 E）：**服务端重启 = 活跃会话丢失、历史仍可 replay**。
    这条写进 README，不假装持久。
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.sessions: dict[str, WebSession] = {}
        # key 一律是 `resolve()` 过的绝对路径（决策 D 的 📌）
        self._by_workspace: dict[Path, list[str]] = {}

    # ---------- workspace 派生（决策 D） ----------

    def resolve_workspace(self, name: str | None) -> Path:
        """把"工作区名字"解析成沙箱根。

        前端**不得**传任意路径：服务端 `resolve()` 后校验它必须在
        `WEB_WORKSPACE_ROOT` 之下，否则 400。
        """
        root = Path(self.cfg.WEB_WORKSPACE_ROOT).resolve()
        if not name:
            target = root
        else:
            p = Path(name)
            target = (p if p.is_absolute() else root / p).resolve()
        if not target.is_relative_to(root):
            raise WorkspaceNotAllowed(
                f"工作区 {name!r} 解析为 {target}，不在服务端允许的根 {root} 之下。"
                f"只能提交工作区**名字**（相对根），不能提交任意路径。"
            )
        if not target.is_dir():
            # 不自动 mkdir：打错一个字母就会静默建出一个空目录，
            # agent 在里面什么也看不到、也不报错——又一个"不报错的错误结论"
            raise WorkspaceNotAllowed(f"工作区目录不存在：{target}")
        return target

    def peers_of(self, workspace: Path) -> list[str]:
        """该 workspace 上**已有的**活跃会话（决策 D：key 已 resolve，可直接比）。"""
        return sorted(self._by_workspace.get(Path(workspace).resolve(), ()))

    # ---------- 建 / 取 / 关 ----------

    def create(self, workspace_name: str | None, *, llm_client=None) -> WebSession:
        loop = asyncio.get_running_loop()
        path = self.resolve_workspace(workspace_name)
        peers = self.peers_of(path)
        sid = new_session_id()

        session = WebSession(sid=sid, workspace=path, cfg=self.cfg,
                             loop=loop, shared_with=peers)

        # 组合根仍然是 main.build_session（**不新写"web 版 agent"**）：
        #   审批 → 桥的 sync callback（跑在 threadpool）
        #   输出 → session.out（推 SSE）
        #   审计 → session.sink（落盘 + 推 SSE）
        agent, logger = main_mod.build_session(
            workspace=path,
            approval_callback=session.approvals.callback,
            config=self.cfg,
            llm_client=llm_client,
            out=session.out,
            event_sink=session.sink,
            session_start_extra={
                # 决策 D 选 C：同名 workspace 是**显式共享**，把事实写进日志与 UI，
                # 不假装隔离（静默共享就是又一个"不报错的错误结论"）
                "shared": bool(peers),
                "shared_with": peers,
            },
        )
        session.agent = agent
        session.logger = logger
        if logger is not None:
            # listener 在建会话**之后**挂上：session_start 那一条不会推给 SSE，
            # 但前端首次连接时本来就会收到 `replay`（含 session_start 的派生结果），
            # 所以不需要为它开特例。
            logger.listener = session.on_record

        self.sessions[sid] = session
        self._by_workspace.setdefault(path, []).append(sid)
        return session

    def get(self, sid: str) -> WebSession:
        session = self.sessions.get(sid)
        if session is None:
            raise SessionNotFound(f"会话不存在（或已被关闭）：{sid}")
        return session

    async def close(self, sid: str) -> bool:
        session = self.sessions.pop(sid, None)
        if session is None:
            return False
        # 1. 待决审批全部拒绝（不留僵尸线程）
        session.approvals.reject_all()
        # 2. 从 workspace 索引里摘掉
        peers = self._by_workspace.get(session.workspace)
        if peers is not None:
            if sid in peers:
                peers.remove(sid)
            if not peers:
                del self._by_workspace[session.workspace]
        # 3. 停任务（显式 DELETE ≠ "客户端关页面"：前者用户是在说"结束这次会话"）
        if session.task is not None and not session.task.done():
            session.task.cancel()
            try:
                await session.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # 4. 排干并关日志（丢尾部记录 = 丢 commit 边界，正是 note3 §4.4 的意义）
        if session.logger is not None:
            # close() 是阻塞等待 writer 线程收工 → 搬出 loop，别卡住整个服务器
            await asyncio.to_thread(session.logger.close)
        return True

    async def close_all(self) -> int:
        sids = list(self.sessions)
        for sid in sids:
            await self.close(sid)
        return len(sids)
