"""events.py —— UI 事件模型 + SSE 序列化 + **有界**队列（决策 I / J）

三条只在文档里写清楚就会出错的事，这个模块就是它们的实现：

决策 I —— **存在两个序号空间**，只有落盘事件能用 `id:`：
  `seq` 由 `SessionLogger.emit` 在锁内分配，所以**只有被 emit 的记录才有 seq**
  （session_start / user / assistant / tool / state / clear / approval / task_end）。
  流式 delta **从不 emit**（note3 §4.3）→ delta 没有 seq。
  按 SSE 规范，**只有出现 `id:` 字段的块才更新浏览器的 last-event-ID 缓冲**，
  所以"delta 不带 id"不是漏写，而是用规范本身的机制声明"这里不可续传"。

决策 J —— 消费者会**消失**，所以队列**必须有界**，且按事件类分级丢：
  浏览器关页后任务继续跑完（§8 风险 8），队列没有消费者却在无限增长；
  delta 是每 token 一个事件，长任务单个僵尸会话攒到几十 MB 完全正常。
  注意这个不对称：`SessionLogger` 的队列也没有界，但它的消费者是
  **随 logger 同生共死的 writer 线程**；SSE 队列的消费者**随时会消失**。
  → **有"会消失的消费者"的队列必须限长。**

决策 A —— `_push_ui` **跑在 loop 线程**：跨线程只负责"把事件送到 loop"，
  "满了怎么办"必须在 loop 线程里显式决策（否则失败形态是 `QueueFull`
  变成 loop 上的 unhandled callback exception，事件**照样丢**且不报错）。
"""
from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# ------------------------------------------------------------
# 事件分类
# ------------------------------------------------------------

# 有落盘记录的记录类型：**只有它们带 `id:`**（决策 I）
DISK_EVENT_TYPES: frozenset[str] = frozenset({
    "session_start", "user", "assistant", "tool", "state", "task_end",
    "approval", "clear",
})

# 只有 UI 才有的事件：**一律没有 `id:`**
#   - 前 10 个来自 `CodingAgent._out` / `ToolContext.out`（W1 的输出通道）
#   - `approval_*` / `replay` / `lagged` / `error` 由 web 层自己产生
UI_ONLY_TYPES: frozenset[str] = frozenset({
    "delta", "stream_end", "empty_reply", "tool_call", "tool_parse_error",
    "max_iterations", "compress_failed", "task_cost", "log_path",
    "approval_prompt",
    "approval_request", "approval_missed", "replay", "lagged", "error",
})

# 唯一"可以随便丢"的事件类：delta（定稿文本已在磁盘上）。其余都不许丢。
DROPPABLE_KINDS: frozenset[str] = frozenset({"delta"})

# 落盘记录的"信封"字段：它们不进 UI 事件的 fields（seq 单独作 `id:`）
_ENVELOPE_KEYS: tuple[str, ...] = ("v", "seq", "ts", "type", "session_id")

# SSE 心跳：防中间层掐掉空闲连接（尤其是 Windows 上的代理/杀软）
HEARTBEAT = ": ping\n\n"


@dataclass
class WebEvent:
    """一条要送给浏览器的事件。

    `seq is None` ⇔ 这条事件**不可续传**（没有落盘记录）⇔ SSE 帧里**不写 `id:`**。
    """

    kind: str
    fields: dict = field(default_factory=dict)
    seq: int | None = None

    @property
    def has_id(self) -> bool:
        return self.seq is not None


def sse_frame(event: WebEvent) -> str:
    """把一个事件编成 SSE 帧。

    ⚠️ `data:` 必须是**单行** JSON：审批 detail 里有路径与换行
    （`run_command` 的 detail 就是两行），裸露的换行会直接把 SSE 分帧切断，
    前端只会看到"事件少了一条"，**不报错**。`json.dumps` 天然把 `\\n` 转成字面量，
    但仍然断言一次——这是"静默坏掉"的高危点。
    """
    payload = json.dumps({"type": event.kind, **event.fields}, ensure_ascii=False)
    assert "\n" not in payload and "\r" not in payload, "SSE data 必须是单行 JSON"
    lines = [f"event: {event.kind}"]
    if event.seq is not None:
        lines.append(f"id: {event.seq}")
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


def record_to_event(record: dict) -> WebEvent:
    """把一条落盘记录还原成 SSE 事件（`id:` = 它的 seq，决策 I）。"""
    fields = {k: v for k, v in record.items() if k not in _ENVELOPE_KEYS}
    return WebEvent(
        kind=str(record.get("type", "unknown")),
        fields=fields,
        seq=record.get("seq"),
    )


def parse_last_event_id(value: Any) -> int | None:
    """解析 `Last-Event-ID`（头 / query 均可）。非法值当成"没有"。"""
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------
# 有界队列 + 背压策略
# ------------------------------------------------------------

class EventQueue:
    """SSE 事件队列：**有界** + 分级丢 + 全程记账。

    为什么不用裸 `asyncio.Queue`：
      1. 它是**线程不安全**的（文档明写 "not thread-safe"），而事件来自
         主循环线程与 threadpool 线程两处；
      2. 更关键的是"满了怎么办"——实测过：投递侧会全部返回成功，而
         `QueueFull` 变成 loop 上的 unhandled callback exception，事件**照样丢**。
         → 所以"丢什么、丢多少"必须是**显式策略**，而且**丢了多少要可读**。

    策略（决策 J 的表）：
      - `delta`：可丢。detach 后**根本不入队**（省内存，还省掉每 token 一次的
        JSON 序列化）；未 detach 时队列满则丢**最旧**的 delta。
      - 落盘事件（user/assistant/tool/task_end/state/approval）：**不可丢**。
        它们天然有界（一条磁盘记录 = 一个事件），队列满时也绝不先丢它们。
      - `approval_request` 是**功能性**事件：丢了任务会白等满超时。
        所以断连时不是"丢"，而是**根本不进入等待**（见 approvals.py）。
    """

    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = max(1, int(maxsize))
        self._buf: deque[WebEvent] = deque()
        # 唤醒信号：push 只发生在 loop 线程（call_soon_threadsafe），
        # 与消费者在同一个线程里协作，因此 clear/check/await 三步之间不会被抢占
        self._ready = asyncio.Event()
        # 丢了多少（重连时用 `lagged` 事件告诉用户，绝不静默）
        self.dropped_deltas = 0
        self.dropped_events = 0
        # 改过队列的线程 id —— W-T11 断言的正是"这里只有一个线程"：
        # 换成裸 put_nowait（从 worker 线程直接改）时这个集合会多出一个 id，用例变红。
        self.mutating_threads: set[int] = set()

    # ---------- 只读视图 ----------

    def qsize(self) -> int:
        return len(self._buf)

    def empty(self) -> bool:
        return not self._buf

    def snapshot(self) -> list[WebEvent]:
        return list(self._buf)

    # ---------- 消费者（loop 线程） ----------

    async def get(self) -> WebEvent:
        while True:
            self._ready.clear()
            if self._buf:
                return self._buf.popleft()
            await self._ready.wait()

    def try_get(self) -> WebEvent | None:
        return self._buf.popleft() if self._buf else None

    # ---------- 生产者（**只允许 loop 线程**） ----------

    def push(self, event: WebEvent) -> bool:
        """入队。返回 False 表示这条被丢了（调用方**不该**据此静默）。

        ⚠️ 只允许在 loop 线程调用（决策 A）。跨线程请走
        `WebSession._dispatch_threadsafe`（它用 `call_soon_threadsafe` 把
        这个函数排到 loop 上）。
        """
        self.mutating_threads.add(threading.get_ident())
        if len(self._buf) >= self.maxsize:
            if not self._evict_one_droppable():
                # 队列里一条可丢的都没有 → 只能丢最旧的（一定是落盘事件）。
                # 这**不该发生**：maxsize 远大于"一条磁盘记录一个事件"的自然水位。
                # 真发生了也必须记账，而不是静默挤掉
                self._buf.popleft()
                self.dropped_events += 1
        self._buf.append(event)
        self._ready.set()
        return True

    def _evict_one_droppable(self) -> bool:
        """丢**最旧的可丢事件**（delta）。返回是否丢掉了。"""
        for i in range(len(self._buf)):
            if self._buf[i].kind in DROPPABLE_KINDS:
                del self._buf[i]
                self.dropped_deltas += 1
                return True
        return False


def replay_payload(state: Any) -> dict:
    """把 `ReplayedState` 摘成给前端的"已提交态"（决策 F/W4 的断线重连基础）。

    v1 **不做**"重放未定型的 delta"——那需要把 delta 落盘，正是 note3 反对的写放大。
    所以前端重连后拿到的历史**以一条 assistant 为单位**（定稿态），不是逐字重播。
    """
    return {
        "session_id": state.session_id,
        "workspace": state.workspace,
        "model": state.model,
        "messages": state.messages,
        "summary": state.summary,
        "compressed_turns": state.compressed_turns,
        "dropped_upto_seq": state.dropped_upto_seq,
        "task_ends": state.task_ends,
        "approvals": state.approvals,
        "unknown_lines": state.unknown_lines,
    }
