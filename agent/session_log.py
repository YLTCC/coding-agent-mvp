"""
session_log.py —— JSONL 会话日志（append-only + 状态行，不每轮全量 dump）

本模块是 note3.md §1–§4 的可执行实现。设计要点（照 note3 抄，别在这里改口径）：

1. **一行一条 JSON**，`newline="\\n"` + `ensure_ascii=False` + 一次 `write()` 写完一整行。
   三个都是硬性要求：前者防 Windows 把 LF 翻成 CRLF；中者省 3 倍体积；
   后者是"绝不写出半行 JSON"的根源（多次 write 才是交错的成因）。

2. **源事件 vs 派生状态**（note3 §0 的核心区分）：
   - 源事件（user/assistant/tool）：地面真值，重放时按 seq push 进 messages；
   - 派生状态（state 行里的 summary/compressed_turns/dropped_upto_seq）：
     重算要花模型调用且非确定 → **直接赋值，永不重算**。

3. **单一 seq 权威**：seq 只在 `emit()` 里、同一把锁的临界区内分配并入队，
   因此 `队列顺序 == seq 顺序 == 文件行顺序`，保证 I2（严格递增、无空洞、无重复）。
   写入由唯一的 writer 线程串行执行（§4.2）。

   ⚠️ 与 note3 §4.2 的偏离（有意为之，已记录）：note3 写的是"writer 任务 + asyncio.Queue
   + call_soon_threadsafe"。这里改成"writer 线程 + queue.Queue"，原因是：
     (a) 生产者可能在任何线程（审批事件来自 threadpool）；
     (b) `emit()` 需要在**没有运行中事件循环**时也能工作（构造即写 session_start、pytest 里同步建 logger）；
     (c) 省掉每个生产者身上的 call_soon_threadsafe 舞蹈。
   不变量没有一条被放松：单点分配 seq、FIFO 全序、一次 write 一行、commit 边界 flush 全部保留。
   将来 web 层若要 loop 原生背压，只需换掉 `self._q` 与 `_writer_loop`，其余接口不动。

4. **落盘时机 = commit 边界**：流式 delta 绝不落盘（note3 §4.3）。
   谁负责在边界调用 `emit` 是 agent.py 的事，本模块只管"来了就写"。
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# schema 版本，为将来格式演进留的逃生舱（note3 §1.3）
SCHEMA_VERSION = 1

# 源事件：不可再生的地面真值，重放时按序 push 进 messages
SOURCE_TYPES: tuple[str, ...] = ("user", "assistant", "tool")
# meta：会话元信息 / 不变式重置
META_TYPES: tuple[str, ...] = ("session_start", "clear")
# derived：可由源事件推出，但"推出来要花模型调用且非确定" → 快照
DERIVED_TYPES: tuple[str, ...] = ("state", "task_end")
# audit：不进上下文，只给审计 / 回放 UI / 成本看板
AUDIT_TYPES: tuple[str, ...] = ("approval",)
KNOWN_TYPES: tuple[str, ...] = SOURCE_TYPES + META_TYPES + DERIVED_TYPES + AUDIT_TYPES

# 这些行之后 flush 一次（低频且语义上是"检查点"，note3 §4.4）
# 其余行只进用户态缓冲——不每行 flush，更不每行 fsync（那才是写放大）
_FLUSH_TYPES: frozenset[str] = frozenset({"session_start", "state", "task_end", "clear"})

# 影响上下文语义的配置项，必须随 session_start 落盘：
# 半年后重放旧日志时 .env 早变了，不记就复原不出当时的裁剪行为（note3 §2.1）
CONFIG_SNAPSHOT_KEYS: tuple[str, ...] = (
    "HISTORY_WINDOW_TURNS",
    "HISTORY_COMPRESS_LAG",
    "SUMMARY_MAX_CHARS",
    "MAX_HISTORY_MESSAGES",
    "MAX_ITERATIONS",
    "MAX_TOOL_OUTPUT_CHARS",
)

# 消息里允许落盘的键（与 messages 的 dict 逐字同构，不改写 / 不截断 / 不归一）
_PAYLOAD_KEYS: tuple[str, ...] = ("content", "tool_calls", "tool_call_id")


class SessionLogError(Exception):
    """日志损坏且不可容忍（中间行坏 / 结构非法）时抛出。

    note3 I6：末行撕裂可以丢弃（append 的代价），但**中间行损坏必须报错**——
    静默跳过会得到"不报错的错误结论"，那比崩溃更贵。
    """


def _now_ms() -> int:
    """Unix epoch 毫秒（int：省空间、无时区歧义）"""
    return int(time.time() * 1000)


def new_session_id() -> str:
    """生成 URL / 文件名安全的会话 ID"""
    return uuid.uuid4().hex[:12]


def default_sessions_dir() -> Path:
    return Path("sessions")


def config_snapshot(cfg: Any) -> dict:
    """从 config 对象上摘出影响上下文语义的几个值。

    只记参数名而非全量环境变量——别把密钥写进日志（note3 §2.1）。
    """
    out: dict = {}
    for key in CONFIG_SNAPSHOT_KEYS:
        if hasattr(cfg, key):
            value = getattr(cfg, key)
            out[key] = str(value) if isinstance(value, Path) else value
    return out


def normalize_message(msg: dict) -> dict:
    """把 `content=None` 归一成 `""`（note3 §2.2）。

    必须在落盘前做：否则日志里 `null` 与 `""` 两种形态并存，重放要额外兼容。
    运行时的 content 本来就是字符串，这里只是把边界补齐。
    """
    if msg.get("content", "") is None:
        msg = dict(msg)
        msg["content"] = ""
    return msg


def payload_of(msg: dict) -> dict:
    """从 messages 里的一条 dict 摘出要落盘的载荷（role 由 type 隐含，可省）。

    不做任何加工：加工会让"重放态 != 运行态"。
    """
    out: dict = {}
    for key in _PAYLOAD_KEYS:
        if key in msg:
            out[key] = msg[key]
    return out


# ============================================================
# 写者：单点 seq 分配 + 单线程串行落盘
# ============================================================

class SessionLogger:
    """一个会话一个实例，对应一个 JSONL 文件（会话隔离靠文件隔离）。

    生命周期：`open_session(...)` 建实例并写 session_start → 反复 `emit(...)`
    → 会话结束 `close()`。
    """

    def __init__(
        self,
        path: str | Path,
        session_id: str | None = None,
        *,
        auto_start: bool = False,
    ) -> None:
        self.path = Path(path)
        self.session_id = session_id or new_session_id()

        self._q: "queue.Queue[tuple]" = queue.Queue()
        # seq 与入队在**同一临界区**内完成：否则并发下"队列顺序"可能背离"seq 顺序"
        self._lock = threading.Lock()
        self._seq = 0
        self._closed = False
        self._fh = None
        self._thread: threading.Thread | None = None

        # 观测用（测试 T22/T23 直接读这两个计数，不需要重新解析文件）
        self.rows_written = 0
        self.bytes_written = 0

        # 观察者（web 层用）：每条记录在**拿到 seq 之后**回调一次
        # `listener(seq, rtype, fields)`。
        #
        # 为什么挂在这里而不是让 web 层去包 `emit`：seq 是"记录的落盘编号"，
        # UI 侧要用它做 SSE 的 `id:`（决策 I：只有落盘事件才可续传）。
        # 只有在本类的临界区内才拿得到"seq 与这条记录"的对应关系。
        # 默认 None → 零开销、行为与加这个钩子之前完全一致。
        #
        # ⚠️ 约定：listener **不得调用 emit**（会产生自锁）。它就只该做
        #    "把 (seq, type, fields) 挪走"这种事（web 侧是 call_soon_threadsafe）。
        # ⚠️ 它在**任何线程**里被调用（emit 的调用方可能是 threadpool）。
        self.listener: "Callable[[int, str, dict], None] | None" = None

        if auto_start:
            self.start()

    # ---------- 生命周期 ----------

    def start(self) -> "SessionLogger":
        """打开文件（append）并启动 writer 线程。"""
        if self._thread is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" 必须显式：否则 Windows 上 \n 被翻译成 \r\n（_fix_crcrlf 的老账）
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")
        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"session-log-{self.session_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self, timeout: float = 10.0) -> bool:
        """把队列排干、flush、关文件、收线程。幂等。"""
        with self._lock:
            if self._closed:
                return True
            self._closed = True
        done = threading.Event()
        self._q.put(("close", done))
        ok = done.wait(timeout)
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        return ok

    def flush(self, timeout: float = 10.0) -> bool:
        """排干队列并 flush 到 OS（不 fsync）。返回是否在超时内完成。"""
        if self._closed:
            return True
        done = threading.Event()
        self._q.put(("flush", done))
        return done.wait(timeout)

    # ---------- 唯一的写入接口 ----------

    def emit(self, rtype: str, **fields: Any) -> int:
        """投递一条记录，返回它拿到的 seq。

        任何线程都可以调。生产者只给**载荷**，信封（v/ts/session_id）由 writer 补。
        """
        with self._lock:
            if self._closed:
                raise SessionLogError(
                    f"会话日志已关闭（session_id={self.session_id}），不能再写入"
                )
            self._seq += 1
            seq = self._seq
            # put 与 seq 分配同锁，保证队列顺序 == seq 顺序
            self._q.put((seq, rtype, fields))
            # listener 在**锁内**调：这样"UI 收到的顺序"与"文件的 seq 顺序"一致。
            # 放到锁外则两个线程可能乱序回调（A 拿到 seq=5、B 拿到 seq=6，
            # 却是 B 先回调），UI 上会出现 seq 6 排在 5 前面。
            # 代价是 listener 绝不能再调 emit（非重入锁，会自锁）——已在 __init__ 写明。
            if self.listener is not None:
                self.listener(seq, rtype, fields)
        return seq

    # ---------- writer 线程 ----------

    def _writer_loop(self) -> None:
        """唯一消费者：串行写盘，任何时刻只有一个线程碰文件句柄。"""
        while True:
            item = self._q.get()
            try:
                if isinstance(item[0], int):
                    seq, rtype, fields = item
                    self._write_one(seq, rtype, fields)
                    continue
                kind, done = item
                if kind == "flush":
                    if self._fh is not None:
                        self._fh.flush()
                    done.set()
                    continue
                if kind == "close":
                    if self._fh is not None:
                        self._fh.flush()
                        self._fh.close()
                        self._fh = None
                    done.set()
                    return
            finally:
                self._q.task_done()

    def _write_one(self, seq: int, rtype: str, fields: dict) -> None:
        record = {
            "v": SCHEMA_VERSION,
            "seq": seq,
            "ts": _now_ms(),
            "type": rtype,
            "session_id": self.session_id,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False)
        # 一次 write() 写完一整行 —— 拆成多次 write 才是"半行 JSON"的成因
        self._fh.write(line + "\n")
        if rtype in _FLUSH_TYPES:
            self._fh.flush()
        self.rows_written += 1
        self.bytes_written += len(line.encode("utf-8")) + 1

    # ---------- 便捷方法 ----------

    @classmethod
    def open_session(
        cls,
        directory: str | Path | None = None,
        *,
        system_prompt: str,
        workspace: str = "",
        model: str = "",
        config: dict | None = None,
        session_id: str | None = None,
        extra: dict | None = None,
    ) -> "SessionLogger":
        """建文件 + 写 session_start（必为文件首行，不可省）。

        system_prompt 必须落盘：它是源事件，不来自任何一次模型调用，
        不落盘重放就推不出 messages[0]。

        extra：追加进 session_start 载荷的字段（web 层用它记同名 workspace 的
        `shared` / `shared_with`，决策 D 选 C）。缺省 None → 载荷与加这个参数
        之前**逐字一致**。
        """
        directory = Path(directory) if directory is not None else default_sessions_dir()
        session_id = session_id or new_session_id()
        logger = cls(directory / f"{session_id}.jsonl", session_id)
        logger.start()
        payload: dict = {
            "system_prompt": system_prompt,
            "workspace": str(workspace),
            "model": model,
            "config": config or {},
        }
        if extra:
            payload.update(extra)
        logger.emit("session_start", **payload)
        return logger


# ============================================================
# 读者：容忍撕裂的尾巴，不容忍中间损坏
# ============================================================

def read_records(path: str | Path, tolerate_torn_tail: bool = True) -> list[dict]:
    """读回全部记录（按文件顺序）。

    - 文件不存在 / 空文件 → `[]`
    - 末行撕裂（写一半就崩）→ 丢弃该行，正常返回（note3 I6）
    - **中间任何行**损坏 / 空行 / 非法结构 → 抛 SessionLogError
    """
    p = Path(path)
    if not p.exists():
        return []
    raw = p.read_bytes().decode("utf-8")
    if raw == "":
        return []

    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]          # 正常收尾的那个换行产生的空串

    records: list[dict] = []
    errors: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            errors.append((idx, "空行"))
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append((idx, f"JSON 解析失败：{e}"))
            continue
        if not isinstance(rec, dict):
            errors.append((idx, f"不是 JSON 对象（{type(rec).__name__}）"))
            continue
        records.append(rec)

    if errors:
        # 只有"最后一行"失败才允许容忍——那正是 append 被中断的形态
        only_torn_tail = len(errors) == 1 and errors[0][0] == len(lines) - 1
        if not (tolerate_torn_tail and only_torn_tail):
            idx, why = errors[0]
            raise SessionLogError(
                f"会话日志第 {idx + 1} 行损坏（{why}）：{p}。"
                f"中间行损坏不允许静默跳过（note3 I6）——静默跳过会得到"
                f"'不报错的错误结论'，比直接报错更贵。"
            )
    return records


# ============================================================
# 重放：由日志重建派生状态，零模型调用
# ============================================================

@dataclass
class ReplayedState:
    """重放结果。

    只承诺三样东西与运行态逐字一致（note3 I1）：
    `messages` / `summary` / `compressed_turns`。
    其余字段是给 UI 与测试的便利视图，不参与不变量。
    """

    messages: list[dict] = field(default_factory=list)
    summary: str | None = None
    compressed_turns: int = 0
    dropped_upto_seq: int = 0

    session_id: str | None = None
    system_prompt: str | None = None
    workspace: str | None = None
    model: str | None = None
    config: dict = field(default_factory=dict)

    unknown_lines: int = 0
    cleared_count: int = 0
    source_events: int = 0
    approvals: list[dict] = field(default_factory=list)
    task_ends: list[dict] = field(default_factory=list)


def _message_from_record(rec: dict) -> dict:
    """把源事件记录还原成 messages 里的一条 dict（role 由 type 还原）。"""
    msg: dict = {"role": rec.get("type")}
    for key in _PAYLOAD_KEYS:
        if key in rec:
            value = rec[key]
            if key == "content" and value is None:
                value = ""          # 兼容早期日志里的 null（note3 §2.2）
            msg[key] = value
    msg.setdefault("content", "")
    return msg


def _prune_by_watermark(
    messages: list[dict], seqs: list[int], dropped: int
) -> tuple[list[dict], list[int]]:
    """丢掉 seq <= dropped 的源事件。index 0（system）永远保留。"""
    if not messages:
        return messages, seqs
    keep = [0] + [i for i in range(1, len(messages)) if seqs[i] > dropped]
    return [messages[i] for i in keep], [seqs[i] for i in keep]


def replay(records: Iterable[dict]) -> ReplayedState:
    """单遍重放（水位单调，故无需两遍）。

    grep 关键约定（note3 §3.2）：**收尾必须复用运行时的同一组纯函数**
    （`_drop_leading_orphan_tools`）。被物理裁掉的 `assistant(tool_calls)`
    留下的"孤儿 tool"消息，其 seq > dropped_upto_seq，水位表达不了它——
    在重放器里另写一遍裁剪语义必然与运行时分叉。这是本规范最容易被违反、
    也最贵的一条，所以这里直接 import 过来用。
    """
    from .agent import _drop_leading_orphan_tools  # 延迟导入：避免 agent <-> session_log 循环依赖

    state = ReplayedState()
    messages: list[dict] = []
    seqs: list[int] = []
    summary: str | None = None
    compressed_turns = 0
    dropped = 0

    for rec in sorted(records, key=lambda r: r.get("seq") or 0):
        rtype = rec.get("type")

        if rtype == "session_start":
            messages = [{"role": "system", "content": rec.get("system_prompt") or ""}]
            seqs = [rec.get("seq") or 0]
            summary, compressed_turns, dropped = None, 0, 0
            state.session_id = rec.get("session_id")
            state.system_prompt = rec.get("system_prompt")
            state.workspace = rec.get("workspace")
            state.model = rec.get("model")
            state.config = rec.get("config") or {}

        elif rtype == "clear":
            # 回到 session_start 刚写完的状态；仍是同一文件里的一行
            messages = messages[:1]
            seqs = seqs[:1]
            summary, compressed_turns, dropped = None, 0, 0
            state.cleared_count += 1

        elif rtype == "state":
            # 派生状态：整体覆盖，**绝不重算**（重算要花模型且结果非确定）
            summary = rec.get("summary")
            compressed_turns = int(rec.get("compressed_turns") or 0)
            dropped = int(rec.get("dropped_upto_seq") or 0)
            messages, seqs = _prune_by_watermark(messages, seqs, dropped)

        elif rtype in SOURCE_TYPES:
            if int(rec.get("seq") or 0) <= dropped:
                continue            # 水位之下：已被裁剪，内存态里不留
            messages.append(_message_from_record(rec))
            seqs.append(rec.get("seq") or 0)
            state.source_events += 1

        elif rtype == "approval":
            state.approvals.append({
                "seq": rec.get("seq"),
                "action": rec.get("action"),
                "detail": rec.get("detail"),
                "granted": rec.get("granted"),
                "source": rec.get("source"),
            })

        elif rtype == "task_end":
            state.task_ends.append(dict(rec))

        else:
            # 未知 type：向前兼容，跳过但**记账**（note3 T21）
            state.unknown_lines += 1

    if messages:
        messages = messages[:1] + _drop_leading_orphan_tools(messages[1:])

    state.messages = messages
    state.summary = summary
    state.compressed_turns = compressed_turns
    state.dropped_upto_seq = dropped
    return state


def replay_file(path: str | Path, tolerate_torn_tail: bool = True) -> ReplayedState:
    """读文件 + 重放的一步版（UI / 测试常用）。"""
    return replay(read_records(path, tolerate_torn_tail=tolerate_torn_tail))


def make_sink(logger: SessionLogger) -> Callable[[str, dict], None]:
    """给 tools 层用的"事件接收器"：把 (type, fields) 投进会话日志。

    tools 不直接依赖本模块（它只认一个 `Callable[[str, dict], None]`），
    这样审批审计与工具执行层解耦。
    """
    def sink(rtype: str, fields: dict) -> None:
        logger.emit(rtype, **fields)

    return sink
