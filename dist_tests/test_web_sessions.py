"""test_web_sessions.py —— 注册表级验收（不需要起服务器）

对应 webtodolist.md §6 的：
  W-T10 断连背压：delta 不入队、队列有界、重连收到 lagged 且计数正确
  W-T11 sink 线程安全：从非 loop 线程调 sink，事件全部到达且不丢
  W-T12 审计日志的**按路径**守卫（决策 K 第 2 条）：含两条反向断言
  以及决策 D 的"同名 workspace = 显式共享"标记、决策 K 第 1 条的启动断言

跑法：python -m pytest test_web_sessions.py -q
"""
import asyncio
import json
import threading
from pathlib import Path

import pytest

from agent import tools as tools_mod
from agent.config import Config
from agent.session_log import read_records
from agent.tools import ToolContext

from conftest import ScriptedLLM
from web.app import _sse_stream
from web.session_registry import SessionRegistry, prepare_web_config


async def _frames_until(gen, kind: str, timeout: float = 3.0):
    """驱动 SSE 生成器，收集到指定 kind 为止（超时即返回已收到的）。"""
    frames = []
    while True:
        try:
            frame = await asyncio.wait_for(gen.__anext__(), timeout)
        except (asyncio.TimeoutError, StopAsyncIteration):
            break
        frames.append(frame)
        if frame.startswith(f"event: {kind}\n"):
            break
    return frames


def _kinds(frames):
    return [f.split("\n", 1)[0].removeprefix("event: ") for f in frames]


def _data_of(frame: str) -> dict:
    for line in frame.split("\n"):
        if line.startswith("data: "):
            return json.loads(line[len("data: "):])
    return {}


# ============================================================
# 决策 K 第 1 条：启动断言（审计目录必须在工作区之外）
# ============================================================

def test_prepare_web_config_moves_session_dir_out_of_workspace(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.delenv("AGENT_SESSION_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "la"))

    cfg = Config()
    cfg.WEB_WORKSPACE_ROOT = root
    target = prepare_web_config(cfg)

    assert target == (tmp_path / "la" / "coding-agent" / "sessions").resolve()
    assert not target.is_relative_to(root)
    assert cfg.SESSION_DIR == target


def test_prepare_web_config_refuses_to_start_when_inside_workspace(tmp_path, monkeypatch):
    """**只打 warning 等于没做**（警告会被忽略、日志会滚走）→ 必须启动失败。"""
    root = tmp_path / "root"
    (root / "sessions").mkdir(parents=True)
    monkeypatch.setenv("AGENT_SESSION_DIR", str(root / "sessions"))

    cfg = Config()
    cfg.WEB_WORKSPACE_ROOT = root
    assert cfg.SESSION_DIR.is_relative_to(root)

    with pytest.raises(RuntimeError) as excinfo:
        prepare_web_config(cfg)
    message = str(excinfo.value)
    assert "拒绝启动" in message
    assert "AGENT_SESSION_DIR" in message          # 必须给出明确修法


# ============================================================
# 决策 D 选 C：同名 workspace 是**显式共享**，不假装隔离
# ============================================================

async def test_same_workspace_is_marked_shared(web_cfg):
    registry = SessionRegistry(web_cfg)
    try:
        first = registry.create(None, llm_client=ScriptedLLM([]))
        second = registry.create("", llm_client=ScriptedLLM([]))     # 同一个目录

        assert first.shared_with == []
        assert second.shared_with == [first.sid]

        start_first = read_records(first.logger.path)[0]
        start_second = read_records(second.logger.path)[0]
        assert start_first["type"] == "session_start"
        assert start_first["shared"] is False
        assert start_first["shared_with"] == []
        assert start_second["shared"] is True
        assert start_second["shared_with"] == [first.sid]
        # W-T2 ② 只断言"标记出现"，**不断言文件隔离**——因为根本不可能隔离
    finally:
        await registry.close_all()


def test_workspace_resolution_uses_resolved_path_as_key(web_cfg):
    """📌 key 必须先 resolve 再比：否则 Windows 大小写/短名/相对路径会把
    同一个目录判成两个 → **静默共享照旧**，而 UI 却标着"独占"（标记变成谎话）。
    """
    registry = SessionRegistry(web_cfg)
    root: Path = web_cfg.WEB_WORKSPACE_ROOT
    assert registry.resolve_workspace(None) == root.resolve()
    assert registry.resolve_workspace("") == root.resolve()
    assert registry.resolve_workspace(".") == root.resolve()
    assert registry.resolve_workspace("./") == root.resolve()


def test_workspace_outside_root_is_rejected(web_cfg):
    """决策 D：前端只能提交**名字**，不能提交任意路径。"""
    from web.session_registry import WorkspaceNotAllowed

    registry = SessionRegistry(web_cfg)
    with pytest.raises(WorkspaceNotAllowed):
        registry.resolve_workspace("../..")
    with pytest.raises(WorkspaceNotAllowed):
        registry.resolve_workspace(str(Path.home()))
    # 打错一个字母不该**静默建目录**（那样 agent 在里面什么也看不到，还不报错）
    with pytest.raises(WorkspaceNotAllowed):
        registry.resolve_workspace("typo-dir")


# ============================================================
# W-T10：断连背压（决策 J）
# ============================================================

async def test_W_T10_detached_session_does_not_queue_deltas(web_cfg):
    web_cfg.WEB_QUEUE_MAXSIZE = 8
    registry = SessionRegistry(web_cfg)
    session = registry.create(None, llm_client=ScriptedLLM([]))
    try:
        await asyncio.sleep(0.02)                      # 让建会话时投的 log_path 落地
        while session.events.try_get() is not None:    # 排空，免得污染计数
            pass
        session.consumer_attached()
        assert not session.detached
        session.consumer_detached()          # 浏览器关页
        assert session.detached

        for i in range(50):                  # 任务继续跑（§8 风险 8：不硬中断）
            session.out("delta", text=str(i), first=(i == 0))
        for i in range(20):                  # 落盘事件仍然要留
            session.on_record(100 + i, "assistant", {"content": str(i)})
        await asyncio.sleep(0.05)

        # delta **根本不入队**（省内存，还省掉每 token 一次的 JSON 序列化）
        assert session.events.dropped_deltas == 50
        # 队列**有界**：不随 token 数增长
        assert session.events.qsize() == 8
        assert all(e.kind != "delta" for e in session.events.snapshot())
        # 落盘事件实在放不下时必须**记账**（绝不静默挤掉）
        assert session.events.dropped_events == 12
    finally:
        await registry.close_all()


async def test_W_T10_reconnect_reports_lagged_count(web_cfg):
    web_cfg.WEB_QUEUE_MAXSIZE = 8
    registry = SessionRegistry(web_cfg)
    session = registry.create(None, llm_client=ScriptedLLM([]))
    gen = None
    try:
        session.consumer_attached()
        session.consumer_detached()
        for i in range(37):
            session.out("delta", text=str(i))
        await asyncio.sleep(0.05)
        assert session.events.dropped_deltas == 37

        gen = _sse_stream(session, None)      # 重连（全新连接）
        frames = await _frames_until(gen, "lagged")
        kinds = _kinds(frames)
        assert kinds[0] == "replay"           # 先给已提交态
        assert "lagged" in kinds

        lagged = _data_of([f for f in frames if f.startswith("event: lagged")][0])
        assert lagged["dropped_deltas"] == 37  # 计数必须正确，且说出来
        assert "37" in lagged["detail"]
        assert session.events.dropped_deltas == 0   # 报过就清零，不重复报
    finally:
        if gen is not None:
            await gen.aclose()
        await registry.close_all()


async def test_W_T10_detach_happens_when_stream_is_closed(web_cfg):
    registry = SessionRegistry(web_cfg)
    session = registry.create(None, llm_client=ScriptedLLM([]))
    try:
        gen = _sse_stream(session, None)
        await _frames_until(gen, "replay")
        assert session.consumers == 1
        await gen.aclose()
        # 消费者消失 → 脱钩（决策 J）+ 唤醒并拒绝所有待决审批（决策 C.3）
        assert session.consumers == 0
        assert session.detached
    finally:
        await registry.close_all()


# ============================================================
# W-T11：sink 线程安全（决策 A）
# ============================================================

async def test_W_T11_sink_from_worker_thread_delivers_everything(web_cfg):
    """从**非 loop 线程**调 sink：事件必须全部到达、不丢，且 seq 单调。

    `_emit_event` 是同步直调，而它的上游 `_ask_approval` 跑在 threadpool 里
    —— 所以"跨线程投递"不是理论洁癖，是这条链路的日常。
    """
    registry = SessionRegistry(web_cfg)
    session = registry.create(None, llm_client=ScriptedLLM([]))
    loop_tid = threading.get_ident()
    try:
        def worker():
            for i in range(30):
                session.sink("approval", {"action": "写入文件", "detail": f"f{i}.txt",
                                          "granted": True, "source": "user"})

        await asyncio.to_thread(worker)
        await asyncio.sleep(0.05)

        received = []
        while True:
            event = session.events.try_get()
            if event is None:
                break
            received.append(event)
        approvals = [e for e in received if e.kind == "approval"]
        assert len(approvals) == 30, f"丢了事件：只到 {len(approvals)} 条"

        seqs = [e.seq for e in approvals]
        assert seqs == sorted(seqs) and len(set(seqs)) == 30, "seq 不是严格递增的"

        # 队列的**变更**只发生在 loop 线程：把 call_soon_threadsafe 换成
        # 裸 put_nowait（从 worker 线程直接改队列）时，这里会多出一个线程 id
        assert session.events.mutating_threads == {loop_tid}, (
            "队列被非 loop 线程改过 —— asyncio 的队列不是线程安全的，"
            "这种误用只会表现为'偶发少一条事件'，不报错"
        )
    finally:
        await registry.close_all()


async def test_W_T11_sink_signature_matches_tools_calling_convention(web_cfg):
    """`_emit_event` 用**位置参数**调 `sink(rtype, fields)`（tools.py）。

    签名写成 `sink(rtype, **fields)` 会 TypeError，而这个异常会被
    `_emit_event` 的 `except: pass` **吞掉** → 审计静默消失。
    """
    registry = SessionRegistry(web_cfg)
    session = registry.create(None, llm_client=ScriptedLLM([]))
    try:
        await asyncio.sleep(0.02)
        while session.events.try_get() is not None:
            pass
        emitted = []
        ctx = ToolContext(workspace=session.workspace, config=web_cfg,
                          event_sink=session.sink)
        tools_mod._emit_event("approval", ctx=ctx, action="写入文件",
                              detail="x", granted=True, source="user")
        await asyncio.sleep(0.02)
        while True:
            event = session.events.try_get()
            if event is None:
                break
            emitted.append(event)
        assert [e.kind for e in emitted] == ["approval"]
        assert emitted[0].seq is not None, "审计事件应当同时落盘（带 seq）"
    finally:
        await registry.close_all()


async def test_W_T11_sink_still_pushes_when_logging_disabled(web_cfg):
    """关掉 SESSION_LOG_ENABLED 时没有 logger：审计仍然要推到 UI（只是没有 seq）。"""
    web_cfg.SESSION_LOG_ENABLED = False
    registry = SessionRegistry(web_cfg)
    session = registry.create(None, llm_client=ScriptedLLM([]))
    try:
        assert session.logger is None
        session.sink("approval", {"action": "写入文件", "detail": "x",
                                  "granted": False, "source": "user"})
        await asyncio.sleep(0.02)
        event = session.events.try_get()
        assert event is not None and event.kind == "approval"
        assert event.seq is None            # 没有磁盘记录 → 不写 `id:`
    finally:
        await registry.close_all()


# ============================================================
# W-T12：审计日志的**按路径**守卫（决策 K 第 2 条）
# ============================================================

@pytest.fixture
def audit_ws(tmp_path):
    """一个工作区：里面既有**审计目录**，也有**用户自己的 jsonl 数据**。"""
    ws = tmp_path / "ws"
    (ws / "sessions").mkdir(parents=True)
    (ws / "sessions" / "a1b2c3d4e5f6.jsonl").write_text(
        '{"seq": 1, "type": "user", "content": "别人的对话"}\n', encoding="utf-8")
    (ws / "train_data.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    (ws / "notes.md").write_text("hello\n", encoding="utf-8")
    return ws


def _audit_ctx(ws, cfg, **kw):
    cfg.SESSION_DIR = ws / "sessions"       # 故意让它待在沙箱里（CLI 的真实形态）
    return cfg, ToolContext(workspace=ws, config=cfg, **kw)


def test_W_T12_reading_audit_log_is_refused_with_truthful_reason(audit_ws, web_cfg):
    events_append: list = []

    def sink(rtype, fields):        # 与 tools._emit_event 的**位置参数**调用严格同形
        events_append.append((rtype, fields))

    cfg, ctx = _audit_ctx(audit_ws, web_cfg, event_sink=sink)
    result = tools_mod.t_read_file({"path": "sessions/a1b2c3d4e5f6.jsonl"}, ctx)

    assert "审计日志" in result, result
    # 文案必须**说真话**：套用凭据类那句是撒谎（模型会据此推断"里面有密钥"）
    assert "凭据" not in result and "密钥" not in result
    assert "别人的对话" not in result
    # 拒绝要留 policy 审计（这类拒绝**根本没打扰用户**，与"用户点了拒绝"不同）
    assert events_append and events_append[0][0] == "approval"
    assert events_append[0][1]["source"] == "policy"
    assert "审计" in events_append[0][1]["action"]


def test_W_T12_user_jsonl_is_not_collateral_damage(audit_ws, web_cfg):
    """反向断言：`train_data.jsonl` **不是**审计日志，必须照读不误。

    这条正是"把 `*.jsonl` 加进 `_SENSITIVE_FILE_PATTERNS`"被否决的原因：
    名字型模式写不出"只拦 sessions 目录"，落地必然误伤用户自己的数据。
    """
    _cfg, ctx = _audit_ctx(audit_ws, web_cfg)
    assert '{"x": 1}' in tools_mod.t_read_file({"path": "train_data.jsonl"}, ctx)
    assert "hello" in tools_mod.t_read_file({"path": "notes.md"}, ctx)


def test_W_T12_guard_follows_configured_session_dir(audit_ws, web_cfg):
    """反向断言：`SESSION_DIR` 被**误配**时照样拦得住。

    判别式跟着配置走（`_cfg(ctx).SESSION_DIR`），而不是硬编码"名字叫 sessions 的目录"。
    """
    moved = audit_ws / "logs" / "audit"
    moved.mkdir(parents=True)
    (moved / "aa.jsonl").write_text('{"seq": 1}\n', encoding="utf-8")
    web_cfg.SESSION_DIR = moved
    ctx = ToolContext(workspace=audit_ws, config=web_cfg)

    # 新的审计目录被拦
    assert "审计日志" in tools_mod.t_read_file({"path": "logs/audit/aa.jsonl"}, ctx)
    # 而原来的 sessions/ 现在**只是普通目录**了（配置说了算，不是名字说了算）
    result = tools_mod.t_read_file({"path": "sessions/a1b2c3d4e5f6.jsonl"}, ctx)
    assert "审计日志" not in result


def test_W_T12_write_entry_is_refused_before_asking_the_user(audit_ws, web_cfg):
    """写入口必须在 `_ask_approval` **之前**就拒。

    否则用户要为一次必然失败的写点一次"允许"——审批窗口里出现"点了允许
    但什么都没发生"，用户会开始怀疑自己的操作。
    """
    approvals: list[str] = []
    cfg, ctx = _audit_ctx(
        audit_ws, web_cfg, approval_callback=lambda a, d: approvals.append(d) or True)

    out = tools_mod.t_write_file(
        {"path": "sessions/a1b2c3d4e5f6.jsonl", "content": "篡改"}, ctx)
    assert "审计日志" in out
    assert approvals == [], "用户被白问了一次"

    out = tools_mod.t_edit_file(
        {"path": "sessions/a1b2c3d4e5f6.jsonl", "old_string": "1", "new_string": "2"}, ctx)
    assert "审计日志" in out
    assert approvals == []

    # 文件没被动过
    assert "别人的对话" in (audit_ws / "sessions" / "a1b2c3d4e5f6.jsonl").read_text(
        encoding="utf-8")


def test_W_T12_all_read_channels_are_guarded(audit_ws, web_cfg):
    """六个 `_is_sensitive_file` 调用点 + 两个写入口都要过这道守卫。"""
    _cfg, ctx = _audit_ctx(audit_ws, web_cfg)

    assert "审计日志" in tools_mod.t_list_dir({"path": "sessions"}, ctx)
    assert "审计日志" in tools_mod.t_grep({"pattern": "对话", "path": "sessions"}, ctx)
    assert "审计日志" in tools_mod.t_glob(
        {"pattern": "*.jsonl", "path": "sessions/a1b2c3d4e5f6.jsonl"}, ctx)
    # 走目录递归时：审计目录整棵子树被跳过（连"存在性"都不暴露）
    listing = tools_mod.t_list_dir({"path": "."}, ctx)
    assert "sessions" not in listing
    assert "train_data.jsonl" in listing        # 用户数据不受影响

    grep_all = tools_mod.t_grep({"pattern": "x", "path": "."}, ctx)
    assert "sessions" not in grep_all
    assert "train_data.jsonl" in grep_all

    glob_all = tools_mod.t_glob({"pattern": "**/*.jsonl", "path": "."}, ctx)
    assert "sessions/" not in glob_all
    assert "train_data.jsonl" in glob_all


def test_W_T12_run_command_path_arguments_are_guarded(audit_ws, web_cfg):
    _cfg, ctx = _audit_ctx(audit_ws, web_cfg)
    reason = tools_mod._check_command_safety(
        "type sessions\\a1b2c3d4e5f6.jsonl", ctx)
    assert reason and "审计日志" in reason
