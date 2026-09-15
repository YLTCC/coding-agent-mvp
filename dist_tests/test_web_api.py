"""test_web_api.py —— 端到端验收：**真 uvicorn 服务器** + httpx 客户端

为什么不挂 ASGITransport：实测它会把整个响应缓冲下来（`_probe_asgi_stream.py`），
SSE 这种"永不结束"的响应会直接卡死测试。而真服务器还顺带给出三样只在这一层
才能验的东西：
  * **审批 POST 与 loop 同不同线程**（决策 C.2：写成 `def` 就会落进 threadpool）；
  * SSE 真的分帧、真的能断线重连；
  * 关服 / 断连时消费者计数与审批拒绝的真实时序。

对应 §6：W-T1 / W-T2 / W-T3b② / W-T4 / W-T5 / W-T6 / W-T7 / W-T9

跑法：python -m pytest test_web_api.py -q
"""
import collections
import asyncio
import json
import threading
import time

import httpx
import pytest
import uvicorn

from agent.session_log import read_records

from conftest import GatedLLM, ScriptedLLM, parse_frame, text_chunk, tool_chunk
from web.app import create_app
from web.events import DISK_EVENT_TYPES


class Harness:
    """把 app 跑在真 uvicorn 上（独立线程 + 独立事件循环）。

    `llm_factory` 每次从队列里取一份"脚本"，于是每个会话拿到**自己的**假模型
    —— 这正是"并发不串台"要证明的事。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.models: "collections.deque" = collections.deque()

        def factory():
            return self.models.popleft() if self.models else ScriptedLLM([])

        self.app = create_app(cfg, llm_factory=factory)
        self._server = uvicorn.Server(
            uvicorn.Config(self.app, host="127.0.0.1", port=0, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    # ---------- 生命周期 ----------

    def __enter__(self):
        self._thread.start()
        for _ in range(1000):
            if self._server.started:
                break
            time.sleep(0.01)
        assert self._server.started, "uvicorn 没起来"
        port = self._server.servers[0].sockets[0].getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        return self

    def __exit__(self, *_exc):
        self._server.should_exit = True
        self._thread.join(10)

    # ---------- 工具 ----------

    def queue_model(self, model):
        self.models.append(model)
        return self

    def script(self, *rounds):
        return self.queue_model(ScriptedLLM(list(rounds)))

    def session(self, sid):
        return self.app.state.registry.get(sid)


@pytest.fixture
def harness(web_cfg):
    with Harness(web_cfg) as h:
        yield h


# ============================================================
# SSE 采集器
# ============================================================

def _data(frame: str) -> dict:
    for line in frame.split("\n"):
        if line.startswith("data: "):
            return json.loads(line[len("data: "):])
    return {}


async def collect(client, sid, text=None, *, approve=True, stop=("task_cost",),
                  last_event_id=None, timeout=30.0, first_round_only=False):
    """开一条 SSE 流（可选：顺手派个任务），收帧到 stop 里的 kind 为止。

    `first_round_only=True` 时收到第一个非 delta 事件就断开——用来造"半路断线"。
    """
    frames: list[str] = []
    url = f"/api/sessions/{sid}/stream"
    if last_event_id is not None:
        url += f"?last_event_id={last_event_id}"

    async with client.stream("GET", url) as resp:
        assert resp.status_code == 200, f"{resp.status_code}"
        assert "charset=utf-8" in resp.headers.get("content-type", "")
        if text is not None:
            posted = await client.post(f"/api/sessions/{sid}/messages", json={"text": text})
            assert posted.status_code == 202, posted.text

        buf = ""
        deadline = time.time() + timeout
        async for chunk in resp.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                raw, buf = buf.split("\n\n", 1)
                if not raw.strip():
                    continue
                if raw.startswith(":"):          # 心跳注释行
                    continue
                frames.append(raw)
                kind = raw.split("\n", 1)[0].removeprefix("event: ")
                if approve and kind == "approval_request":
                    aid = _data(raw)["aid"]
                    decided = await client.post(
                        f"/api/sessions/{sid}/approvals/{aid}", json={"allow": True})
                    assert decided.status_code == 200, decided.text
                if kind in stop:
                    return frames
                if first_round_only and kind not in ("delta", "stream_end"):
                    return frames
            if time.time() > deadline:
                kinds = [f.split("\n", 1)[0] for f in frames]
                raise AssertionError(f"SSE 超时；已收到 {kinds}")
    return frames


def _contains_path(text: str, path) -> bool:
    """宽松比对：忽略 JSON 转义的反斜杠、分隔符方向与大小写（Windows 路径）。"""
    flat = text.replace("\\\\", "\\").replace("/", "\\").lower()
    return str(path).replace("/", "\\").lower() in flat


# ============================================================
# W-T1：单会话全流程
# ============================================================

async def test_W_T1_single_session_full_flow(harness):
    harness.script(
        [tool_chunk("c1", "write_file", {"path": "out.txt", "content": "hi"})],
        [text_chunk("已经"), text_chunk("写好了。")],
    )
    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        created = await client.post("/api/sessions", json={})
        assert created.status_code == 201
        sid = created.json()["sid"]

        frames = await collect(client, sid, "写个 out.txt")
        parsed = [parse_frame(f) for f in frames]
        kinds = [p["kind"] for p in parsed]

        # 以 task_cost 收尾（它是 UI 事件，跟着 task_end 的落盘记录一起来）
        assert kinds[-1] == "task_cost"
        assert "task_end" in kinds

        # 序号单调递增、无重复（决策 I：id 就是落盘 seq）
        ids = [p["id"] for p in parsed if p["id"] is not None]
        assert ids == sorted(ids), f"id 不单调：{ids}"
        assert len(set(ids)) == len(ids), f"id 有重复：{ids}"

        # delta 拼接 == 最终 assistant 消息（W-T4 的"非空"核心）
        deltas = "".join(p["data"]["text"] for p in parsed if p["kind"] == "delta")
        assert deltas == "已经写好了。"

        # 审批三态之"允许"：文件**真的**被写了
        approvals = [p for p in parsed if p["kind"] == "approval"]
        assert approvals and approvals[0]["data"]["granted"] is True
        assert approvals[0]["data"]["source"] == "user"
        assert (harness.cfg.WEB_WORKSPACE_ROOT / "out.txt").read_text(
            encoding="utf-8") == "hi"

        # 落盘契约没变：replay 能重建状态
        replay = (await client.get(f"/api/sessions/{sid}/replay")).json()
        assert replay["task_ends"], "task_end 没落盘"
        said = [m["content"] for m in replay["messages"]
                if m["role"] == "assistant" and m.get("content")]
        assert said[-1] == "已经写好了。"


# ============================================================
# W-T2：并发两会话不串台
# ============================================================

async def test_W_T2_different_workspaces_do_not_cross(harness, web_cfg):
    root = web_cfg.WEB_WORKSPACE_ROOT
    (root / "a").mkdir()
    (root / "b").mkdir()
    harness.script([tool_chunk("a1", "write_file", {"path": "a.txt", "content": "A"})],
                   [text_chunk("A 完成")])
    harness.script([tool_chunk("b1", "write_file", {"path": "b.txt", "content": "B"})],
                   [text_chunk("B 完成")])

    async with httpx.AsyncClient(base_url=harness.base, timeout=40) as client:
        info_a = (await client.post("/api/sessions", json={"workspace": "a"})).json()
        info_b = (await client.post("/api/sessions", json={"workspace": "b"})).json()
        assert info_a["shared"] is False and info_b["shared"] is False

        frames_a, frames_b = await asyncio.gather(
            collect(client, info_a["sid"], "A 的任务"),
            collect(client, info_b["sid"], "B 的任务"),
        )

    text_a, text_b = "\n".join(frames_a), "\n".join(frames_b)
    ws_a, ws_b = info_a["workspace"], info_b["workspace"]

    # ① 对方 workspace **一个字都不该出现**在事件流里（审批 detail / 工具结果 / 提示词）
    assert _contains_path(text_a, ws_a)
    assert not _contains_path(text_a, ws_b), "A 的事件流里出现了 B 的沙箱"
    assert _contains_path(text_b, ws_b)
    assert not _contains_path(text_b, ws_a), "B 的事件流里出现了 A 的沙箱"

    # ② 各写各的文件（通道 1：写）
    assert (root / "a" / "a.txt").read_text(encoding="utf-8") == "A"
    assert (root / "b" / "b.txt").read_text(encoding="utf-8") == "B"
    assert not (root / "a" / "b.txt").exists()
    assert not (root / "b" / "a.txt").exists()


async def test_W_T2_same_workspace_is_marked_shared(harness, web_cfg):
    """决策 D 选 C：同名 workspace 的文件层**不可能隔离**，所以要**摆到明面上**。

    只断言"标记出现"，**不断言文件隔离**——静默共享就是又一个"不报错的错误结论"。
    """
    harness.script([text_chunk("甲")])
    harness.script([text_chunk("乙")])

    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        first = (await client.post("/api/sessions", json={})).json()
        second = (await client.post("/api/sessions", json={})).json()

        assert first["shared"] is False and first["shared_with"] == []
        assert second["shared"] is True
        assert second["shared_with"] == [first["sid"]]

        # 日志里也要记（半年后回看才知道当时是共享的）
        start = read_records(second["log_path"])[0]
        assert start["type"] == "session_start"
        assert start["shared"] is True
        assert start["shared_with"] == [first["sid"]]

        listed = (await client.get("/api/sessions")).json()["sessions"]
        by_sid = {s["sid"]: s for s in listed}
        assert by_sid[second["sid"]]["shared"] is True
        assert by_sid[first["sid"]]["shared"] is False


# ============================================================
# W-T3b ②：审批 POST 必须与 loop 同线程
# ============================================================

async def test_W_T3b_approval_post_runs_on_loop_thread(harness):
    """把路由的 `async def` 改成 `def`，FastAPI 会把它丢进 threadpool
    （`run_in_threadpool`）→ `future.set_result()` 变成跨线程 → 复发决策 C.1 第 2 条
    （唤醒丢失）。**这个坑从类型签名上看不出来**，所以只能拿这条断言钉住。
    """
    harness.script([tool_chunk("c1", "write_file", {"path": "x.txt", "content": "1"})],
                   [text_chunk("好")])

    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        sid = (await client.post("/api/sessions", json={})).json()["sid"]
        session = harness.session(sid)
        await collect(client, sid, "写文件")

    assert session.approvals.resolve_threads, "审批 POST 根本没被调到"
    assert session.approvals.resolve_threads == {session.loop_thread_id}, (
        "审批 POST 跑在 loop 之外的线程 —— 说明它被写成同步处理器，"
        "set_result 会跨线程丢唤醒"
    )
    # 桥自己的两条线程约束也一起复验（与单元版一致）
    assert next(iter(session.approvals.thread_ids["worker"])) != session.loop_thread_id
    assert session.approvals.thread_ids["loop"] == {session.loop_thread_id}


# ============================================================
# W-T4：SSE 契约（id 只给落盘事件，且等于该记录的 seq）
# ============================================================

async def test_W_T4_id_matches_disk_seq_and_delta_has_none(harness):
    harness.script([tool_chunk("c1", "write_file", {"path": "y.txt", "content": "1"})],
                   [text_chunk("好了")])

    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        created = (await client.post("/api/sessions", json={})).json()
        sid, log_path = created["sid"], created["log_path"]
        frames = await collect(client, sid, "写文件")
        records = read_records(log_path)

    by_seq = {r["seq"]: r for r in records}
    parsed = [parse_frame(f) for f in frames]

    for item in parsed:
        if item["kind"] in DISK_EVENT_TYPES:
            assert item["id"] is not None, f"{item['kind']} 是落盘事件，必须带 id"
            assert item["id"] in by_seq, f"id={item['id']} 在日志里不存在"
            assert by_seq[item["id"]]["type"] == item["kind"], "id 与记录类型对不上"
        else:
            assert item["id"] is None, f"{item['kind']} 不是落盘事件，不许带 id"

    # 负向条：delta 一定没有 id（决策 I）
    deltas = [p for p in parsed if p["kind"] == "delta"]
    assert deltas, "这次任务没有产生 delta，断言会变成空的"
    assert all(p["id"] is None for p in deltas)
    assert all("id:" not in f for f in frames if f.startswith("event: delta\n"))


# ============================================================
# W-T5：断线重连
# ============================================================

async def test_W_T5_reconnect_resumes_from_last_disk_seq(harness):
    harness.script([tool_chunk("c1", "list_dir", {"path": "."})], [text_chunk("看完了")])

    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        sid = (await client.post("/api/sessions", json={})).json()["sid"]

        # 第一段：读到第一个落盘事件（tool）就断开
        first = await collect(client, sid, "看看目录", stop={"tool"})
        first_ids = [parse_frame(f)["id"] for f in first if parse_frame(f)["id"]]
        assert first_ids, "第一段没读到任何落盘事件"

        # 等任务跑完（关页面 ≠ 停任务；结果仍然落盘）
        for _ in range(200):
            replay = (await client.get(f"/api/sessions/{sid}/replay")).json()
            if replay["task_ends"]:
                break
            await asyncio.sleep(0.05)
        assert replay["task_ends"], "任务没跑完"

        # 第二段：带 Last-Event-ID 重连
        second = await collect(client, sid, last_event_id=max(first_ids),
                               stop={"task_cost", "task_end"}, timeout=10)
        second_ids = [parse_frame(f)["id"] for f in second if parse_frame(f)["id"]]

    # 不重复：两段的 id 交集为空
    assert not (set(first_ids) & set(second_ids)), "重连后重复投递了已提交内容"
    # 不丢：能从断点直接续上（seq 连续）
    assert min(second_ids) == max(first_ids) + 1, (first_ids, second_ids)
    # 重连后拿到的 assistant 定稿与磁盘一致
    contents = [parse_frame(f)["data"].get("content") for f in second
                if f.startswith("event: assistant")]
    assert "看完了" in contents


# ============================================================
# W-T6：安全（决策 G / D / K）
# ============================================================

async def test_W_T6_token_is_enforced(harness, web_cfg):
    auth = {"Authorization": "Bearer s3cret"}
    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        # 未配 token 时放行（默认只绑 127.0.0.1，别改成 0.0.0.0）
        assert (await client.get("/api/sessions")).status_code == 200

        web_cfg.WEB_TOKEN = "s3cret"
        try:
            assert (await client.get("/api/sessions")).status_code == 401
            assert (await client.get("/api/sessions", headers=auth)).status_code == 200
            # SSE 只能走 query（EventSource 不能设 header）
            assert (await client.get("/api/sessions", params={"token": "s3cret"})).status_code == 200
            assert (await client.get("/api/sessions",
                                     params={"token": "wrong"})).status_code == 401

            sid = (await client.post("/api/sessions", json={}, headers=auth)).json()["sid"]
            async with client.stream("GET", f"/api/sessions/{sid}/stream",
                                     params={"token": "s3cret"}) as resp:
                assert resp.status_code == 200
            async with client.stream("GET", f"/api/sessions/{sid}/stream") as resp:
                assert resp.status_code == 401
        finally:
            web_cfg.WEB_TOKEN = ""


async def test_W_T6_workspace_escape_is_rejected(harness):
    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        for bad in ("../..", "C:\\Windows", "does-not-exist"):
            resp = await client.post("/api/sessions", json={"workspace": bad})
            assert resp.status_code == 400, f"{bad} 没有被拒：{resp.text}"


async def test_W_T6_reading_env_over_web_is_still_denied(harness):
    """web 层**没有**"列目录/读文件/下载"的旁路 API——浏览文件只能让模型调工具，
    于是凭据封锁（Q5 四通道）照样生效。
    """
    root = harness.cfg.WEB_WORKSPACE_ROOT
    (root / ".env").write_text("LLM_API_KEY=sk-should-never-leak\n", encoding="utf-8")
    harness.script([tool_chunk("c1", "read_file", {"path": ".env"})], [text_chunk("好")])

    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        sid = (await client.post("/api/sessions", json={})).json()["sid"]
        frames = await collect(client, sid, "读一下 .env")

    tool_frames = [f for f in frames if f.startswith("event: tool\n")]
    assert tool_frames, "没有工具结果"
    payload = _data(tool_frames[0])
    assert "凭据" in payload["content"], payload["content"]
    assert "sk-should-never-leak" not in "\n".join(frames), "密钥泄漏到 SSE 里了"
    # 拒绝要留 policy 审计
    approval_frames = [f for f in frames if f.startswith("event: approval\n")]
    assert any(_data(f)["source"] == "policy" for f in approval_frames)


# ============================================================
# W-T9：每会话同时只允许 1 个 in-flight 任务（决策 H）
# ============================================================

async def test_W_T9_second_task_gets_409_and_first_is_unaffected(harness):
    gate = threading.Event()
    harness.queue_model(GatedLLM(gate, "慢任务完成了"))

    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        sid = (await client.post("/api/sessions", json={})).json()["sid"]
        first = await client.post(f"/api/sessions/{sid}/messages", json={"text": "慢任务"})
        assert first.status_code == 202

        second = await client.post(f"/api/sessions/{sid}/messages", json={"text": "抢跑"})
        assert second.status_code == 409, second.text
        assert "in-flight" in second.json()["detail"]

        gate.set()
        replay = {}
        for _ in range(200):
            replay = (await client.get(f"/api/sessions/{sid}/replay")).json()
            if replay["task_ends"]:
                break
            await asyncio.sleep(0.05)

        assert replay["task_ends"], "第一个任务被影响了"
        said = [m["content"] for m in replay["messages"] if m["role"] == "assistant"]
        assert "慢任务完成了" in said
        assert "抢跑" not in [m["content"] for m in replay["messages"] if m["role"] == "user"]


# ============================================================
# W-T7：非空验证 —— 把 `_out` 换回裸 print，delta 断言必须失去依据
# ============================================================

async def test_W_T7_without_out_channel_there_are_no_delta_events(harness):
    """W-T1 / W-T4 里"delta 拼接 == 定稿文本"的断言，**依据就是 `_out` 这条通道**。

    这里把 agent 的输出通道摘掉（回到改造前的裸 print），SSE 上就不该再有
    delta —— 如果还有，说明那些断言其实什么都没测。
    同时落盘的 `assistant` 记录**必须还在**（它走的是 logger，不是 `_out`）。
    """
    harness.script([text_chunk("你好")])

    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        sid = (await client.post("/api/sessions", json={})).json()["sid"]
        harness.session(sid).agent._out_impl = None      # 回到 CLI 的裸 print
        frames = await collect(client, sid, "打个招呼", stop={"task_end"}, timeout=15)

    kinds = [parse_frame(f)["kind"] for f in frames]
    assert "delta" not in kinds, "摘掉 _out 之后还有 delta，说明 delta 断言是空的"
    assert "assistant" in kinds, "落盘通道不该受影响"


# ============================================================
# 前端（W5）：浏览器真的能拿到页面与脚本
# ============================================================

async def test_static_frontend_is_served(harness):
    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        index = await client.get("/")
        assert index.status_code == 200
        assert "Coding Agent" in index.text

        script = await client.get("/app.js")
        assert script.status_code == 200
        assert "EventSource" in script.text          # 前端确实在接 SSE

        style = await client.get("/style.css")
        assert style.status_code == 200

        # 旁路 API 不存在（§2.1：浏览文件只能让模型调工具）
        assert (await client.get("/api/files")).status_code == 404


# ============================================================
# W-T3（HTTP 侧）：拒绝 → 文件没写；超时后补点"允许" → 409 且不改变结论
# ============================================================

async def test_W_T3_deny_over_http_leaves_file_untouched(harness):
    harness.script([tool_chunk("c1", "write_file", {"path": "nope.txt", "content": "x"})],
                   [text_chunk("好")])
    target = harness.cfg.WEB_WORKSPACE_ROOT / "nope.txt"

    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        sid = (await client.post("/api/sessions", json={})).json()["sid"]
        frames = []
        async with client.stream("GET", f"/api/sessions/{sid}/stream") as resp:
            assert (await client.post(f"/api/sessions/{sid}/messages",
                                      json={"text": "写文件"})).status_code == 202
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n\n" in buf:
                    raw, buf = buf.split("\n\n", 1)
                    if not raw.strip() or raw.startswith(":"):
                        continue
                    frames.append(raw)
                    kind = raw.split("\n", 1)[0].removeprefix("event: ")
                    if kind == "approval_request":
                        aid = _data(raw)["aid"]
                        r = await client.post(f"/api/sessions/{sid}/approvals/{aid}",
                                              json={"allow": False})
                        assert r.status_code == 200
                    if kind == "task_cost":
                        break
                else:
                    continue
                break

    assert not target.exists(), "用户点了拒绝，文件却被写了"
    tool_result = _data([f for f in frames if f.startswith("event: tool\n")][0])["content"]
    # 回喂文案必须与 CLI 一致（tools.py 的那句原文）
    assert "用户拒绝了该写入操作" in tool_result
    denied = [f for f in frames if f.startswith("event: approval\n")]
    assert _data(denied[0])["granted"] is False
    assert _data(denied[0])["source"] == "user"


async def test_W_T3_late_click_over_http_returns_409(harness, web_cfg):
    """超时已拒之后又点"允许" → **409，且结论不变**（决策 C.3）。

    若这里返回 200，用户会以为批准生效了，而工具早已按拒绝返回——
    最典型的"不报错的错误结论"。
    """
    web_cfg.WEB_APPROVAL_TIMEOUT = 0.5
    harness.script([tool_chunk("c1", "write_file", {"path": "late.txt", "content": "x"})],
                   [text_chunk("好")])
    target = web_cfg.WEB_WORKSPACE_ROOT / "late.txt"

    async with httpx.AsyncClient(base_url=harness.base, timeout=30) as client:
        sid = (await client.post("/api/sessions", json={})).json()["sid"]
        aid = None
        frames = []
        async with client.stream("GET", f"/api/sessions/{sid}/stream") as resp:
            assert (await client.post(f"/api/sessions/{sid}/messages",
                                      json={"text": "写文件"})).status_code == 202
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n\n" in buf:
                    raw, buf = buf.split("\n\n", 1)
                    if not raw.strip() or raw.startswith(":"):
                        continue
                    frames.append(raw)
                    kind = raw.split("\n", 1)[0].removeprefix("event: ")
                    if kind == "approval_request":
                        aid = _data(raw)["aid"]        # 故意**不点**，让它超时
                    if kind == "task_cost":
                        break
                else:
                    continue
                break

        replay = (await client.get(f"/api/sessions/{sid}/replay")).json()
        late = await client.post(f"/api/sessions/{sid}/approvals/{aid}",
                                 json={"allow": True})

    assert late.status_code == 409, late.text
    assert not target.exists(), "晚到的'允许'居然把文件写出来了"
    approvals = [r for r in read_records(harness.session(sid).logger.path)
                 if r["type"] == "approval"]
    assert approvals and approvals[0]["granted"] is False
    assert replay["task_ends"], "超时后任务应当正常收尾"
