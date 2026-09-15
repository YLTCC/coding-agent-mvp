"""
test_session_log.py —— JSONL 会话日志的行为规格（note3 §5 的 T1–T25）

跑法：python -m pytest test_session_log.py -q

本文件把 note3 §5 的"验收测试清单"逐条落成可执行断言。特别注意 §5 F 组：
**不许出现"显著小于"这类模糊比较**，全部用恒等式（note1 Q7 的教训——
含控制流的改动人工 review 三轮没抓干净，必须固化为 pytest）。
"""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import agent as agent_mod
from agent import tools as tools_mod
from agent.agent import CodingAgent, SYSTEM_PROMPT
from agent.config import config as cfg
from agent.session_log import (
    SOURCE_TYPES,
    SessionLogError,
    SessionLogger,
    replay,
    replay_file,
    read_records,
)


# ============================================================
# 测试替身 / 小工具
# ============================================================

def _open_log(tmp_path, session_id="testsession"):
    # system_prompt 必须与 Agent 实际使用的一致：它是源事件，重放靠它复原
    # messages[0]。写在日志里的值与运行态不同，正是 T5/T11 会抓出来的 bug。
    return SessionLogger.open_session(
        tmp_path,
        system_prompt=SYSTEM_PROMPT,
        workspace=".",
        model="test-model",
        config={"HISTORY_WINDOW_TURNS": 8},
        session_id=session_id,
    )


def _make_agent(tmp_path, monkeypatch, **overrides):
    for key, value in overrides.items():
        monkeypatch.setattr(cfg, key, value)
    log = _open_log(tmp_path)
    return CodingAgent(session_log=log), log


def _turn(n: int, tools: bool = True) -> list[dict]:
    """造一轮：user (+ assistant(tool_calls) + tool) + assistant 文本。"""
    base = f"file{n}.py"
    msgs = [{"role": "user", "content": f"任务{n}：读取文件 {base}"}]
    if tools:
        msgs += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{n}",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": base}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"call_{n}", "content": f"{base} 的内容"},
        ]
    msgs.append({"role": "assistant", "content": f"已完成任务{n}"})
    return msgs


def _fill_turns(agent: CodingAgent, count: int, start: int = 1) -> None:
    """把 count 轮历史塞进 Agent。

    必须走 `_append_message`（而不是直接改 agent.messages）：否则 `_message_seqs`
    与 messages 错位，水位就无从算起——这正是被测代码的关键不变量。
    """
    for n in range(start, start + count):
        for msg in _turn(n):
            agent._append_message(dict(msg))


# ---------- LLM 替身 ----------

def _chunk(content=None, tool_calls=None, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))
        ],
        usage=usage,
    )


def _tc_chunk(index, call_id, name, args):
    return _chunk(tool_calls=[
        SimpleNamespace(
            index=index,
            id=call_id,
            function=SimpleNamespace(name=name, arguments=args),
        )
    ])


def _patch_llm(monkeypatch, replies, calls=None):
    """replies：每轮模型回复 → chunk 列表。摘要调用自动被识别，不消耗 replies。"""
    calls = calls if calls is not None else []
    pending = list(replies)

    async def fake_chat_stream(messages, tools=None):
        is_summary = any(
            m.get("role") == "system" and "摘要" in (m.get("content") or "")
            for m in messages
        )
        calls.append({"kind": "summary" if is_summary else "main", "messages": list(messages)})
        if is_summary:
            yield _chunk(content="【摘要】早期若干轮：读过若干文件")
            return
        chunks = pending.pop(0) if pending else []
        for chunk in chunks:
            yield chunk

    monkeypatch.setattr(agent_mod.llm, "chat_stream", fake_chat_stream)
    return calls


def _raw_lines(path) -> list[str]:
    raw = Path(path).read_bytes().decode("utf-8")
    return [line for line in raw.split("\n") if line != ""]


# ============================================================
# A. 格式与信封
# ============================================================

class TestFormat:
    def test_T1_seq_continuous_from_one(self, tmp_path, monkeypatch):
        """T1：每行都是合法 JSON；seq == 1..N 连续、无重复"""
        _patch_llm(monkeypatch, [[_chunk(content="你好")]])
        agent, log = _make_agent(tmp_path, monkeypatch)
        import asyncio

        asyncio.get_event_loop_policy()
        asyncio.run(agent.run_task("你好"))
        log.close()

        seqs = []
        for line in _raw_lines(log.path):
            rec = json.loads(line)
            assert isinstance(rec, dict)
            seqs.append(rec["seq"])
        assert seqs == list(range(1, len(seqs) + 1))

    def test_T2_first_line_is_session_start(self, tmp_path):
        """T2：首行是 session_start 且四个字段齐全"""
        log = _open_log(tmp_path)
        log.close()
        first = json.loads(_raw_lines(log.path)[0])
        assert first["type"] == "session_start"
        for key in ("system_prompt", "workspace", "model", "config"):
            assert key in first, f"session_start 缺字段 {key}"
        assert first["v"] == 1
        assert first["session_id"] == log.session_id

    def test_T3_no_cr_in_file(self, tmp_path):
        """T3：文件字节中不含 \\r（对 _fix_crcrlf.py 前科的防线）"""
        log = _open_log(tmp_path)
        log.emit("user", content="中文内容，含标点。")
        log.close()
        assert b"\r" not in Path(log.path).read_bytes()

    def test_T4_chinese_roundtrip_and_not_escaped(self, tmp_path):
        """T4：中文 round-trip 逐字一致，且文件里不出现 \\uXXXX"""
        text = "中文 content：请读取 配置文件，注意 换行符。"
        log = _open_log(tmp_path)
        log.emit("user", content=text)
        log.close()

        raw = Path(log.path).read_bytes().decode("utf-8")
        assert "\\u" not in raw            # ensure_ascii=False
        assert text in raw
        users = [r for r in read_records(log.path) if r["type"] == "user"]
        assert users[0]["content"] == text


# ============================================================
# B. 重放正确性（I1 / I5）
# ============================================================

class TestReplay:
    async def test_T5_replay_equals_runtime(self, tmp_path, monkeypatch):
        """T5：含多工具轮次的任务 → 重放 messages 与运行态逐字一致"""
        _patch_llm(monkeypatch, [
            [_tc_chunk(0, "call_1", "list_dir", '{"path": "."}')],
            [_chunk(content="目录看完了")],
        ])
        # 4.6 终态：工具层的 ctx 是必填参数，所以替身也要收 3 个
        # （agent 的 _run_tool 恒定传三参；2 参替身会 TypeError）
        monkeypatch.setattr(agent_mod, "dispatch_tool", lambda name, args, ctx: "[列表]")
        agent, log = _make_agent(tmp_path, monkeypatch)
        await agent.run_task("看看目录")
        log.close()

        state = replay_file(log.path)
        assert state.messages == agent.messages

    async def test_T6_replay_makes_zero_model_calls(self, tmp_path, monkeypatch):
        """T6：重放全程零模型调用（重放绝不重算摘要）"""
        _patch_llm(monkeypatch, [[_chunk(content="好")]])
        agent, log = _make_agent(tmp_path, monkeypatch)
        await agent.run_task("你好")
        log.close()

        calls = []

        async def forbidden(messages, tools=None):
            calls.append(messages)
            raise AssertionError("重放不该调用模型")
            yield  # pragma: no cover

        monkeypatch.setattr(agent_mod.llm, "chat_stream", forbidden)
        replay_file(log.path)
        assert calls == []

    async def test_T7_replay_is_deterministic(self, tmp_path, monkeypatch):
        """T7：同一日志重放两次，序列化结果字节级一致"""
        _patch_llm(monkeypatch, [[_chunk(content="答案：42")]])
        agent, log = _make_agent(tmp_path, monkeypatch)
        await agent.run_task("算一下")
        log.close()

        a = json.dumps(replay_file(log.path).messages, ensure_ascii=False, sort_keys=True)
        b = json.dumps(replay_file(log.path).messages, ensure_ascii=False, sort_keys=True)
        assert a == b

    def test_T8_content_empty_and_null(self, tmp_path):
        """T8：content="" 与 content=None 两种形态重放后都复原成 "" """
        log = _open_log(tmp_path)
        log.emit("user", content="")
        log.emit("assistant", content=None)      # 手写一行 null（模拟早期日志）
        log.emit("assistant", content="")
        log.close()

        state = replay_file(log.path)
        contents = [m["content"] for m in state.messages[1:]]
        assert contents == ["", "", ""]


# ============================================================
# C. 水位坐标（I4 / P2 的正面验证）
# ============================================================

class TestWatermark:
    async def test_T9_compression_state_snapshot(self, tmp_path, monkeypatch):
        """T9：压缩后重放，摘要与已压缩轮数与运行态相等（state 行直接赋值）"""
        _patch_llm(monkeypatch, [])
        agent, log = _make_agent(
            tmp_path, monkeypatch,
            HISTORY_WINDOW_TURNS=8, HISTORY_COMPRESS_LAG=4,
        )
        _fill_turns(agent, 12)
        assert await agent.compress_history() is True
        log.close()

        state = replay_file(log.path)
        assert state.summary == agent._summary
        assert state.compressed_turns == agent._compressed_turns

    async def test_T10_tampered_summary_is_taken_verbatim(self, tmp_path, monkeypatch):
        """T10：篡改日志里的 summary → 重放取到篡改值（证明没重算，不是碰巧算对）"""
        _patch_llm(monkeypatch, [])
        agent, log = _make_agent(
            tmp_path, monkeypatch,
            HISTORY_WINDOW_TURNS=8, HISTORY_COMPRESS_LAG=4,
        )
        _fill_turns(agent, 12)
        await agent.compress_history()
        log.close()

        lines = _raw_lines(log.path)
        tampered = []
        for line in lines:
            rec = json.loads(line)
            if rec["type"] == "state":
                rec["summary"] = "【被人为篡改的摘要】"
            tampered.append(json.dumps(rec, ensure_ascii=False))
        Path(log.path).write_bytes(("\n".join(tampered) + "\n").encode("utf-8"))

        state = replay_file(log.path)
        assert state.summary == "【被人为篡改的摘要】"
        assert state.summary != agent._summary

    async def test_T11_trim_after_task_consistency(self, tmp_path, monkeypatch):
        """T11：裁剪后重放，条数 / 已压缩轮数 / 水位三者与运行态一致"""
        _patch_llm(monkeypatch, [])
        agent, log = _make_agent(
            tmp_path, monkeypatch,
            HISTORY_WINDOW_TURNS=8, HISTORY_COMPRESS_LAG=4,
            MAX_HISTORY_MESSAGES=20,
        )
        _fill_turns(agent, 12)
        await agent.compress_history()
        _fill_turns(agent, 4, start=13)
        agent._maybe_trim_after_task()
        log.close()

        state = replay_file(log.path)
        assert len(state.messages) == len(agent.messages)
        assert state.compressed_turns == agent._compressed_turns
        assert state.dropped_upto_seq == agent._dropped_upto_seq
        assert state.messages == agent.messages

    async def test_T12_compress_trim_compress_composite(self, tmp_path, monkeypatch):
        """T12：连续 3 个任务（压缩→裁剪→再压缩）后重放 == 运行态
        专抓 _compressed_turns 回退错位（裁剪会同步回退它，退错一位切片就整体错位）"""
        _patch_llm(monkeypatch, [])
        agent, log = _make_agent(
            tmp_path, monkeypatch,
            HISTORY_WINDOW_TURNS=8, HISTORY_COMPRESS_LAG=4,
            MAX_HISTORY_MESSAGES=20,
        )
        _fill_turns(agent, 12)
        assert await agent.compress_history() is True      # 任务一：压缩
        _fill_turns(agent, 4, start=13)
        agent._maybe_trim_after_task()                     # 任务二：裁剪
        _fill_turns(agent, 12, start=17)
        assert await agent.compress_history() is True      # 任务三：再压缩
        log.close()

        state = replay_file(log.path)
        assert state.messages == agent.messages
        assert state.summary == agent._summary
        assert state.compressed_turns == agent._compressed_turns
        assert state.dropped_upto_seq == agent._dropped_upto_seq

    async def test_T13_watermark_boundary_and_orphan_tool(self, tmp_path, monkeypatch):
        """T13：整轮被裁 + 孤儿 tool 被 _drop_leading_orphan_tools 丢弃，两种情形重放一致

        裁剪点选在 34：34 % 4 == 2，正好落在 _turn 结构里的 tool 位置，
        于是保留区首条是"孤儿 tool"（其 assistant(tool_calls) 已被裁掉）。
        它的 seq > 水位，水位表达不了它——必须靠复用运行时纯函数丢弃。
        """
        _patch_llm(monkeypatch, [])
        agent, log = _make_agent(
            tmp_path, monkeypatch,
            HISTORY_WINDOW_TURNS=8, HISTORY_COMPRESS_LAG=4,
            MAX_HISTORY_MESSAGES=14,
        )
        _fill_turns(agent, 12)
        assert agent.messages[1 + 34].get("role") == "tool"   # 前置条件：确实造出了孤儿
        agent._maybe_trim_after_task()
        assert agent.messages[1].get("role") != "tool"        # 运行态已把它丢掉
        log.close()

        state = replay_file(log.path)
        assert state.messages == agent.messages
        assert state.dropped_upto_seq == agent._dropped_upto_seq

    async def test_T14_clear_resets_to_session_start_state(self, tmp_path, monkeypatch):
        """T14：clear 之后的重放态 == session_start 刚写完的态"""
        _patch_llm(monkeypatch, [])
        agent, log = _make_agent(
            tmp_path, monkeypatch,
            HISTORY_WINDOW_TURNS=8, HISTORY_COMPRESS_LAG=4,
        )
        _fill_turns(agent, 12)
        await agent.compress_history()
        agent.reset_conversation()
        log.close()

        state = replay_file(log.path)
        assert state.messages == agent.messages == [agent.messages[0]]
        assert state.summary is None
        assert state.compressed_turns == 0
        assert state.dropped_upto_seq == 0
        assert state.cleared_count == 1

    def _assert_aligned(self, agent):
        """不变量：_message_seqs 与 messages 逐位对齐，且 seq 严格递增、全部高于水位。

        只比长度是抓不出错位的——必须逐位确认没整体平移。
        """
        assert len(agent._message_seqs) == len(agent.messages)
        seqs = [s for s in agent._message_seqs[1:] if s is not None]
        assert all(b > a for a, b in zip(seqs, seqs[1:])), f"seq 非递增：{seqs}"
        assert all(s > agent._dropped_upto_seq for s in seqs), \
            f"保留消息的 seq 未全部高于水位 {agent._dropped_upto_seq}"

    def _abort_tail_turns(self, agent, n, k):
        """复刻 MAX_ITERATIONS 用尽时的中止态：assistant(K 个 tool_calls) + K 条 tool，
        末尾**不留** assistant 文本。一轮多工具会连续追加 K 条 tool，
        于是"末 MAX_HISTORY_MESSAGES 条全是 tool"成为可能。"""
        calls = [
            {
                "id": f"call_{n}_{j}",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{n}_{j}.py"})},
            }
            for j in range(k)
        ]
        agent._append_message({"role": "user", "content": f"任务{n}"})
        agent._append_message({"role": "assistant", "content": "", "tool_calls": calls})
        for j in range(k):
            agent._append_message(
                {"role": "tool", "tool_call_id": f"call_{n}_{j}", "content": f"内容{j}"}
            )

    def test_T27_trim_all_orphans_leaves_system_only(self, tmp_path, monkeypatch):
        """T27：保留区被孤儿 tool **吃光**（retained 为空）时也保持一致。

        这是 `else` 分支——正常任务路径不可达（轮末必是 assistant 文本），
        只有中止态（末几条全是 tool）才触发。专防"取末尾 len(retained) 个"
        在 len(retained)==0 时退化成错误切片。
        """
        _patch_llm(monkeypatch, [])
        agent, log = _make_agent(
            tmp_path, monkeypatch,
            HISTORY_WINDOW_TURNS=8, HISTORY_COMPRESS_LAG=4,
            MAX_HISTORY_MESSAGES=2,          # 小到让末 2 条 tool 能被整个吃掉
        )
        for n in range(1, 5):
            self._abort_tail_turns(agent, n, 3)
            agent._maybe_trim_after_task()
            self._assert_aligned(agent)

        # 前置条件：确实走到了"只剩 system"的 else 分支，而不是运气好没触发
        assert agent.messages == [agent.messages[0]]
        assert agent._message_seqs == [None]
        log.close()

        state = replay_file(log.path)
        assert state.messages == agent.messages
        assert state.dropped_upto_seq == agent._dropped_upto_seq

    async def test_T28_randomized_interleaving_stays_aligned(self, tmp_path, monkeypatch):
        """T28：随机交错 append/trim/compress/clear 后，对齐不变量 + 逐位对应日志 + 重放一致。

        T11–T13 用的是固定序列，覆盖不到"多次裁剪与压缩交错"的组合。
        种子固定 → 确定性，不是随机翻车测试。
        """
        import random

        rnd = random.Random(20260914)
        _patch_llm(monkeypatch, [])
        agent, log = _make_agent(
            tmp_path, monkeypatch,
            HISTORY_WINDOW_TURNS=8, HISTORY_COMPRESS_LAG=2,
            MAX_HISTORY_MESSAGES=3,          # 激进裁剪：多制造边界
        )

        n = 0
        for _ in range(40):
            op = rnd.choice(["append", "append", "trim", "compress"])
            if op == "append":
                for _ in range(rnd.randint(1, 4)):
                    n += 1
                    _fill_turns(agent, 1, start=n)
            elif op == "trim":
                agent._maybe_trim_after_task()
            else:
                await agent.compress_history()
            self._assert_aligned(agent)
        log.close()

        # 逐位对应日志里的真实源事件——比只比长度更能抓"整体平移"式错位
        by_seq = {
            rec["seq"]: rec
            for rec in read_records(log.path)
            if rec.get("type") in SOURCE_TYPES
        }
        for i, seq in enumerate(agent._message_seqs):
            if seq is None:
                continue
            rec = by_seq[seq]
            assert rec["type"] == agent.messages[i]["role"]
            assert (rec.get("content") or "") == (agent.messages[i].get("content") or "")

        state = replay_file(log.path)
        assert state.messages == agent.messages
        assert state.compressed_turns == agent._compressed_turns
        assert state.dropped_upto_seq == agent._dropped_upto_seq


# ============================================================
# D. 并发与写入（I2 / §4）
# ============================================================

class TestConcurrency:
    def test_T15_multithread_stress_no_half_lines_no_dup_seq(self, tmp_path):
        """T15：N 线程 + 主线程并发投递 M 条 → 行数 == M+meta，无半行、seq 无重号"""
        log = _open_log(tmp_path)
        threads_n, per_thread = 8, 25
        errors: list[BaseException] = []

        def worker(tid: int):
            try:
                for i in range(per_thread):
                    log.emit("user", content=f"线程{tid}-第{i}条")
            except BaseException as e:      # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(threads_n)]
        for t in threads:
            t.start()
        for i in range(per_thread):
            log.emit("user", content=f"主线程-第{i}条")
        for t in threads:
            t.join()
        log.close()

        assert errors == []
        lines = _raw_lines(log.path)
        assert len(lines) == threads_n * per_thread + per_thread + 1   # +1 = session_start

        seqs = []
        for line in lines:
            rec = json.loads(line)          # 半行 JSON 会在这里直接炸
            seqs.append(rec["seq"])
        assert seqs == list(range(1, len(seqs) + 1))   # 连续、无重号、无空洞

    async def test_T16_assistant_seq_before_tool_seq(self, tmp_path, monkeypatch):
        """T16：工具路径（asyncio.to_thread 内的审批投递）与主循环写入交错，
        assistant(tool_calls) 的 seq 恒小于对应 tool 的 seq（I3）"""
        _patch_llm(monkeypatch, [
            [_tc_chunk(0, "call_a", "run_command", '{"command": "python main.py"}')],
            [_chunk(content="跑完了")],
        ])
        # 真实审批路径：审批事件从 threadpool 线程里投出来。
        # AUTO_APPROVE 必须显式钉成 False：test_tools.py 在导入时就把
        # os.environ["AUTO_APPROVE"] 设成了 "true"，整个 pytest 会话都受影响，
        # 不写死这里就会偶发走到 "auto" 分支（实测：单跑绿、全量红）
        monkeypatch.setattr(cfg, "AUTO_APPROVE", False)

        def fake_dispatch(name, args, ctx):
            # 4.6 终态：审批回调不再有进程级的可依赖（set_approval_callback 已删），
            # 于是这里**只能**用 agent 真的传下来的 ctx 去审批——这反而把
            # "工具层确实拿到了本会话的 ctx"也一并钉住了。
            assert tools_mod._ask_approval("执行命令", "python main.py", ctx) is True
            return "命令输出"

        monkeypatch.setattr(agent_mod, "dispatch_tool", fake_dispatch)
        log = _open_log(tmp_path)
        agent = CodingAgent(session_log=log, approval_callback=lambda action, detail: True)
        await agent.run_task("跑一下")
        log.close()

        records = read_records(log.path)
        approvals = [r for r in records if r["type"] == "approval"]
        assert approvals, "审批事件应当落盘"
        assert approvals[0]["source"] == "user" and approvals[0]["granted"] is True

        call_seq = {
            tc["id"]: r["seq"]
            for r in records if r["type"] == "assistant"
            for tc in (r.get("tool_calls") or [])
        }
        for r in records:
            if r["type"] == "tool":
                assert call_seq[r["tool_call_id"]] < r["seq"], "I3 被违反：tool 先于 assistant"

    def test_T26_no_cross_session_leak_of_approval_events(self, tmp_path, monkeypatch):
        """T26（实现期新增；4.6 终态后**重写**）：无 logger 的 Agent 不得把审批事件
        写进**上一个会话**的日志。

        改造前这条钉的是"默认上下文的 sink 必须无条件设置（含 None）"——因为那个
        sink 是**进程级共享**的：只在"有 logger 时"才清，后构造的无 logger Agent
        就会沿用旧 sink → 跨会话串台。4.6 终态删掉了默认上下文，每个 Agent 构造时
        自造私有 ctx，那条失败路径在结构上已经不存在。

        所以本用例改成钉**新的不变式**（比原来更强，也更难绕过）：实例之间的 ctx
        是各自独立的对象；没有 logger 的那个**没有 sink**，它的审批事件因此
        无处可去——不报错，但也绝不可能落进别人的日志。
        """
        monkeypatch.setattr(cfg, "AUTO_APPROVE", False)

        log = _open_log(tmp_path)
        a = CodingAgent(session_log=log, approval_callback=lambda action, detail: True)
        b = CodingAgent(approval_callback=lambda action, detail: True)   # 没有 logger

        # ① 两个实例的 ctx 各自独立，不是共享的同一个对象
        assert a.ctx is not b.ctx
        # ② 有 logger 的那个有 sink；没有 logger 的那个**没有**（也没有回落目标）
        assert a.ctx.event_sink is not None
        assert b.ctx.event_sink is None

        # ③ A 的审批走 A 的 ctx → 落进 A 的日志
        assert tools_mod._ask_approval("执行命令", "python main.py", a.ctx) is True
        # ④ B 的审批走 B 的 ctx → 无处可去
        assert tools_mod._ask_approval("执行命令", "whoami", b.ctx) is True
        log.close()

        approvals = [r for r in read_records(log.path) if r["type"] == "approval"]
        assert [r["detail"] for r in approvals] == ["python main.py"], (
            "B 的审批事件泄漏进了 A 的会话日志"
        )

    async def test_T17_no_streaming_delta_rows(self, tmp_path, monkeypatch):
        """T17：多 chunk 流式回复只落 1 条 assistant 行（定稿），无"部分内容"行"""
        _patch_llm(monkeypatch, [
            [
                _chunk(content="你"),
                _chunk(content="好"),
                _chunk(content="！"),
                _tc_chunk(0, "call_z", "list_dir", ""),
                _tc_chunk(0, None, "", '{"path": "."}'),      # 参数被切成两个 chunk
            ],
            [_chunk(content="完成")],
        ])
        monkeypatch.setattr(agent_mod, "dispatch_tool", lambda name, args, ctx: "[列表]")
        agent, log = _make_agent(tmp_path, monkeypatch)
        await agent.run_task("你好")
        log.close()

        records = read_records(log.path)
        assistants = [r for r in records if r["type"] == "assistant"]
        assert len(assistants) == 2                     # 两轮模型调用 → 两行，不是五个 chunk
        assert assistants[0]["content"] == "你好！"
        # 参数分片必须聚合成完整 JSON，且落盘的是聚合后的结构
        assert assistants[0]["tool_calls"][0]["function"]["arguments"] == '{"path": "."}'


# ============================================================
# E. 鲁棒性（I6）
# ============================================================

class TestRobustness:
    async def test_T18_torn_tail_is_dropped(self, tmp_path, monkeypatch):
        """T18：末行写一半 → 重放成功，结果等于截断前最后一次完整 commit 的态"""
        _patch_llm(monkeypatch, [[_chunk(content="好的")]])
        agent, log = _make_agent(tmp_path, monkeypatch)
        await agent.run_task("第一问")
        log.close()

        before = replay_file(log.path)
        raw = Path(log.path).read_bytes()
        Path(log.path).write_bytes(raw + '{"v":1,"seq":99,"ty'.encode("utf-8"))

        after = replay_file(log.path)
        assert after.messages == before.messages
        assert after.summary == before.summary

    async def test_T19_middle_corruption_raises(self, tmp_path, monkeypatch):
        """T19：中间行损坏 → 抛明确异常（不得静默跳过）"""
        _patch_llm(monkeypatch, [[_chunk(content="好的")]])
        agent, log = _make_agent(tmp_path, monkeypatch)
        await agent.run_task("第一问")
        log.close()

        lines = _raw_lines(log.path)
        lines[1] = "{这不是合法 JSON"                 # 破坏第二行（非末行）
        Path(log.path).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))

        with pytest.raises(SessionLogError):
            replay_file(log.path)

    def test_T20_empty_and_start_only(self, tmp_path):
        """T20：空文件 / 只有 session_start → 不崩，得到默认派生态"""
        empty = tmp_path / "empty.jsonl"
        empty.write_bytes(b"")
        state = replay_file(empty)
        assert state.messages == []
        assert state.summary is None and state.compressed_turns == 0

        log = _open_log(tmp_path, session_id="onlystart")
        log.close()
        state2 = replay_file(log.path)
        assert len(state2.messages) == 1
        assert state2.messages[0]["role"] == "system"
        assert state2.messages[0]["content"] == SYSTEM_PROMPT

    def test_T21_unknown_type_skipped_but_counted(self, tmp_path):
        """T21：未知 type 行 → 跳过不报错，但计入"未知行计数"便于排查"""
        log = _open_log(tmp_path)
        log.emit("user", content="正常一行")
        log.emit("future_thing", whatever=1)          # 将来版本才有的行
        log.close()

        state = replay_file(log.path)
        assert state.unknown_lines == 1
        assert [m["content"] for m in state.messages[1:]] == ["正常一行"]


# ============================================================
# F. 写放大上界（§0 的量化，用恒等式而不是"显著小于"）
# ============================================================

class FullDumpWriter:
    """对照实现：每轮结束把"到目前的全部消息"重新序列化写一遍（全量 dump）。

    为了与 append 侧比**同一批字节**，它吃的是行字符串（不含换行），
    而不是自己重新序列化——否则信封差异会污染结论。
    """

    def __init__(self):
        self.lines: list[str] = []
        self.bytes = 0
        self.commits = 0

    def append(self, line: str) -> None:
        self.lines.append(line)

    def commit(self) -> None:
        blob = "".join(line + "\n" for line in self.lines)
        self.bytes += len(blob.encode("utf-8"))
        self.commits += 1


def _source_lines_by_turn(path) -> list[tuple[int, str]]:
    """从真实日志里取源事件行（原始字符串）+ 轮次号。

    轮次划分：一条 user 行开始新的一轮，其后的 assistant/tool 都算这一轮。
    """
    out: list[tuple[int, str]] = []
    turn = 0
    for line in _raw_lines(path):
        rec = json.loads(line)
        if rec["type"] not in SOURCE_TYPES:
            continue
        if rec["type"] == "user":
            turn += 1
        out.append((turn, line))
    return out


def _per_turn_bytes(pairs: list[tuple[int, str]]) -> list[int]:
    """Δⱼ：第 j 轮新落盘的源事件字节（含行尾换行）"""
    buckets: dict[int, int] = {}
    for turn, line in pairs:
        buckets[turn] = buckets.get(turn, 0) + len(line.encode("utf-8")) + 1
    return [buckets[t] for t in sorted(buckets)]


def _full_dump_simulation(pairs: list[tuple[int, str]]) -> FullDumpWriter:
    writer = FullDumpWriter()
    turns = sorted({turn for turn, _ in pairs})
    cursor = 0
    for turn in turns:
        while cursor < len(pairs) and pairs[cursor][0] == turn:
            writer.append(pairs[cursor][1])
            cursor += 1
        writer.commit()
    return writer


class TestWriteAmplification:
    async def test_T22_no_delta_and_no_per_turn_dump_rows(self, tmp_path, monkeypatch):
        """T22：源事件行数 == 提交消息数；state 行数 <= 任务数 × 2"""
        _patch_llm(monkeypatch, [
            [_tc_chunk(0, "call_1", "list_dir", '{"path": "."}')],
            [_chunk(content="好了")],
            [_tc_chunk(0, "call_2", "glob", '{"pattern": "*.py"}')],
            [_chunk(content="真好了")],
        ])
        monkeypatch.setattr(agent_mod, "dispatch_tool", lambda name, args, ctx: "[结果]")
        agent, log = _make_agent(tmp_path, monkeypatch)
        await agent.run_task("任务一")
        await agent.run_task("任务二")
        log.close()

        records = read_records(log.path)
        source_rows = [r for r in records if r["type"] in SOURCE_TYPES]
        state_rows = [r for r in records if r["type"] == "state"]
        tasks = 2

        assert len(source_rows) == len(agent.messages) - 1     # 无 delta 行、无每轮 dump 行
        assert len(state_rows) <= tasks * 2

    async def test_T23_identity_full_equals_weighted_sum(self, tmp_path, monkeypatch):
        """T23：恒等式（精确，非"显著"）
        full == Σⱼ Δⱼ(T-j+1)，且与独立实现的 FullDumpWriter 字节数逐字相等"""
        _patch_llm(monkeypatch, [
            [_tc_chunk(0, "call_1", "list_dir", '{"path": "."}')],
            [_chunk(content="第一轮完成")],
            [_tc_chunk(0, "call_2", "read_file", '{"path": "a.py"}')],
            [_chunk(content="第二轮完成")],
            [_chunk(content="第三轮完成")],
        ])
        monkeypatch.setattr(agent_mod, "dispatch_tool", lambda name, args, ctx: "[工具结果]" * 20)
        agent, log = _make_agent(tmp_path, monkeypatch)
        await agent.run_task("任务一")
        await agent.run_task("任务二")
        await agent.run_task("任务三")
        log.close()

        pairs = _source_lines_by_turn(log.path)
        deltas = _per_turn_bytes(pairs)
        turns = len(deltas)
        assert turns >= 3

        append_bytes = sum(deltas)
        full_formula = sum(d * (turns - j) for j, d in enumerate(deltas))   # j 从 0 起
        full_simulated = _full_dump_simulation(pairs).bytes

        assert full_simulated == full_formula                # 独立实现 == 公式（恒等式）
        assert 1 <= full_formula / append_bytes <= turns     # 比值落在 [1, T]

    def test_T24_uniform_flow_closed_form(self, tmp_path):
        """T24：均匀流（Δⱼ ≡ Δ）→ full/append == (T+1)/2
        这条专门当"有人把系数写成 1/T"的护栏"""
        for turns in (2, 3, 5, 10):
            line = "y" * 100
            writer = FullDumpWriter()
            for _ in range(turns):
                writer.append(line)
                writer.commit()
            append_bytes = turns * (len(line) + 1)
            # 整数比较，避开浮点误差：full / append == (T+1) / 2
            assert writer.bytes * 2 == append_bytes * (turns + 1)

    def test_T25_skew_robustness(self):
        """T25：Δ 后重时比值 → 1，前重时 → T；两端都落在 [1, T] 内，不假设常数比率
        （真实 agent 会话往往后重——最后几步才 read_file 大文件，
         这正是 1/T 那个错误系数会挂的地方）"""
        def ratio(deltas):
            turns = len(deltas)
            full = sum(d * (turns - j) for j, d in enumerate(deltas))
            return full / sum(deltas), turns

        front, tf = ratio([1000, 100, 100, 100, 100])
        uniform, tu = ratio([100] * 5)
        back, tb = ratio([100, 100, 100, 100, 1000])
        all_last, tl = ratio([0, 0, 0, 0, 1000])

        assert 1 <= back < uniform < front <= tf
        assert all_last == 1                      # 全在末轮：一个字节都没被重写
        assert uniform == (tu + 1) / 2            # 均匀流取中段
        for value, turns in ((front, tf), (uniform, tu), (back, tb)):
            assert 1 <= value <= turns
