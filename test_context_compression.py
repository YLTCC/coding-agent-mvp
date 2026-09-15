"""
test_context_compression.py —— 上下文压缩（滑动窗口 + 滚动摘要）行为规格 [TDD 红]

本文件是"上下文压缩"这一阶段的**行为规格**：先写测试，后补实现。
  当前 agent.py 只有 MAX_HISTORY_MESSAGES 硬砍，下面的 async 测试预期全部 FAIL，
  直到 CodingAgent 补上 compress_history / 新的 _build_context 之后转绿。
  绿灯阶段的唯一改动对象应是 agent/agent.py；本测试文件本身不应再改。

跑法：python -m pytest test_context_compression.py -q

────────────────────────────────────────────────────────────
被测接口（已冻结，实现请对齐）
────────────────────────────────────────────────────────────
config（已存在，直接复用）
    HISTORY_WINDOW_TURNS   窗口轮数，默认 8
    HISTORY_COMPRESS_LAG   压缩滞后轮数，默认 4
    SUMMARY_MAX_CHARS      累积摘要字符预算，默认 2000
    MAX_HISTORY_MESSAGES   最后一道硬砍（条数），默认 20

agent.agent 模块级
    SUMMARY_SYSTEM_PROMPT               : str    摘要用的 system prompt（须含"摘要"二字）
    _count_turns(messages)              -> int   数"轮"：role=="user" 的条数
    _window_start(messages, window)     -> int   滑窗起点下标（保证首条是 user）
    _summary_message(summary)           -> dict  {"role":"user","content":...}
    _cap_summary(text, max_chars=None)  -> str   按预算截断
    _summarize(old_messages, prev_summary) -> str  async，调 llm.chat_stream 拿摘要

CodingAgent
    self._summary          : str | None  累积摘要，初始 None
    self._compressed_turns : int         已折进摘要的"轮"数，初始 0
    async compress_history() -> bool     攒够 lag 才压；压了 True，否则 False
    _build_context()       -> list[dict] 仍同步；顺序 = system, [摘要], 未压缩历史

────────────────────────────────────────────────────────────
核心不变量（行为契约）
────────────────────────────────────────────────────────────
1. 攒够 lag 才压：滑出窗口的轮数 < HISTORY_COMPRESS_LAG → 不调用模型，返回 False
2. 滚动：新一轮摘要把"上一版摘要"一起喂回去，累积成一份（不是各压各的）
3. 摘要失败 fail-safe：异常时保留原摘要、不推进游标、任务继续跑
4. 只压"滑出窗口"的部分，仍在窗口里的最近几轮原样保留
5. 发给模型的上下文 = system + 摘要(若有, 紧跟 system) + 未压缩历史
6. MAX_HISTORY_MESSAGES 是最后一道硬砍，且不砍摘要本身
7. _build_context / compress_history 绝不就地修改 self.messages
8. **压缩边界只落在"完整 turn"上**：轮 = 一条 user 消息 + 它之后直到
   下一条 user 之前的所有消息。若历史尾部是 `user → assistant(tool_calls)`
   而工具结果还没回来，这段 partial turn 必须留在窗口里，绝不能被压进摘要
   （压掉它 = 丢掉模型刚发出的 tool_calls，下一轮请求会因 tool_call_id 失配报错）
   约定：轮按 user 消息切分，partial turn 也占一个窗口槽位。
9. **摘要不得随压缩次数膨胀**：连续压 N 次后长度只能小幅抖动（≤ +20%），
   绝不线性增长/翻倍；且任何时刻都不破 SUMMARY_MAX_CHARS。
10. **摘要注入后窗口首条不得是 role="tool"**：切除历史时留下的"孤儿 tool"
    （其 assistant(tool_calls) 已被切掉）必须一并丢弃，否则接口 400。
"""
import json
from types import SimpleNamespace

from agent.config import config as cfg
from agent import agent as agent_mod
from agent.agent import CodingAgent


# ============================================================
# 测试替身 / 小工具
# ============================================================

def _turn(n: int, tools: bool = True) -> list[dict]:
    """造"一轮"：user (+ assistant(tool_calls) + tool) + assistant 文本。

    tools=False 时只造 user + assistant（2 条），用于测最少条数场景。
    """
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


def _partial_turn(n: int) -> list[dict]:
    """尾部残缺的一轮：user → assistant(tool_calls)，工具结果还没回来。"""
    return [
        {"role": "user", "content": f"任务{n}：改完记得跑测试"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{n}",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({"command": "python -m pytest"}),
                    },
                }
            ],
        },
    ]


def _turns(*nums: int, tools: bool = True) -> list[dict]:
    out: list[dict] = []
    for n in nums:
        out += _turn(n, tools=tools)
    return out


def _new_agent(turn_count: int) -> CodingAgent:
    """构造一个只有 turn_count 轮历史、其余为空的 Agent。"""
    agent = CodingAgent()
    agent.messages = [agent.messages[0]] + _turns(*range(1, turn_count + 1))
    return agent


def _text_of(messages: list[dict]) -> str:
    return "\n".join(m.get("content") or "" for m in messages)


def _has_tool_call_id(messages: list[dict], tc_id: str) -> bool:
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if tc.get("id") == tc_id:
                return True
    return False


def _patch_llm(monkeypatch, summary: str = "【摘要】早期几轮：读过若干文件，改了入口"):
    """替换 llm.chat_stream，记录每次请求并返回逐字文本。

    返回的 calls 是 dict 列表：{"kind": "summary"|"main", "messages": [...]}
    - kind=="summary"：请求里带 SUMMARY_SYSTEM_PROMPT（靠"摘要"二字识别）
    - kind=="main"   ：正常主循环请求
    """
    calls: list[dict] = []

    async def fake_chat_stream(messages, tools=None):
        is_summary = any(
            m["role"] == "system" and "摘要" in (m.get("content") or "")
            for m in messages
        )
        calls.append({"kind": "summary" if is_summary else "main", "messages": messages})
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=summary, tool_calls=None)
                )
            ],
            usage=None,
        )

    monkeypatch.setattr(agent_mod.llm, "chat_stream", fake_chat_stream)
    return calls


def _summary_calls(calls: list[dict]) -> list[dict]:
    return [c for c in calls if c["kind"] == "summary"]


def _main_calls(calls: list[dict]) -> list[dict]:
    return [c for c in calls if c["kind"] == "main"]


def _summary_text(call: dict) -> str:
    """摘要请求里非 system 消息的内容拼起来，便于断"喂了哪些历史进去"。"""
    return "\n".join(
        m.get("content") or "" for m in call["messages"] if m["role"] != "system"
    )


# ============================================================
# 1. 触发条件：攒够 lag 才压
# ============================================================

class TestCompressionTrigger:
    async def test_no_compress_below_lag(self, monkeypatch):
        """滑出窗口只 1 轮、lag=4 → 不触发摘要。"""
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 4)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch)

        agent = _new_agent(3)  # 溢出 = 3 - 2 = 1 轮 < lag 4

        assert await agent.compress_history() is False
        assert calls == []
        assert agent._summary is None
        assert agent._compressed_turns == 0

    async def test_compresses_once_when_lag_reached(self, monkeypatch):
        """溢出恰好等于 lag → 压一次，且只压滑出窗口的那几轮。"""
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 2)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch, summary="【摘要】早期两轮都读了文件")

        agent = _new_agent(4)  # 溢出 = 4 - 2 = 2 轮 == lag

        assert await agent.compress_history() is True

        sum_calls = _summary_calls(calls)
        assert len(sum_calls) == 1
        text = _summary_text(sum_calls[0])
        assert "file1.py" in text and "file2.py" in text   # 滑出的第 1/2 轮被喂进摘要
        assert "file3.py" not in text                       # 第 3/4 轮还在窗口里
        assert "file4.py" not in text

        assert agent._summary == "【摘要】早期两轮都读了文件"
        assert agent._compressed_turns == 2

        # 窗口里的第 3/4 轮原样保留，被压掉的第 1/2 轮不在上下文里
        ctx_text = _text_of(agent._build_context())
        assert "file3.py" in ctx_text and "file4.py" in ctx_text
        assert "file1.py" not in ctx_text

    async def test_repeat_call_is_noop(self, monkeypatch):
        """压完后没有新滑出的轮次 → 再调是 no-op，不再多烧一次 API。"""
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 2)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch, summary="【摘要v1】")

        agent = _new_agent(4)
        assert await agent.compress_history() is True
        assert await agent.compress_history() is False

        assert len(_summary_calls(calls)) == 1
        assert agent._summary == "【摘要v1】"
        assert agent._compressed_turns == 2


# ============================================================
# 2. 滚动累积 / 预算 / 失败兜底
# ============================================================

class TestRollingSummary:
    async def test_rolling_summary_accumulates(self, monkeypatch):
        """第二次压缩要把"上一版摘要"一起喂回去，累积成一份。"""
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 2)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch, summary="【摘要v2】")

        agent = _new_agent(4)
        assert await agent.compress_history() is True
        assert agent._summary == "【摘要v2】"
        assert agent._compressed_turns == 2

        agent.messages += _turns(5, 6)
        assert await agent.compress_history() is True

        assert len(_summary_calls(calls)) == 2
        second = _summary_calls(calls)[1]
        assert "【摘要v2】" in _summary_text(second)   # 上一版摘要被喂回（滚动累积）
        assert "file3.py" in _summary_text(second)      # 本次新滑出的是第 3/4 轮
        assert "file5.py" not in _summary_text(second)  # 第 5 轮还在窗口里

        assert agent._compressed_turns == 4

    async def test_summary_capped_by_budget(self, monkeypatch):
        """摘要模型话痨时，按 SUMMARY_MAX_CHARS 截断，别让摘要自己膨胀。"""
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 2)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        monkeypatch.setattr(cfg, "SUMMARY_MAX_CHARS", 40)
        _patch_llm(monkeypatch, summary="【摘要】" + "啰" * 500)

        agent = _new_agent(4)
        assert await agent.compress_history() is True

        assert agent._summary is not None
        assert len(agent._summary) <= 40

    async def test_llm_failure_is_fail_safe(self, monkeypatch):
        """摘要模型挂了不能把整个任务带崩：吞掉异常、不推进游标。"""
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 2)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)

        async def boom(messages, tools=None):
            raise RuntimeError("摘要模型挂了")
            yield  # 让它是 async generator

        monkeypatch.setattr(agent_mod.llm, "chat_stream", boom)

        agent = _new_agent(4)
        assert await agent.compress_history() is False   # 不向外抛

        assert agent._summary is None
        assert agent._compressed_turns == 0
        assert agent_mod._count_turns(agent.messages[1:]) == 4  # 历史没被动过


# ============================================================
# 3. 摘要稳定性：连续压多次也不膨胀
# ============================================================

class TestSummaryStability:
    async def test_summary_not_inflating(self, monkeypatch):
        """连续压 3 次，摘要长度必须稳定：允许 +20% 抖动，绝不能翻倍。

        摘要模型每轮返回"等长"的浓缩要点（模拟听话的模型）。
        若实现是把新结果 `prev + 新结果` 直接拼上去（而不是把旧摘要交回模型
        重写合并），长度会随压缩次数线性膨胀 → 本用例报红。
        """
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 1)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        monkeypatch.setattr(cfg, "SUMMARY_MAX_CHARS", 2000)
        calls = _patch_llm(monkeypatch, summary="【摘要】" + "要" * 200)

        agent = _new_agent(3)          # 溢出 1 轮 == lag → 压第 1 次
        lengths: list[int] = []

        for extra in (None, 4, 5):
            if extra is not None:
                agent.messages += _turn(extra)
            assert await agent.compress_history() is True
            lengths.append(len(agent._summary))

        assert len(_summary_calls(calls)) == 3          # 确实压了 3 次
        assert lengths[-1] <= lengths[0] * 1.2          # +20% 容差内，不许翻倍

    async def test_summary_hard_capped_across_rounds(self, monkeypatch):
        """反向兜底：模型不听话越压越长，也不破 SUMMARY_MAX_CHARS。"""
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 1)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        monkeypatch.setattr(cfg, "SUMMARY_MAX_CHARS", 120)

        async def growing(messages, tools=None):
            # 最坏情况：摘要模型每轮都吐一大坨
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="啰" * 300, tool_calls=None)
                    )
                ],
                usage=None,
            )

        monkeypatch.setattr(agent_mod.llm, "chat_stream", growing)

        agent = _new_agent(3)
        lengths: list[int] = []
        for extra in (None, 4, 5):
            if extra is not None:
                agent.messages += _turn(extra)
            assert await agent.compress_history() is True
            lengths.append(len(agent._summary))

        assert all(n <= 120 for n in lengths)


# ============================================================
# 4. 压缩边界：只压"完整 turn"，尾部 partial turn 必须留窗口
# ============================================================

class TestTurnBoundary:
    async def test_partial_turn_not_compressed(self, monkeypatch):
        """尾部是 user → assistant(tool_calls)（工具还没回）→ 这条 partial turn 留在窗口。

        场景：3 个完整 turn 之后跟一条残缺的当前轮。window=1 时窗口里只剩这条
        partial turn，它前面的完整 turn 才允许被压进摘要。
        """
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 1)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 1)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch, summary="【摘要】前面几轮都读过文件")

        agent = CodingAgent()
        agent.messages = [agent.messages[0]] + _turns(1, 2, 3) + _partial_turn(4)

        assert await agent.compress_history() is True

        req = _summary_calls(calls)[0]

        # partial turn 的两条消息都没进摘要载荷
        assert not any("任务4" in (m.get("content") or "") for m in req["messages"])
        assert not any("改完记得跑测试" in (m.get("content") or "") for m in req["messages"])
        assert not _has_tool_call_id(req["messages"], "call_4")
        assert "python -m pytest" not in _summary_text(req)

        # 压缩载荷里的历史是"完整 turn"：最后一条 assistant 有文本、且不悬空 tool_calls
        asst = [m for m in req["messages"] if m["role"] == "assistant"]
        assert asst, "摘要载荷里应包含完整 turn 的 assistant 消息"
        assert not (asst[-1].get("tool_calls") or [])
        assert "已完成任务3" in (asst[-1].get("content") or "")

        # 边界落在 user 消息上：载荷里最后一个 user 是第 3 轮的开头
        users = [m for m in req["messages"] if m["role"] == "user"]
        assert users
        assert "任务3" in (users[-1].get("content") or "")
        assert "任务4" not in (users[-1].get("content") or "")

        # 上下文里 partial turn 原样保留，且历史首条不是孤儿 tool
        ctx = agent._build_context()
        assert ctx[0]["role"] == "system"
        assert ctx[1]["role"] == "user"
        assert "任务4" in _text_of(ctx)
        assert _has_tool_call_id(ctx, "call_4")
        assert ctx[-1]["role"] == "assistant"
        assert _has_tool_call_id([ctx[-1]], "call_4")
        assert agent._compressed_turns == 3

    async def test_partial_turn_survives_across_repeated_compression(self, monkeypatch):
        """连续压缩也不能把 partial turn 卷进去（它永远在窗口内）。"""
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 1)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch, summary="【摘要】")

        agent = CodingAgent()
        agent.messages = [agent.messages[0]] + _turns(1, 2, 3, 4) + _partial_turn(5)

        assert await agent.compress_history() is True
        assert await agent.compress_history() is False   # 没新滑出的轮 → no-op

        for call in _summary_calls(calls):
            assert not _has_tool_call_id(call["messages"], "call_5")
            assert "任务5" not in _summary_text(call)

        ctx = agent._build_context()
        assert "任务5" in _text_of(ctx)
        assert _has_tool_call_id(ctx, "call_5")

    async def test_compacted_prefix_never_ends_on_tool_call(self, monkeypatch):
        """压缩前沿必须落在 user 边界：被压掉的部分不能以"半截 tool 流"收尾。"""
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 1)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 1)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch, summary="【摘要】")

        agent = CodingAgent()
        agent.messages = [agent.messages[0]] + _turns(1, 2) + _partial_turn(3)
        assert await agent.compress_history() is True

        ctx = agent._build_context()[1:]
        assert ctx and ctx[0]["role"] == "user"


# ============================================================
# 5. 孤儿 tool：摘要注入后窗口首条不得是 role="tool"
# ============================================================

class TestOrphanToolAfterCompression:
    async def test_orphan_tool_after_compression(self, monkeypatch):
        """摘要注入 + 滑窗后，历史首条若是 tool 就是孤儿（其 assistant 已被切掉）。

        孤儿 tool 会在下一次请求里造成 tool_call_id 失配 → 接口 400。
        要求：压缩/切窗时把这种孤儿一并丢弃，且别把它塞进摘要载荷。
        """
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 1)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 1)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch, summary="【摘要】")

        agent = CodingAgent()
        # 轮1、轮2 完整；轮3 只剩 user + tool（assistant(tool_calls) 已丢 → 孤儿）
        agent.messages = [agent.messages[0]] + _turns(1, 2) + [
            {"role": "user", "content": "任务3：继续"},
            {"role": "tool", "tool_call_id": "call_3", "content": "孤儿工具结果"},
        ]

        await agent.compress_history()
        ctx = agent._build_context()
        history = ctx[1:]

        assert ctx[0]["role"] == "system"
        assert not history or history[0]["role"] != "tool"   # ← 核心断言

        # 孤儿 tool 也不该被喂进摘要载荷
        assert len(calls) >= 1
        assert not any(
            m["role"] == "tool" and m.get("tool_call_id") == "call_3"
            for m in _summary_calls(calls)[0]["messages"]
        )


# ============================================================
# 6. 摘要请求的形状
# ============================================================

class TestSummaryRequestShape:
    async def test_prompt_and_payload(self, monkeypatch):
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 2)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch, summary="【摘要】")

        agent = _new_agent(4)
        assert await agent.compress_history() is True

        assert "摘要" in agent_mod.SUMMARY_SYSTEM_PROMPT

        req = _summary_calls(calls)[0]
        system_msgs = [m for m in req["messages"] if m["role"] == "system"]
        assert any("摘要" in (m.get("content") or "") for m in system_msgs)

        text = _summary_text(req)
        assert "file1.py" in text and "file2.py" in text       # 只压滑出的部分
        assert "file3.py" not in text and "file4.py" not in text
        assert "【摘要】" not in text                          # 首次压缩无旧摘要

    async def test_does_not_use_main_system_prompt(self, monkeypatch):
        """摘要请求要用自己的 prompt，不能复用主对话那段工作规范。"""
        from agent.agent import SYSTEM_PROMPT

        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 2)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch, summary="【摘要】")

        agent = _new_agent(4)
        assert await agent.compress_history() is True

        req = _summary_calls(calls)[0]
        system_msgs = [m.get("content") for m in req["messages"] if m["role"] == "system"]
        assert system_msgs
        assert all(c != SYSTEM_PROMPT for c in system_msgs)


# ============================================================
# 7. 组装上下文 _build_context（同步，无需模型）
# ============================================================

class TestBuildContext:
    def test_short_history_keeps_old_behaviour(self):
        """历史很短、没摘要 → 与旧行为一致：原样返回。"""
        agent = _new_agent(3)
        assert agent._build_context() == agent.messages

    def test_summary_injected_once_right_after_system(self):
        agent = _new_agent(3)
        agent._summary = "【摘要】早期读了 file1.py"

        ctx = agent._build_context()

        assert ctx[0]["role"] == "system"
        assert "【摘要】早期读了 file1.py" in (ctx[1].get("content") or "")
        hits = [i for i, m in enumerate(ctx) if "早期读了 file1.py" in (m.get("content") or "")]
        assert hits == [1]

    def test_compressed_range_absent_from_context(self):
        agent = _new_agent(6)
        agent._summary = "【摘要】前四轮都读过文件"
        agent._compressed_turns = 4

        ctx = agent._build_context()
        text = _text_of(ctx)

        for n in (1, 2, 3, 4):
            assert f"file{n}.py" not in text   # 已折进摘要的轮次不再出现在明文里
        assert "file5.py" in text
        assert "file6.py" in text

    def test_window_applies_to_unsummarized_part(self, monkeypatch):
        """窗口只作用于"未压缩"的那部分历史。"""
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 1)
        agent = _new_agent(6)
        agent._summary = "【摘要】"
        agent._compressed_turns = 5  # 只剩第 6 轮未压缩

        ctx = agent._build_context()
        text = _text_of(ctx)

        assert "file6.py" in text
        assert "file5.py" not in text
        assert "【摘要】" in text

    def test_hard_cap_respects_limit(self, monkeypatch):
        """Q13 P0: _build_context 不再硬砍（保护 prefix cache）；硬砍移到 _maybe_trim_after_task。"""
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 6)
        agent = _new_agent(6)  # 6 轮 = 24 条消息

        # _build_context 不裁剪——单任务内只增不减（保护 prefix cache）
        ctx = agent._build_context()
        assert len(ctx) - 1 > 6  # 全部历史都在，没被硬砍

        # 任务结束后才裁剪
        agent._maybe_trim_after_task()
        ctx_after = agent._build_context()
        assert len(ctx_after) - 1 <= 6  # 裁剪后只剩 6 条

    def test_hard_cap_does_not_count_summary_message(self, monkeypatch):
        """摘要消息不占 MAX_HISTORY_MESSAGES 的预算。"""
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 4)
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 1)
        agent = _new_agent(6)
        agent._summary = "【摘要】占了不该占的预算"
        agent._compressed_turns = 5

        ctx = agent._build_context()

        assert (ctx[1].get("content") or "").startswith("【摘要】")  # 摘要没被硬砍吃掉
        assert len(ctx[2:]) <= cfg.MAX_HISTORY_MESSAGES

    def test_current_turn_survives(self, monkeypatch):
        """硬砍后当前这一轮的 user 消息必须还在，否则模型不知道要干嘛。"""
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 4)
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 4)
        agent = _new_agent(6)

        ctx = agent._build_context()
        assert ctx[1]["role"] == "user"
        assert "file6.py" in _text_of(ctx)

    def test_history_never_starts_with_tool(self, monkeypatch):
        """任意硬砍阈值下，历史首条都不能是孤儿 tool 消息（否则接口报错）。"""
        for cap in range(1, 9):
            monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", cap)
            agent = _new_agent(4)
            history = agent._build_context()[1:]
            assert not history or history[0]["role"] != "tool", f"cap={cap} 首条是孤儿 tool"

    def test_does_not_mutate_messages(self):
        agent = _new_agent(6)
        before = [dict(m) for m in agent.messages]
        agent._summary = "【摘要】x"
        agent._compressed_turns = 5

        agent._build_context()

        assert agent.messages == before
        assert agent._summary == "【摘要】x"


# ============================================================
# 8. 与主循环集成：压缩发生在"请求模型之前"
# ============================================================

class TestRunTaskIntegration:
    async def test_compresses_before_model_call(self, monkeypatch):
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 2)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)
        calls = _patch_llm(monkeypatch, summary="【摘要】")

        agent = _new_agent(4)
        await agent.run_task("再改一处")   # 总 5 轮，溢出 3 轮 > lag 2 → 先压后调

        assert len(_summary_calls(calls)) >= 1
        assert len(_main_calls(calls)) >= 1
        assert "file1.py" in _summary_text(_summary_calls(calls)[0])

        main_text = _text_of(_main_calls(calls)[0]["messages"])
        assert "【摘要】" in main_text        # 摘要进了主上下文
        assert "file1.py" not in main_text   # 早期的第 1 轮被压掉了
        assert "再改一处" in main_text        # 新任务在

    async def test_summary_failure_does_not_break_task(self, monkeypatch):
        monkeypatch.setattr(cfg, "HISTORY_WINDOW_TURNS", 2)
        monkeypatch.setattr(cfg, "HISTORY_COMPRESS_LAG", 2)
        monkeypatch.setattr(cfg, "MAX_HISTORY_MESSAGES", 200)

        seen: list[str] = []

        async def fake_chat_stream(messages, tools=None):
            is_summary = any(
                m["role"] == "system" and "摘要" in (m.get("content") or "")
                for m in messages
            )
            seen.append("summary" if is_summary else "main")
            if is_summary:
                raise RuntimeError("摘要服务 502")
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="好的", tool_calls=None)
                    )
                ],
                usage=None,
            )

        monkeypatch.setattr(agent_mod.llm, "chat_stream", fake_chat_stream)

        agent = _new_agent(4)
        await agent.run_task("再改一处")   # 不能因为摘要挂了就抛出去

        assert seen[0] == "summary"   # 先尝试压缩
        assert "main" in seen         # 摘要挂了，主任务照常发
        assert agent._summary is None
        assert agent._compressed_turns == 0


# ============================================================
# 9. 模块级小工具的独立契约
# ============================================================

class TestHelpers:
    def test_count_turns_counts_user_messages(self):
        assert agent_mod._count_turns(_turns(1, 2, 3)) == 3
        assert agent_mod._count_turns(_turns(1, tools=False)) == 1
        assert agent_mod._count_turns(_partial_turn(9)) == 1   # partial 也算一轮

    def test_window_start_lands_on_user(self):
        messages = _turns(1, 2, 3)
        idx = agent_mod._window_start(messages, 2)
        assert idx == len(_turns(1))   # window=2 保留最后 2 轮 → 起点落在第 2 轮的 user
        assert messages[idx]["role"] == "user"
        assert messages[idx]["content"].startswith("任务2")

    def test_window_start_with_partial_tail(self):
        messages = _turns(1, 2) + _partial_turn(3)
        idx = agent_mod._window_start(messages, 1)
        assert messages[idx]["role"] == "user"
        assert messages[idx]["content"].startswith("任务3")

    def test_summary_message_shape(self):
        msg = agent_mod._summary_message("【摘要】xyz")
        assert msg["role"] == "user"
        assert "【摘要】xyz" in msg["content"]

    def test_cap_summary_truncates(self):
        out = agent_mod._cap_summary("x" * 100, max_chars=10)
        assert len(out) <= 10


# ============================================================
# 10. 配置自洽性（不需要 agent 也能跑）
# ============================================================

class TestConfigConsistency:
    def test_lag_at_least_one(self):
        assert cfg.HISTORY_COMPRESS_LAG >= 1

    def test_window_at_least_one(self):
        assert cfg.HISTORY_WINDOW_TURNS >= 1

    def test_hard_cap_can_hold_the_window(self):
        # 一轮少说 2 条消息（user + assistant），硬砍阈值不该比窗口本身还紧
        assert cfg.MAX_HISTORY_MESSAGES >= cfg.HISTORY_WINDOW_TURNS * 2
