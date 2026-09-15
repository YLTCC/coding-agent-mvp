"""
test_instance_isolation.py —— per-session 实例化改造（note3 §4.6 前置）验收测试 [I1–I5]

本文件是 §4.6 前置改造的**验收规格**，不是"168 还绿"那种底线检查：
两个 Agent（不同 workspace / approval_callback / event_sink / logger）**并发**跑，
必须互不干扰。

改造前必然失败（CodingAgent 还不支持 config/llm/ctx 注入，tools 层还是模块级单例、
tools 读的还是全局 config.WORKSPACE），改造后必须全绿。

跑法：python -m pytest test_instance_isolation.py -q

  I1  两实例的会话日志零交叉（各自的 user/approval 只进各自的文件）
  I2  A 的 approval_callback 只被 A 的审批调用（B 同理）
  I3  A 的 workspace 只碰 A 的目录树（写/读/列出/命令 cwd 四个通道）
  I4  两实例的 system prompt 里分别是各自的 workspace（提示词与沙箱同源）
  I5  session_start.workspace == 该实例的 ctx.workspace（日志不说谎）

非空验证（note3 §7.3 对 T27/T28 的同款要求）：把 tools._workspace() 改成永远
回落到全局 config.default_workspace，I3 必须变红。不红 = 断言是空的。
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as main_mod
from agent.config import config as cfg
from agent.session_log import read_records


# ============================================================
# 测试替身：按脚本返回 tool_calls 的假模型（每实例一份，互不共享）
# ============================================================

def _text_chunk(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))],
        usage=None,
    )


def _tool_chunk(call_id: str, name: str, args: dict):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id=call_id,
                            function=SimpleNamespace(
                                name=name, arguments=json.dumps(args)
                            ),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )


class _ScriptedLLM:
    """每轮请求消费一条脚本；脚本用尽后只回一句文本（结束任务）。

    它是"每实例一份"的——这正是 LLMClient 实例化要证明的事：
    两个实例的模型客户端互不覆盖。
    """

    def __init__(self, script: list[list], tag: str):
        self._script = list(script)
        self.tag = tag
        self.requests: list[list[dict]] = []

    async def chat_stream(self, messages, tools=None):
        self.requests.append(list(messages))
        chunks = self._script.pop(0) if self._script else [_text_chunk("结束")]
        for chunk in chunks:
            yield chunk


def _script(tag: str) -> list[list]:
    """一套脚本，四个工具通道各走一遍（写 / 读对方文件 / 列目录 / 命令 cwd）。"""
    other = "B_own.txt" if tag == "A" else "A_own.txt"
    return [
        [_tool_chunk(f"{tag}-c1", "write_file", {"path": f"{tag}_own.txt",
                                                 "content": f"{tag}-content"})],
        [_tool_chunk(f"{tag}-c2", "read_file", {"path": other})],
        [_tool_chunk(f"{tag}-c3", "list_dir", {"path": "."})],
        [_tool_chunk(f"{tag}-c4", "run_command", {"command": "echo %CD%"})],
        [_text_chunk(f"{tag} 完成")],
    ]


async def _run_both(agent_a, agent_b) -> None:
    await asyncio.gather(
        agent_a.run_task("A 的任务"),
        agent_b.run_task("B 的任务"),
    )


def _tools_of(records: list[dict]) -> list[dict]:
    return [r for r in records if r["type"] == "tool"]


def _path_in(text: str, path: Path) -> bool:
    """宽松比对：忽略分隔符方向与大小写（Windows 路径大小写不敏感）。"""
    return path.as_posix().lower() in text.replace("\\", "/").lower()


# ============================================================
# fixture：两个会话并发跑完，返回全部现场
# ============================================================

@pytest.fixture
def two_agents(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "SESSION_LOG_ENABLED", True)
    monkeypatch.setattr(cfg, "SESSION_DIR", tmp_path / "sessions")
    # AUTO_APPROVE 必须是 False：否则 _ask_approval 走 "auto" 分支，
    # ctx 里的 approval_callback 一次都不会被调用，I2 就成了空断言
    monkeypatch.setattr(cfg, "AUTO_APPROVE", False)

    ws_a = tmp_path / "a"
    ws_b = tmp_path / "b"
    ws_a.mkdir()
    ws_b.mkdir()

    calls: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}

    def cb_a(action: str, detail: str) -> bool:
        calls["a"].append((action, detail))
        return True

    def cb_b(action: str, detail: str) -> bool:
        calls["b"].append((action, detail))
        return True

    agent_a, log_a = main_mod.build_session(
        workspace=ws_a, approval_callback=cb_a,
        llm_client=_ScriptedLLM(_script("A"), "A"), config=cfg,
    )
    agent_b, log_b = main_mod.build_session(
        workspace=ws_b, approval_callback=cb_b,
        llm_client=_ScriptedLLM(_script("B"), "B"), config=cfg,
    )

    asyncio.run(_run_both(agent_a, agent_b))   # 并发：真去抢事件循环与线程池
    log_a.close()
    log_b.close()

    return SimpleNamespace(
        ws_a=ws_a, ws_b=ws_b,
        agent_a=agent_a, agent_b=agent_b,
        log_a=log_a, log_b=log_b,
        calls=calls,
        rec_a=read_records(log_a.path),
        rec_b=read_records(log_b.path),
    )


# ============================================================
# I1：会话日志零交叉
# ============================================================

def test_I1_logs_do_not_cross(two_agents):
    t = two_agents

    # 每个文件里只有自己的 session_id
    assert {r["session_id"] for r in t.rec_a} == {t.log_a.session_id}
    assert {r["session_id"] for r in t.rec_b} == {t.log_b.session_id}

    # 用户原话不串台
    assert any(r.get("content") == "A 的任务" for r in t.rec_a)
    assert not any(r.get("content") == "B 的任务" for r in t.rec_a)
    assert any(r.get("content") == "B 的任务" for r in t.rec_b)
    assert not any(r.get("content") == "A 的任务" for r in t.rec_b)

    # 审批事件（note3 §2.4）也不得写进别人的会话
    appr_a = [r for r in t.rec_a if r["type"] == "approval"]
    appr_b = [r for r in t.rec_b if r["type"] == "approval"]
    assert appr_a and appr_b, "审批事件没落盘，I1 失去意义"
    assert all(_path_in(r["detail"], t.ws_a) for r in appr_a)
    assert not any(_path_in(r["detail"], t.ws_b) for r in appr_a)
    assert all(_path_in(r["detail"], t.ws_b) for r in appr_b)
    assert not any(_path_in(r["detail"], t.ws_a) for r in appr_b)


# ============================================================
# I2：审批回调不串台
# ============================================================

def test_I2_approval_callbacks_do_not_cross(two_agents):
    t = two_agents

    # 每个实例各 2 次审批（write_file + run_command）
    assert len(t.calls["a"]) == 2, f"A 收到的审批是 {t.calls['a']}"
    assert len(t.calls["b"]) == 2, f"B 收到的审批是 {t.calls['b']}"

    writes_a = [d for a, d in t.calls["a"] if a == "写入文件"]
    writes_b = [d for a, d in t.calls["b"] if a == "写入文件"]
    assert writes_a and all(_path_in(d, t.ws_a) for d in writes_a)
    assert writes_b and all(_path_in(d, t.ws_b) for d in writes_b)

    # 关键断言：A 的 callback 里绝不出现 B 的会话身份
    assert not any(_path_in(d, t.ws_b) for _, d in t.calls["a"])
    assert not any(_path_in(d, t.ws_a) for _, d in t.calls["b"])


# ============================================================
# I3：workspace 隔离（四个通道）
# ============================================================

def test_I3_workspace_isolation(two_agents):
    t = two_agents

    # 通道 1：写文件落进各自的沙箱
    assert (t.ws_a / "A_own.txt").read_text(encoding="utf-8") == "A-content"
    assert (t.ws_b / "B_own.txt").read_text(encoding="utf-8") == "B-content"
    assert not (t.ws_a / "B_own.txt").exists(), "A 写到了别人的沙箱"
    assert not (t.ws_b / "A_own.txt").exists(), "B 写到了别人的沙箱"

    tools_a = _tools_of(t.rec_a)
    tools_b = _tools_of(t.rec_b)
    assert len(tools_a) == 4 and len(tools_b) == 4

    # 通道 2：读对方的文件 —— 必须在自己的沙箱里"不存在"
    assert "B_own.txt" in tools_a[1]["content"] and "不存在" in tools_a[1]["content"]
    assert "A_own.txt" in tools_b[1]["content"] and "不存在" in tools_b[1]["content"]

    # 通道 3：list_dir 只看得见自己的目录树
    assert "A_own.txt" in tools_a[2]["content"]
    assert "B_own.txt" not in tools_a[2]["content"]
    assert "B_own.txt" in tools_b[2]["content"]
    assert "A_own.txt" not in tools_b[2]["content"]

    # 通道 4：run_command 的 cwd 是自己的沙箱（写的是全局就成了别人的目录）
    assert _path_in(tools_a[3]["content"], t.ws_a)
    assert not _path_in(tools_a[3]["content"], t.ws_b)
    assert _path_in(tools_b[3]["content"], t.ws_b)
    assert not _path_in(tools_b[3]["content"], t.ws_a)


# ============================================================
# I4：提示词与沙箱同源
# ============================================================

def test_I4_system_prompt_is_per_session(two_agents):
    t = two_agents
    prompt_a = t.agent_a.messages[0]["content"]
    prompt_b = t.agent_b.messages[0]["content"]

    assert _path_in(prompt_a, t.ws_a), "A 的提示词没说自己沙箱在哪"
    assert not _path_in(prompt_a, t.ws_b)
    assert _path_in(prompt_b, t.ws_b), "B 的提示词没说自己沙箱在哪"
    assert not _path_in(prompt_b, t.ws_a)

    # I3 的沙箱 + I4 的提示词必须是**同一个值**（§5.2 第 3 类失败形态）
    assert _path_in(prompt_a, Path(str(t.agent_a.ctx.workspace)))
    assert prompt_a != prompt_b


# ============================================================
# I5：日志不说谎
# ============================================================

def test_I5_session_start_records_session_workspace(two_agents):
    t = two_agents

    start_a = [r for r in t.rec_a if r["type"] == "session_start"][0]
    start_b = [r for r in t.rec_b if r["type"] == "session_start"][0]

    assert start_a["workspace"] == str(t.agent_a.ctx.workspace)
    assert start_b["workspace"] == str(t.agent_b.ctx.workspace)
    assert start_a["workspace"] == str(t.ws_a.resolve())
    assert start_b["workspace"] == str(t.ws_b.resolve())


# ============================================================
# 兼容性：模块级 SYSTEM_PROMPT 必须仍是"默认 workspace"的那一份
# ============================================================

def test_default_system_prompt_still_matches_module_constant():
    """main.py / 老测试都 import SYSTEM_PROMPT；它必须等于默认沙箱的提示词。"""
    from agent.agent import SYSTEM_PROMPT, build_system_prompt

    assert SYSTEM_PROMPT == build_system_prompt(cfg.default_workspace, cfg.LLM_MODEL)
