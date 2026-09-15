"""conftest.py —— web 验收测试的公共夹具与替身（阶段三·b）

放在仓库根目录是为了让 pytest 自动发现。**没有任何 autouse 夹具**，
所以对既有的 181 个用例零影响。

这里只放"造场景"的东西（假模型、假 chunk、临时配置），不放断言——
断言留在各自的 test_web_*.py 里，方便按 W-T 编号逐条对照 webtodolist.md。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

# ⚠️ 这里**不能**在模块级 import agent.config：conftest 会在所有测试模块之前被加载，
# 而 test_tools.py 依赖"先设 os.environ['AUTO_APPROVE']，再 import agent.config"
# （模块级 `config = Config()` 在 import 期求值）。提前 import 会把 AUTO_APPROVE
# 冻在 False 上，test_tools 的一批用例就会去真的调 input()。
# 所以 Config 只在夹具内部按需 import。


# ============================================================
# 假模型：按脚本吐 chunk（与 test_instance_isolation.py 同款手法）
# ============================================================

def text_chunk(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))],
        usage=None,
    )


def tool_chunk(call_id: str, name: str, args: dict | None, raw: str | None = None):
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
                                name=name,
                                arguments=raw if raw is not None else json.dumps(args),
                            ),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )


class ScriptedLLM:
    """每轮请求消费一条脚本；脚本用尽后只回一句文本（结束任务）。

    它是**每实例一份**的——这正是 LLMClient 实例化要证明的事：
    两个 session 的模型客户端互不覆盖。
    """

    def __init__(self, script: list | None = None):
        self._script = list(script or [])

    async def chat_stream(self, messages, tools=None):
        chunks = self._script.pop(0) if self._script else [text_chunk("结束")]
        for chunk in chunks:
            yield chunk


class GatedLLM:
    """第一轮卡在 gate 上（用来制造"任务正在跑"的窗口，测 409 守卫）。"""

    def __init__(self, gate, text: str = "任务完成"):
        self.gate = gate
        self.text = text

    async def chat_stream(self, messages, tools=None):
        import asyncio

        await asyncio.to_thread(self.gate.wait, 15)
        yield text_chunk(self.text)


# ============================================================
# 夹具
# ============================================================

@pytest.fixture
def web_cfg(tmp_path, monkeypatch):
    """一份 web 用的配置：工作区根与审计目录**分开**（决策 K 的前提）。

    `AGENT_SESSION_DIR` 必须显式设置：`prepare_web_config` 只在"用户没显式配"
    时才去 %LOCALAPPDATA%——测试里显式配成 tmp 下的目录，既满足"在沙箱之外"
    的断言，也不会真的往用户的 AppData 里写东西。

    `AUTO_APPROVE` 必须显式钉成 False：别的测试文件会往 `os.environ` 里塞
    `AUTO_APPROVE=true`（test_tools.py 就是这么干的），不钉死的话
    `_ask_approval` 走 "auto" 分支、审批回调一次都不被调用 ——
    W-T1 / W-T3 那一堆断言会**静默变成空断言**。
    """
    from agent.config import Config

    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("AGENT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("AUTO_APPROVE", "false")
    cfg = Config()
    cfg.WEB_WORKSPACE_ROOT = root
    cfg.AUTO_APPROVE = False
    cfg.SESSION_LOG_ENABLED = True
    cfg.WEB_APPROVAL_TIMEOUT = 10.0
    cfg.WEB_QUEUE_MAXSIZE = 256
    return cfg


# ============================================================
# SSE 帧解析（测试侧）
# ============================================================

def parse_frame(frame: str) -> dict:
    """把一帧 SSE 拆成 {"kind", "id", "data", "raw"}。"""
    out: dict = {"kind": None, "id": None, "data": None, "raw": frame}
    for line in frame.split("\n"):
        if line.startswith("event: "):
            out["kind"] = line[len("event: "):]
        elif line.startswith("id: "):
            out["id"] = int(line[len("id: "):])
        elif line.startswith("data: "):
            out["data"] = json.loads(line[len("data: "):])
    return out
